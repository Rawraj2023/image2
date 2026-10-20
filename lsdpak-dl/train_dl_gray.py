"""
LS-DPAK-DL grayscale -- training.

Trains NKGGray, HistogramDiscriminatorGray and UNetDenoiserGray from
scratch on grayscale data. Mirrors train.py's three phases exactly (same
epoch defaults, same loss combination, same optimiser settings) but on
single-channel images and a 7-byte hb, so the only thing that differs
between the RGB and grayscale checkpoints is the data they were shown --
not the training recipe.

Neither training phase requires the chaotic 90,000-step ODE: Phase 1/2's
dataset is just SHA-256 hashes of image patches, and Phase 3's dataset is
synthetically corrupted plaintext. So this runs in well under a minute on
CPU, unlike anything touching the Lorenz/Rossler keystream.

Usage:
    python train_dl_gray.py [--phase {1,2,3,all}] [--epochs1 N] ...

Outputs (saved in ./checkpoints_gray/):
    nkg_gray_pretrained.pt
    nkg_gray_gan.pt
    discriminator_gray_gan.pt
    denoiser_gray.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_V5 = os.path.join(_HERE, "..", "lsdpak-v5-grayscale")
sys.path.insert(0, _HERE)
sys.path.insert(0, _V5)

import lsdpak_gray_cpu as gray
from lsdpak_dl_gray_model import (
    NKGGray, HistogramDiscriminatorGray, UNetDenoiserGray,
    uniformity_loss_gray, correlation_loss_gray, DEVICE, IMG_SIZE,
)
from lsdpak_dl import SSIMLoss   # channel-agnostic, reused as-is

CKPT_DIR = os.path.join(_HERE, "checkpoints_gray")
os.makedirs(CKPT_DIR, exist_ok=True)

IMAGE_NAMES = gray.IMAGE_NAMES   # aerial, lena, couple, clock, man


# ============================================================================
# DATASETS
# ============================================================================

class KeystreamDatasetGray(Dataset):
    """7-byte hash tensors from full images and random patches."""

    def __init__(self, images: list, n_patches: int = 150):
        self.samples = []
        for img in images:
            H, W = img.shape[:2]
            hb = gray.image_hash_bytes(img)
            self.samples.append(NKGGray.hash_to_tensor(hb).squeeze(0))
            ph = pw = min(H, 64)
            for _ in range(n_patches):
                r = np.random.randint(0, H - ph + 1)
                c = np.random.randint(0, W - pw + 1)
                patch = img[r:r + ph, c:c + pw]
                hb_p = gray.image_hash_bytes(patch)
                self.samples.append(NKGGray.hash_to_tensor(hb_p).squeeze(0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DenoisingDatasetGray(Dataset):
    """(corrupted_plain, clean_plain) pairs, single-channel, same corruption
    recipe as train.py's DenoisingDataset (S&P / crop / both)."""

    def __init__(self, images: list, n_aug: int = 60):
        self.pairs = []
        for img in images:
            clean = img.astype(np.float32) / 255.0   # (H,W) in [0,1]
            for _ in range(n_aug):
                mode = np.random.choice(["sp", "crop", "both"])
                noisy = clean.copy()
                if mode in ("sp", "both"):
                    noisy = self._salt_pepper(noisy, density=np.random.uniform(0.02, 0.15))
                if mode in ("crop", "both"):
                    noisy = self._random_crop(noisy, max_frac=0.25)
                self.pairs.append((noisy, clean))
        np.random.shuffle(self.pairs)

    @staticmethod
    def _salt_pepper(img, density):
        out = img.copy()
        mask = np.random.random(img.shape)
        out[mask < density / 2] = 0.0
        out[(mask >= density / 2) & (mask < density)] = 1.0
        return out

    @staticmethod
    def _random_crop(img, max_frac):
        out = img.copy()
        H, W = img.shape
        h = np.random.randint(1, max(2, int(H * max_frac)))
        w = np.random.randint(1, max(2, int(W * max_frac)))
        r = np.random.randint(0, H - h + 1)
        c = np.random.randint(0, W - w + 1)
        out[r:r + h, c:c + w] = 0.5
        return out

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy, clean = self.pairs[idx]
        t_noisy = torch.from_numpy(noisy[None, :, :]).to(DEVICE)   # (1,H,W)
        t_clean = torch.from_numpy(clean[None, :, :]).to(DEVICE)
        return t_noisy, t_clean


# ============================================================================
# PHASE 1 -- NKG uniformity pre-training
# ============================================================================

def train_phase1(nkg, loader, epochs=15, lr=1e-3):
    print(f"\n{'='*60}\nPHASE 1 -- NKGGray Uniformity Pre-training "
          f"(epochs={epochs}, lr={lr})\n{'='*60}")
    opt = optim.Adam(nkg.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    nkg.train()
    for ep in range(1, epochs + 1):
        ep_unif = ep_corr = 0.0
        for hb_batch in loader:
            hb_batch = hb_batch.to(DEVICE)
            opt.zero_grad()
            ks = nkg(hb_batch)          # (B, 65536) -- one call, one channel
            loss_u = uniformity_loss_gray(ks)
            loss_c = correlation_loss_gray(ks)
            loss = loss_u + 0.1 * loss_c
            loss.backward()
            nn.utils.clip_grad_norm_(nkg.parameters(), 1.0)
            opt.step()
            ep_unif += loss_u.item()
            ep_corr += loss_c.item()
        scheduler.step()
        n_b = len(loader)
        print(f"  Epoch {ep:3d}/{epochs}  unif={ep_unif/n_b:.6f}  "
              f"corr={ep_corr/n_b:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")
    path = os.path.join(CKPT_DIR, "nkg_gray_pretrained.pt")
    torch.save(nkg.state_dict(), path)
    print(f"\n  Saved -> {path}")
    return nkg


# ============================================================================
# PHASE 2 -- GAN adversarial fine-tuning
# ============================================================================

def train_phase2(nkg, disc, loader, epochs=25, lr_g=2e-4 / 5, lr_d=2e-4 / 5,
                 lambda_unif=0.5):
    print(f"\n{'='*60}\nPHASE 2 -- NKGGray GAN Fine-tuning "
          f"(epochs={epochs})\n{'='*60}")
    opt_g = optim.Adam(nkg.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    nkg.train()
    disc.train()

    for ep in range(1, epochs + 1):
        ep_d = ep_g = 0.0
        for hb_batch in loader:
            hb_batch = hb_batch.to(DEVICE)
            B = hb_batch.shape[0]

            with torch.no_grad():
                ks_fake = nkg(hb_batch)
            hist_fake = HistogramDiscriminatorGray.keystream_to_hist(ks_fake)
            hist_real = HistogramDiscriminatorGray.uniform_hist(B, device=DEVICE)

            opt_d.zero_grad()
            loss_d_real = bce(disc(hist_real), torch.ones(B, 1, device=DEVICE))
            loss_d_fake = bce(disc(hist_fake.detach()), torch.zeros(B, 1, device=DEVICE))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            ks_g = nkg(hb_batch)
            hist_g = HistogramDiscriminatorGray.keystream_to_hist(ks_g)
            loss_g_adv = bce(disc(hist_g), torch.ones(B, 1, device=DEVICE))
            loss_g_unif = uniformity_loss_gray(ks_g)
            loss_g = loss_g_adv + lambda_unif * loss_g_unif
            loss_g.backward()
            nn.utils.clip_grad_norm_(nkg.parameters(), 1.0)
            opt_g.step()

            ep_d += loss_d.item()
            ep_g += loss_g.item()
        n_b = len(loader)
        print(f"  Epoch {ep:3d}/{epochs}  loss_D={ep_d/n_b:.5f}  loss_G={ep_g/n_b:.5f}")

    path_nkg = os.path.join(CKPT_DIR, "nkg_gray_gan.pt")
    path_disc = os.path.join(CKPT_DIR, "discriminator_gray_gan.pt")
    torch.save(nkg.state_dict(), path_nkg)
    torch.save(disc.state_dict(), path_disc)
    print(f"\n  Saved -> {path_nkg}\n  Saved -> {path_disc}")
    return nkg, disc


# ============================================================================
# PHASE 3 -- U-Net denoiser
# ============================================================================

def train_phase3(denoiser, loader, epochs=35, lr=1e-3):
    print(f"\n{'='*60}\nPHASE 3 -- UNetDenoiserGray Training "
          f"(epochs={epochs}, lr={lr})\n{'='*60}")
    opt = optim.Adam(denoiser.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    ssim_loss = SSIMLoss().to(DEVICE)
    l1_loss = nn.L1Loss()
    denoiser.train()

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for noisy, clean in loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            opt.zero_grad()
            pred = denoiser(noisy)
            loss = 0.5 * l1_loss(pred, clean) + 0.5 * ssim_loss(pred, clean)
            loss.backward()
            nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(loader)
        scheduler.step(avg)
        if ep % 5 == 0 or ep == epochs:
            print(f"  Epoch {ep:3d}/{epochs}  loss={avg:.6f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    path = os.path.join(CKPT_DIR, "denoiser_gray.pt")
    torch.save(denoiser.state_dict(), path)
    print(f"\n  Saved -> {path}")
    return denoiser


# ============================================================================
# MAIN
# ============================================================================

def load_images():
    images = []
    for name in IMAGE_NAMES:
        img = gray.load_image_gray(name)
        images.append(img)
        print(f"  Loaded {name:10s}  shape={img.shape}  mode=grayscale")
    return images


def main():
    p = argparse.ArgumentParser(description="LS-DPAK-DL grayscale training")
    p.add_argument("--phase", choices=["1", "2", "3", "all"], default="all")
    p.add_argument("--epochs1", type=int, default=15)
    p.add_argument("--epochs2", type=int, default=25)
    p.add_argument("--epochs3", type=int, default=35)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patches", type=int, default=150)
    p.add_argument("--aug", type=int, default=60)
    args = p.parse_args()

    print("\nLS-DPAK-DL grayscale -- Training")
    print(f"  Device: {DEVICE}   Phase: {args.phase}")
    print("\nLoading images ...")
    images = load_images()

    nkg = NKGGray().to(DEVICE)
    disc = HistogramDiscriminatorGray().to(DEVICE)
    denoiser = UNetDenoiserGray().to(DEVICE)

    def _n(m):
        return sum(x.numel() for x in m.parameters() if x.requires_grad)
    print(f"\n  NKGGray params          : {_n(nkg):,}")
    print(f"  Discriminator params    : {_n(disc):,}")
    print(f"  Denoiser params         : {_n(denoiser):,}")

    run_all = args.phase == "all"

    if run_all or args.phase == "1":
        ds = KeystreamDatasetGray(images, n_patches=args.patches)
        ld = DataLoader(ds, batch_size=args.batch, shuffle=True)
        print(f"\n  Keystream dataset size: {len(ds)}")
        nkg = train_phase1(nkg, ld, epochs=args.epochs1, lr=args.lr)
    elif args.phase in ("2", "3"):
        ckpt = os.path.join(CKPT_DIR, "nkg_gray_pretrained.pt")
        if os.path.exists(ckpt):
            nkg.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print(f"  Loaded Phase 1 NKGGray from {ckpt}")

    if run_all or args.phase == "2":
        ds = KeystreamDatasetGray(images, n_patches=args.patches)
        ld = DataLoader(ds, batch_size=args.batch, shuffle=True)
        print(f"\n  Keystream dataset size: {len(ds)}")
        nkg, disc = train_phase2(nkg, disc, ld, epochs=args.epochs2,
                                 lr_g=args.lr / 5, lr_d=args.lr / 5)
    elif args.phase == "3":
        p2n = os.path.join(CKPT_DIR, "nkg_gray_gan.pt")
        p2d = os.path.join(CKPT_DIR, "discriminator_gray_gan.pt")
        if os.path.exists(p2n):
            nkg.load_state_dict(torch.load(p2n, map_location=DEVICE))
        if os.path.exists(p2d):
            disc.load_state_dict(torch.load(p2d, map_location=DEVICE))

    if run_all or args.phase == "3":
        ds = DenoisingDatasetGray(images, n_aug=args.aug)
        ld = DataLoader(ds, batch_size=4, shuffle=True)
        print(f"\n  Denoising dataset size: {len(ds)}")
        denoiser = train_phase3(denoiser, ld, epochs=args.epochs3, lr=args.lr)

    print("\n\nAll grayscale training complete.")
    print(f"Checkpoints saved in: {CKPT_DIR}")


if __name__ == "__main__":
    main()
