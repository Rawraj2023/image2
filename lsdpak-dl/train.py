"""
LS-DPAK-DL  --  Training Script
================================

Three-phase training:

  Phase 1  NKG Pre-training
           Minimise uniformity_loss + correlation_loss so that the NKG
           produces byte-uniform, uncorrelated keystreams WITHOUT adversarial
           pressure.  Fast convergence (~200 steps).

  Phase 2  GAN Adversarial Fine-tuning
           Alternating Discriminator / NKG updates:
             D   : trained to output 1 for uniform noise, 0 for NKG keystreams
             NKG : trained to fool D (output -> 1) + keep uniformity
           Produces maximally uniform keystreams resistant to histogram analysis.

  Phase 3  U-Net Denoiser Training (supervised)
           Pairs (corrupted_decrypted_plain, clean_plain) from augmented images.
           Loss: 0.5*L1 + 0.5*SSIM.
           Addresses T9/T10 weakness (salt-and-pepper / crop attacks).

Usage:
    python train.py [--phase {1,2,3,all}] [--epochs N] [--lr LR]

Outputs (saved in ./checkpoints/):
    nkg_pretrained.pt      -- after Phase 1
    nkg_gan.pt             -- after Phase 2
    discriminator_gan.pt   -- after Phase 2
    denoiser.pt            -- after Phase 3
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
_V3    = os.path.join(_HERE, "..", "lsdpak-v3-rgb")
sys.path.insert(0, _V3)
sys.path.insert(0, _HERE)

from lsdpak_v3_rgb import (
    load_image_rgb, image_hash_bytes_rgb, adaptive_ics_rgb,
    build_D, build_D3, build_D5,
    scramble3, unscramble3,
    shuffle_blocks_full, unshuffle_blocks_full,
    interchannel_shuffle, interchannel_unshuffle,
    bidir_encrypt_joint, bidir_decrypt_joint,
    IMG_SIZE, MASTER,
)
from lsdpak_dl import (
    NKG, HistogramDiscriminator, UNetDenoiser, SSIMLoss,
    uniformity_loss, correlation_loss, DEVICE,
)

CKPT_DIR = os.path.join(_HERE, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Image names (keys into lsdpak_v3_rgb.TIFF_SRC) ───────────────────────────
IMAGE_NAMES = ["aerial", "lena", "couple", "clock", "4_2_03", "5_3_01",
               "gradient", "checker", "noise_img", "mandrill_synth"]


# =============================================================================
# DATASET
# =============================================================================

class KeystreamDataset(Dataset):
    """
    Returns 21-byte hash tensors for NKG training.
    Augmented by hashing image patches of varying sizes and offsets.
    """

    def __init__(self, images: list, n_patches: int = 200):
        self.samples = []
        for img in images:
            H, W = img.shape[:2]
            # Full image hash
            hb = image_hash_bytes_rgb(img)
            t  = NKG.hash_to_tensor(hb, device=DEVICE).squeeze(0)  # (21,)
            self.samples.append(t)
            # Patch hashes
            ph = min(H, 64)
            pw = min(W, 64)
            for _ in range(n_patches):
                r = np.random.randint(0, H - ph + 1)
                c = np.random.randint(0, W - pw + 1)
                patch = img[r:r+ph, c:c+pw, :]
                hb_p  = image_hash_bytes_rgb(patch)
                t_p   = NKG.hash_to_tensor(hb_p, device=DEVICE).squeeze(0)
                self.samples.append(t_p)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DenoisingDataset(Dataset):
    """
    Returns (corrupted_plain, clean_plain) pairs for denoiser training.

    Corruption modes applied to the *plaintext* (simulating what decryption of
    a noisy ciphertext would produce):
      - Salt-and-pepper noise
      - Block erasure (crop simulation)
      - Combined S&P + crop
    """

    def __init__(self, images: list, n_aug: int = 80):
        self.pairs = []   # list of (corrupted_tensor, clean_tensor)
        for img in images:
            clean = img.astype(np.float32) / 255.0   # (H,W,3) in [0,1]
            for _ in range(n_aug):
                mode = np.random.choice(["sp", "crop", "both"])
                noisy = clean.copy()
                if mode in ("sp", "both"):
                    noisy = self._salt_pepper(noisy, density=np.random.uniform(0.02, 0.15))
                if mode in ("crop", "both"):
                    noisy = self._random_crop(noisy, max_frac=0.25)
                self.pairs.append((noisy, clean))
        # Shuffle
        np.random.shuffle(self.pairs)

    @staticmethod
    def _salt_pepper(img: np.ndarray, density: float) -> np.ndarray:
        out = img.copy()
        mask = np.random.random(img.shape[:2])
        out[mask < density / 2]              = 0.0
        out[(mask >= density / 2) & (mask < density)] = 1.0
        return out

    @staticmethod
    def _random_crop(img: np.ndarray, max_frac: float) -> np.ndarray:
        out = img.copy()
        H, W = img.shape[:2]
        h = np.random.randint(1, max(2, int(H * max_frac)))
        w = np.random.randint(1, max(2, int(W * max_frac)))
        r = np.random.randint(0, H - h + 1)
        c = np.random.randint(0, W - w + 1)
        out[r:r+h, c:c+w, :] = 0.5   # grey fill (mean)
        return out

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy, clean = self.pairs[idx]
        # (H,W,3) -> (3,H,W)
        t_noisy = torch.from_numpy(noisy.transpose(2, 0, 1)).to(DEVICE)
        t_clean = torch.from_numpy(clean.transpose(2, 0, 1)).to(DEVICE)
        return t_noisy, t_clean


# =============================================================================
# LOGGING HELPERS
# =============================================================================

def _log(phase: int, step: int, total: int, **losses):
    parts = [f"[Ph{phase}] step {step:4d}/{total}"]
    for k, v in losses.items():
        parts.append(f"{k}={v:.5f}")
    print("  ".join(parts))


# =============================================================================
# PHASE 1 -- NKG Pre-training
# =============================================================================

def train_phase1(nkg: NKG, loader: DataLoader,
                 epochs: int = 10, lr: float = 1e-3) -> NKG:
    print(f"\n{'='*60}")
    print(f"PHASE 1 -- NKG Uniformity Pre-training  (epochs={epochs}, lr={lr})")
    print(f"{'='*60}")

    opt = optim.Adam(nkg.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    nkg.train()

    total_steps = epochs * len(loader)
    step = 0
    for ep in range(1, epochs + 1):
        ep_unif = ep_corr = 0.0
        for hb_batch in loader:
            hb_batch = hb_batch.to(DEVICE)   # (B, 21)
            opt.zero_grad()

            # Generate 3 independent channel keystreams
            B    = hb_batch.shape[0]
            N    = IMG_SIZE * IMG_SIZE
            ch_keys = []
            for ch in range(3):
                # Create channel-specific hash by XOR with channel index
                ch_offset = torch.full_like(hb_batch, (ch * 73 + 17) / 128.0 - 1.0)
                ch_input  = torch.tanh(hb_batch + 0.3 * ch_offset)
                ch_keys.append(nkg(ch_input)[:, :N])
            ks3 = torch.cat(ch_keys, dim=1)    # (B, 3*N)

            loss_u = uniformity_loss(ks3)
            loss_c = correlation_loss(ks3)
            loss   = loss_u + 0.1 * loss_c

            loss.backward()
            nn.utils.clip_grad_norm_(nkg.parameters(), 1.0)
            opt.step()

            ep_unif += loss_u.item()
            ep_corr += loss_c.item()
            step += 1

        scheduler.step()
        n_b = len(loader)
        print(f"  Epoch {ep:3d}/{epochs}  unif={ep_unif/n_b:.6f}  "
              f"corr={ep_corr/n_b:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    path = os.path.join(CKPT_DIR, "nkg_pretrained.pt")
    torch.save(nkg.state_dict(), path)
    print(f"\n  Saved -> {path}")
    return nkg


# =============================================================================
# PHASE 2 -- GAN Adversarial Fine-tuning
# =============================================================================

def train_phase2(nkg: NKG, disc: HistogramDiscriminator,
                 loader: DataLoader,
                 epochs: int = 20, lr_g: float = 2e-4, lr_d: float = 2e-4,
                 lambda_unif: float = 0.5) -> tuple:
    """
    Adversarial training loop.

    Discriminator convention (matches HistogramDiscriminator docstring):
      label 1 = histogram looks uniform (what D is rewarded for on real uniform noise)
      label 0 = histogram looks structured / non-uniform

    NKG objective: produce histograms that D classifies as 1 (uniform).

    Alternation: update D every step, update G every step (1:1 ratio).
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2 -- GAN Adversarial Fine-tuning  (epochs={epochs})")
    print(f"{'='*60}")

    opt_g = optim.Adam(nkg.parameters(),  lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    bce   = nn.BCELoss()

    nkg.train()
    disc.train()

    for ep in range(1, epochs + 1):
        ep_d = ep_g = 0.0
        for hb_batch in loader:
            hb_batch = hb_batch.to(DEVICE)
            B = hb_batch.shape[0]
            N = IMG_SIZE * IMG_SIZE

            # ── Generate NKG keystream (3 independent channels) ────────────
            def _make_3ch(hb_in, model, no_grad=False):
                ch_keys = []
                for ch in range(3):
                    ch_off = torch.full_like(hb_in, (ch * 73 + 17) / 128.0 - 1.0)
                    ch_inp = torch.tanh(hb_in + 0.3 * ch_off)
                    if no_grad:
                        with torch.no_grad():
                            ch_keys.append(model(ch_inp)[:, :N])
                    else:
                        ch_keys.append(model(ch_inp)[:, :N])
                return torch.cat(ch_keys, dim=1)

            ks3 = _make_3ch(hb_batch, nkg, no_grad=True)
            nkg_hist = HistogramDiscriminator.keystream_to_hist(ks3)   # (B,3,256)

            # Uniform reference histograms
            uni_hist = HistogramDiscriminator.uniform_hist(B, device=DEVICE)

            # ── Train Discriminator ────────────────────────────────────────
            opt_d.zero_grad()
            loss_d_real = bce(disc(uni_hist),          torch.ones(B,  1, device=DEVICE))
            loss_d_fake = bce(disc(nkg_hist.detach()), torch.zeros(B, 1, device=DEVICE))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            # ── Train NKG (Generator) ──────────────────────────────────────
            opt_g.zero_grad()
            ks3_g = _make_3ch(hb_batch, nkg, no_grad=False)
            nkg_hist_g = HistogramDiscriminator.keystream_to_hist(ks3_g)

            # NKG wants D output -> 1 (fool D into thinking keystream is uniform)
            loss_g_adv  = bce(disc(nkg_hist_g), torch.ones(B, 1, device=DEVICE))
            loss_g_unif = uniformity_loss(ks3_g)
            loss_g      = loss_g_adv + lambda_unif * loss_g_unif
            loss_g.backward()
            nn.utils.clip_grad_norm_(nkg.parameters(), 1.0)
            opt_g.step()

            ep_d += loss_d.item()
            ep_g += loss_g.item()

        n_b = len(loader)
        print(f"  Epoch {ep:3d}/{epochs}  loss_D={ep_d/n_b:.5f}  loss_G={ep_g/n_b:.5f}")

    path_nkg  = os.path.join(CKPT_DIR, "nkg_gan.pt")
    path_disc = os.path.join(CKPT_DIR, "discriminator_gan.pt")
    torch.save(nkg.state_dict(),  path_nkg)
    torch.save(disc.state_dict(), path_disc)
    print(f"\n  Saved -> {path_nkg}")
    print(f"  Saved -> {path_disc}")
    return nkg, disc


# =============================================================================
# PHASE 3 -- U-Net Denoiser Training
# =============================================================================

def train_phase3(denoiser: UNetDenoiser, loader: DataLoader,
                 epochs: int = 30, lr: float = 1e-3) -> UNetDenoiser:
    print(f"\n{'='*60}")
    print(f"PHASE 3 -- U-Net Denoiser Training  (epochs={epochs}, lr={lr})")
    print(f"{'='*60}")

    opt       = optim.Adam(denoiser.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    ssim_loss = SSIMLoss().to(DEVICE)
    l1_loss   = nn.L1Loss()

    denoiser.train()

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for noisy, clean in loader:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            opt.zero_grad()
            pred  = denoiser(noisy)
            loss  = 0.5 * l1_loss(pred, clean) + 0.5 * ssim_loss(pred, clean)
            loss.backward()
            nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()

            ep_loss += loss.item()

        avg = ep_loss / len(loader)
        scheduler.step(avg)
        if ep % 5 == 0 or ep == epochs:
            print(f"  Epoch {ep:3d}/{epochs}  loss={avg:.6f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    path = os.path.join(CKPT_DIR, "denoiser.pt")
    torch.save(denoiser.state_dict(), path)
    print(f"\n  Saved -> {path}")
    return denoiser


# =============================================================================
# MAIN
# =============================================================================

def load_images() -> list:
    images = []
    for name in IMAGE_NAMES:
        try:
            img = load_image_rgb(name)
            images.append(img)
            print(f"  Loaded {name:10s}  shape={img.shape}")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
    if not images:
        raise RuntimeError("No images found. Check test_images/ directory.")
    return images


def main():
    parser = argparse.ArgumentParser(description="LS-DPAK-DL Training")
    parser.add_argument("--phase",  choices=["1","2","3","all"], default="all")
    parser.add_argument("--epochs1", type=int, default=15,  help="Phase 1 epochs")
    parser.add_argument("--epochs2", type=int, default=25,  help="Phase 2 epochs")
    parser.add_argument("--epochs3", type=int, default=35,  help="Phase 3 epochs")
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--patches", type=int,   default=150,
                        help="Patches per image for keystream dataset")
    parser.add_argument("--aug",     type=int,   default=60,
                        help="Augmented pairs per image for denoiser dataset")
    args = parser.parse_args()

    print("\nLS-DPAK-DL  Training")
    print(f"  Device  : {DEVICE}")
    print(f"  Phase   : {args.phase}")
    print(f"  Batch   : {args.batch}")

    # ── Load images ─────────────────────────────────────────────────────────
    print("\nLoading images ...")
    images = load_images()

    # ── Instantiate models ───────────────────────────────────────────────────
    nkg      = NKG().to(DEVICE)
    disc     = HistogramDiscriminator().to(DEVICE)
    denoiser = UNetDenoiser().to(DEVICE)

    # Count parameters
    def _nparams(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"\n  NKG params          : {_nparams(nkg):,}")
    print(f"  Discriminator params: {_nparams(disc):,}")
    print(f"  Denoiser params     : {_nparams(denoiser):,}")

    run_all = (args.phase == "all")

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    if run_all or args.phase == "1":
        ks_dataset = KeystreamDataset(images, n_patches=args.patches)
        ks_loader  = DataLoader(ks_dataset, batch_size=args.batch, shuffle=True)
        print(f"\n  Keystream dataset size: {len(ks_dataset)}")

        nkg = train_phase1(nkg, ks_loader, epochs=args.epochs1, lr=args.lr)

    elif args.phase in ("2", "3"):
        # Try to load Phase 1 checkpoint
        p1_ckpt = os.path.join(CKPT_DIR, "nkg_pretrained.pt")
        if os.path.exists(p1_ckpt):
            nkg.load_state_dict(torch.load(p1_ckpt, map_location=DEVICE))
            print(f"  Loaded Phase 1 NKG from {p1_ckpt}")
        else:
            print("  [WARN] Phase 1 checkpoint not found -- NKG starts from random init")

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    if run_all or args.phase == "2":
        ks_dataset = KeystreamDataset(images, n_patches=args.patches)
        ks_loader  = DataLoader(ks_dataset, batch_size=args.batch, shuffle=True)
        print(f"\n  Keystream dataset size: {len(ks_dataset)}")

        nkg, disc = train_phase2(
            nkg, disc, ks_loader,
            epochs=args.epochs2,
            lr_g=args.lr / 5,   # GAN generators benefit from lower lr
            lr_d=args.lr / 5,
        )

    elif args.phase == "3":
        # Try to load Phase 2 checkpoints
        p2_nkg  = os.path.join(CKPT_DIR, "nkg_gan.pt")
        p2_disc = os.path.join(CKPT_DIR, "discriminator_gan.pt")
        if os.path.exists(p2_nkg):
            nkg.load_state_dict(torch.load(p2_nkg,  map_location=DEVICE))
            print(f"  Loaded Phase 2 NKG from {p2_nkg}")
        if os.path.exists(p2_disc):
            disc.load_state_dict(torch.load(p2_disc, map_location=DEVICE))
            print(f"  Loaded Discriminator from {p2_disc}")

    # ── Phase 3 ─────────────────────────────────────────────────────────────
    if run_all or args.phase == "3":
        den_dataset = DenoisingDataset(images, n_aug=args.aug)
        den_loader  = DataLoader(den_dataset, batch_size=4, shuffle=True)
        print(f"\n  Denoising dataset size: {len(den_dataset)}")

        denoiser = train_phase3(
            denoiser, den_loader,
            epochs=args.epochs3,
            lr=args.lr,
        )

    print("\n\nAll training complete.")
    print(f"Checkpoints saved in: {CKPT_DIR}")


if __name__ == "__main__":
    main()
