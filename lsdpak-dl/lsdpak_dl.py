"""
LS-DPAK-DL  —  Deep Learning Enhanced Image Encryption
=======================================================

Architecture overview
---------------------

1. Neural Keystream Generator (NKG)
   - Replaces Lorenz/Rossler with a learned MLP.
   - Input : 21 hash bytes (SHA-256 of plaintext → adaptive ICs).
   - Output: N=65536 float values in [0,1] per channel  (×255 → keystream bytes).
   - Architecture: hash embedding → chunk-wise positional MLP decoder.
   - Trained to produce maximally uniform byte histograms (pre-training)
     and then adversarially fine-tuned against a histogram discriminator (GAN).

2. Histogram Discriminator (D)
   - Input : a (256,)-length normalised histogram per channel
             concatenated → 768-dim vector.
   - Output: scalar ∈ [0,1] — probability that the histogram is "non-uniform".
   - Training: D(uniform_noise)→1, D(cipher)→0  (learns to detect structure).
   - NKG training: fooling D so D(cipher)→1  (make cipher look uniform).

3. U-Net Denoiser
   - Addresses the T9/T10 weakness: bidirectional diffusion is tamper-sensitive,
     so any corruption in ciphertext propagates through the entire decrypted output.
   - Pipeline at test time:
       noisy_cipher → decrypt (standard LS-DPAK) → corrupted_plain
       → UNetDenoiser → clean_plain
   - Architecture: encoder–bottleneck–decoder with skip connections, 4 levels,
     max 32 channels  (CPU-feasible).
   - Loss: L1 + SSIM composite.
"""

import math
import hashlib
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── Import LS-DPAK v3 RGB for the fixed pipeline steps ────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_V3   = os.path.join(_HERE, "..", "lsdpak-v3-rgb")
sys.path.insert(0, _V3)
from lsdpak_v3_rgb import (
    build_D, build_D5, scramble3, unscramble3,
    shuffle_blocks_full, unshuffle_blocks_full,
    interchannel_shuffle, interchannel_unshuffle,
    bidir_encrypt_joint, bidir_decrypt_joint,
    image_hash_bytes_rgb, ics_from_hash_bytes,
    IMG_SIZE, MASTER, SCALE, CH_NAMES,
)

DEVICE = torch.device("cpu")   # change to "cuda" if available

# ─────────────────────────────────────────────────────────────────────────────
# 1.  NEURAL KEYSTREAM GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class NKG(nn.Module):
    """
    Neural Keystream Generator.

    Generates N=65536 bytes per channel from a 21-byte hash input.

    Design:
      - Hash bytes (21,) → embedding (128,) via 2-layer MLP.
      - N bytes split into n_chunks=256 chunks of chunk_size=256 bytes.
      - Each chunk decoded by a small MLP conditioned on:
            [embedding, sinusoidal chunk position encoding (dim=8)]
      - Forward pass: B images × 256 chunks × 256 bytes in one batched call.

    Memory footprint per batch: B × n_chunks × (128+8) = B × 35,328 floats ≈ 140 KB/sample — CPU-feasible.
    """

    CHUNK_SIZE : int = 256
    N_CHUNKS   : int = 256          # 256 × 256 = 65536 bytes = 256×256 image

    def __init__(self, embed_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.embed_dim  = embed_dim
        self.chunk_size = self.CHUNK_SIZE
        self.n_chunks   = self.N_CHUNKS

        # Hash → embedding
        self.embedder = nn.Sequential(
            nn.Linear(21, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
            nn.Tanh(),      # bounded output → stable training
        )

        # Positional encoding: chunk_idx ∈ [0, n_chunks-1]
        pos = torch.arange(self.n_chunks).float() / self.n_chunks   # [0,1)
        pos_enc = torch.stack([torch.sin(pos * 2 ** k * math.pi)
                               for k in range(8)], dim=1)            # (n_chunks, 8)
        self.register_buffer("pos_enc", pos_enc)

        # Chunk decoder: [embed + pos] → chunk_size values
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim + 8, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, self.chunk_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hash_bytes_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hash_bytes_norm: (B, 21) float32 in [-1, 1].
        Returns:
            keystream: (B, N) float32 in [0, 1]  (N = n_chunks × chunk_size).
        """
        B = hash_bytes_norm.shape[0]

        embed = self.embedder(hash_bytes_norm)          # (B, embed_dim)

        # Expand for all chunks
        embed_exp = embed.unsqueeze(1).expand(-1, self.n_chunks, -1)  # (B, n_chunks, E)
        pos_exp   = self.pos_enc.unsqueeze(0).expand(B, -1, -1)       # (B, n_chunks, 8)
        x = torch.cat([embed_exp, pos_exp], dim=2)                    # (B, n_chunks, E+8)

        # Batch decode
        x_flat   = x.reshape(B * self.n_chunks, self.embed_dim + 8)
        out_flat = self.decoder(x_flat)                               # (B*n_chunks, chunk_size)
        out      = out_flat.reshape(B, self.n_chunks * self.chunk_size)  # (B, N)

        return torch.sigmoid(out)    # values in [0, 1]

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def hash_to_tensor(hb: bytes, device=DEVICE) -> torch.Tensor:
        """Convert 21-byte hash slice to normalised [-1,1] float32 tensor (1,21)."""
        arr = np.frombuffer(hb[:21], dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0          # map [0,255] → [-1,1]
        return torch.from_numpy(arr).unsqueeze(0).to(device)   # (1, 21)

    def generate_keystream(self, hb: bytes, size: int = IMG_SIZE) -> np.ndarray:
        """
        Generate an (H, W) uint8 keystream array from 21 hash bytes.
        Used as drop-in replacement for build_keystream().
        """
        self.eval()
        with torch.no_grad():
            t = self.hash_to_tensor(hb)
            ks = self(t).squeeze(0).numpy()          # (N,) in [0,1]
        # Scale to [0,255]
        ks_bytes = np.clip(np.round(ks * 255), 0, 255).astype(np.uint8)
        return ks_bytes[:size * size].reshape(size, size)

    def generate_channel_keystreams(self, hb: bytes, size: int = IMG_SIZE) -> list:
        """
        Generate 3 independent channel keystreams using channel-specific
        hash slices: hb[0:7], hb[7:14], hb[14:21], each zero-padded to 21 bytes.

        This ensures R/G/B keys are truly independent (not circular shifts).
        Returns: list of 3 (H, W) uint8 arrays.
        """
        self.eval()
        keys = []
        for ch in range(3):
            # Extract 7-byte channel slice and pad to 21 bytes
            ch_slice = hb[ch*7:(ch+1)*7]
            ch_hb = ch_slice + bytes(14)  # pad remaining 14 bytes with zeros
            # XOR with channel index for extra differentiation
            ch_hb = bytes([b ^ (ch * 73 + 17) for b in ch_hb])
            ks = self.generate_keystream(ch_hb, size)
            keys.append(ks)
        return keys


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HISTOGRAM DISCRIMINATOR
# ─────────────────────────────────────────────────────────────────────────────

class HistogramDiscriminator(nn.Module):
    """
    Classifies whether a keystream / cipher looks uniform.

    Input : (B, 3, 256) — per-channel normalised histograms.
    Output: (B, 1) — probability of being "non-uniform" (real cipher).

    Training signal:
      D(uniform_noise_hist) → 1  (truly uniform → label 1 for D)
      D(nkg_ks_hist)        → 0  (NKG keystream → D calls it non-uniform)
    NKG fooling:
      NKG wants D(nkg_ks_hist) → 1  (make histogram look uniform).
    """

    def __init__(self, hidden: int = 256):
        super().__init__()
        # Per-channel feature extractor (shared weights)
        self.ch_net = nn.Sequential(
            nn.Linear(256, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
        )
        # Fusion + classification
        self.classifier = nn.Sequential(
            nn.Linear((hidden // 2) * 3, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, hist3: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hist3: (B, 3, 256) normalised histograms.
        Returns:
            (B, 1) probability.
        """
        feats = [self.ch_net(hist3[:, c, :]) for c in range(3)]   # 3 × (B, hidden//2)
        fused = torch.cat(feats, dim=1)                             # (B, 3*hidden//2)
        return self.classifier(fused)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def keystream_to_hist(ks_float: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
        """
        Compute differentiable soft histogram of a keystream.

        Uses soft binning: each value contributes to the two adjacent bins
        proportionally (linear interpolation), making it differentiable w.r.t. ks_float.

        Args:
            ks_float: (B, N) float32 in [0, 1].
        Returns:
            hist: (B, 3, n_bins) normalised  (3 = R/G/B channels interleaved).
        """
        B, N = ks_float.shape
        assert N % 3 == 0, "keystream length must be divisible by 3"
        ch_len = N // 3
        hists = []
        for c in range(3):
            ch = ks_float[:, c * ch_len : (c + 1) * ch_len]   # (B, ch_len)
            # Scale to [0, n_bins-1]
            scaled = ch * (n_bins - 1)                          # (B, ch_len)
            lo     = scaled.floor().long().clamp(0, n_bins - 2) # (B, ch_len)
            frac   = scaled - lo.float()                        # fractional part

            # Scatter into histogram bins
            hist = torch.zeros(B, n_bins, device=ks_float.device)
            hist.scatter_add_(1, lo,           1.0 - frac)
            hist.scatter_add_(1, (lo + 1).clamp(max=n_bins-1), frac)
            hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1e-8)
            hists.append(hist)
        return torch.stack(hists, dim=1)      # (B, 3, n_bins)

    @staticmethod
    def uniform_hist(B: int, n_bins: int = 256, device=DEVICE) -> torch.Tensor:
        """Reference uniform histogram with small random noise for D training."""
        h = torch.ones(B, 3, n_bins, device=device) / n_bins
        noise = torch.randn_like(h) * 0.001
        h = (h + noise).abs()
        return h / h.sum(dim=2, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  U-NET DENOISER
# ─────────────────────────────────────────────────────────────────────────────

class _DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNetDenoiser(nn.Module):
    """
    Small U-Net for noise-robust decryption.

    At test time:
        1. Receive noisy ciphertext (salt-and-pepper / cropped).
        2. Decrypt with correct key → corrupted plaintext.
        3. Pass corrupted plaintext through UNetDenoiser → clean plaintext.

    Architecture: 4-level encoder + bottleneck + 4-level decoder with skip connections.
    Base channels = 16 (CPU-friendly; ~500 K parameters).

    Input/Output: (B, 3, H, W) float32 in [0, 1].
    """

    def __init__(self, base_ch: int = 16):
        super().__init__()
        c = base_ch
        # Encoder
        self.enc1 = _DoubleConv(3,    c)
        self.enc2 = _DoubleConv(c,    c * 2)
        self.enc3 = _DoubleConv(c*2,  c * 4)
        self.enc4 = _DoubleConv(c*4,  c * 8)
        # Bottleneck
        self.bottleneck = _DoubleConv(c*8, c * 16)
        # Decoder
        self.up4   = nn.ConvTranspose2d(c*16, c*8,  2, stride=2)
        self.dec4  = _DoubleConv(c*16, c*8)
        self.up3   = nn.ConvTranspose2d(c*8,  c*4,  2, stride=2)
        self.dec3  = _DoubleConv(c*8,  c*4)
        self.up2   = nn.ConvTranspose2d(c*4,  c*2,  2, stride=2)
        self.dec2  = _DoubleConv(c*4,  c*2)
        self.up1   = nn.ConvTranspose2d(c*2,  c,    2, stride=2)
        self.dec1  = _DoubleConv(c*2,  c)
        # Head
        self.head  = nn.Sequential(
            nn.Conv2d(c, 3, 1),
            nn.Sigmoid(),
        )
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        # Bottleneck
        b  = self.bottleneck(self.pool(e4))
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SSIM LOSS  (used in denoiser training)
# ─────────────────────────────────────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss: 1 - SSIM(pred, target).
    Operates on (B, C, H, W) float tensors in [0, 1].
    """
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._gaussian_window(window_size))

    @staticmethod
    def _gaussian_window(size: int) -> torch.Tensor:
        sigma = 1.5
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)    # (1,1,size,size)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        B, C, H, W = x.shape
        win = self.window.expand(C, 1, -1, -1).to(x.device)
        pad = self.window_size // 2

        mu_x  = F.conv2d(x, win, padding=pad, groups=C)
        mu_y  = F.conv2d(y, win, padding=pad, groups=C)
        mu_xx = mu_x * mu_x; mu_yy = mu_y * mu_y; mu_xy = mu_x * mu_y

        sig_xx = F.conv2d(x * x, win, padding=pad, groups=C) - mu_xx
        sig_yy = F.conv2d(y * y, win, padding=pad, groups=C) - mu_yy
        sig_xy = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        den = (mu_xx + mu_yy + C1) * (sig_xx + sig_yy + C2)
        return (num / den.clamp(min=1e-8)).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DL-ENHANCED FULL ENCRYPTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def encrypt_dl(img_rgb: np.ndarray, D: np.ndarray,
               nkg: NKG, use_nkg: bool = True) -> tuple:
    """
    LS-DPAK-DL encryption.

    Same pipeline as v3 but keystream comes from NKG instead of Lorenz/Rossler.
    The permutation and bidir diffusion remain unchanged (fixed LS-DPAK steps).

    Args:
        img_rgb   : (H,W,3) uint8 plaintext.
        D         : Latin square matrix (32×32).
        nkg       : trained NeuralKeystreamGenerator.
        use_nkg   : if False, falls back to v3 Lorenz/Rossler (for comparison).
    Returns:
        cipher (H,W,3) uint8, hb (21 bytes metadata).
    """
    from lsdpak_v3_rgb import build_keystream as lorenz_ks, adaptive_ics_rgb

    H, W = img_rgb.shape[:2]
    n    = D.shape[0]
    band = H // n
    D5   = build_D5(n)

    hb = image_hash_bytes_rgb(img_rgb)      # 21 bytes

    # Per-channel keystreams (3 independent NKG forward passes)
    if use_nkg:
        keys = nkg.generate_channel_keystreams(hb, H)
    else:
        ics_list = adaptive_ics_rgb(img_rgb)
        keys = [lorenz_ks(ics_list[c], H) for c in range(3)]

    # Per-channel scramble + block shuffle
    shuffled = []
    for ch in range(3):
        sc = scramble3(img_rgb[:, :, ch], D)
        sh = shuffle_blocks_full(sc)
        shuffled.append(sh)

    # Inter-channel LS shuffle
    R_sh, G_sh, B_sh = interchannel_shuffle(shuffled[0], shuffled[1], shuffled[2], D5, band)

    # Joint bidir diffusion
    flat_arr = np.concatenate([R_sh.ravel(), G_sh.ravel(), B_sh.ravel()])
    flat_key = np.concatenate([keys[0].ravel(), keys[1].ravel(), keys[2].ravel()])
    cipher_flat = bidir_encrypt_joint(flat_arr, flat_key)

    N = H * W
    cipher = np.stack([cipher_flat[:N].reshape(H, W),
                       cipher_flat[N:2*N].reshape(H, W),
                       cipher_flat[2*N:].reshape(H, W)], axis=2).astype(np.uint8)
    return cipher, hb


def decrypt_dl(cipher: np.ndarray, D: np.ndarray, hb: bytes,
               nkg: NKG, use_nkg: bool = True,
               denoiser: UNetDenoiser = None) -> np.ndarray:
    """
    LS-DPAK-DL decryption with optional denoiser post-processing.

    Args:
        cipher   : (H,W,3) uint8 ciphertext (possibly noisy).
        D        : Latin square (32×32).
        hb       : 21-byte hash metadata.
        nkg      : trained NKG.
        use_nkg  : if False, uses Lorenz/Rossler keystream.
        denoiser : if not None, applies UNetDenoiser to the decrypted output.
    Returns:
        plaintext (H,W,3) uint8.
    """
    from lsdpak_v3_rgb import build_keystream as lorenz_ks

    H, W = cipher.shape[:2]
    n    = D.shape[0]
    band = H // n
    D5   = build_D5(n)

    if use_nkg:
        keys = nkg.generate_channel_keystreams(hb, H)
    else:
        ics = ics_from_hash_bytes(hb)
        keys = [lorenz_ks(ics[c], H) for c in range(3)]

    N = H * W
    flat_cipher = np.concatenate([cipher[:,:,0].ravel(),
                                  cipher[:,:,1].ravel(),
                                  cipher[:,:,2].ravel()])
    flat_key    = np.concatenate([keys[0].ravel(), keys[1].ravel(), keys[2].ravel()])
    plain_flat  = bidir_decrypt_joint(flat_cipher, flat_key)

    R_sh = plain_flat[:N].reshape(H, W).astype(np.uint8)
    G_sh = plain_flat[N:2*N].reshape(H, W).astype(np.uint8)
    B_sh = plain_flat[2*N:].reshape(H, W).astype(np.uint8)

    R_sh, G_sh, B_sh = interchannel_unshuffle(R_sh, G_sh, B_sh, D5, band)

    result = np.empty((H, W, 3), np.uint8)
    for i, sh in enumerate([R_sh, G_sh, B_sh]):
        ush = unshuffle_blocks_full(sh)
        result[:, :, i] = unscramble3(ush, D)

    # Optional denoiser post-processing
    if denoiser is not None:
        denoiser.eval()
        with torch.no_grad():
            inp = torch.from_numpy(result.astype(np.float32) / 255.0)
            inp = inp.permute(2, 0, 1).unsqueeze(0)          # (1,3,H,W)
            out = denoiser(inp).squeeze(0).permute(1, 2, 0)  # (H,W,3)
            result = (out.numpy() * 255.0).clip(0, 255).astype(np.uint8)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LOSS UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def uniformity_loss(ks_float: torch.Tensor) -> torch.Tensor:
    """
    Chi-square-style loss: sum of squared deviations from uniform.
    ks_float: (B, N) float32 in [0,1].
    Returns scalar loss (mean over batch).
    """
    hist = HistogramDiscriminator.keystream_to_hist(ks_float)   # (B, 3, 256)
    target = torch.ones_like(hist) / 256.0
    return F.mse_loss(hist, target)


def correlation_loss(ks_float: torch.Tensor, size: int = IMG_SIZE) -> torch.Tensor:
    """
    Pixel correlation loss: mean squared H/V correlation across cipher channels.
    ks_float: (B, N) float32, N = 3 * size * size.
    Returns scalar.
    """
    B = ks_float.shape[0]
    N = size * size
    losses = []
    for c in range(3):
        ch = ks_float[:, c*N:(c+1)*N].reshape(B, size, size)
        # Horizontal
        rh = ((ch[:, :, :-1] - ch[:, :, :-1].mean(dim=-1, keepdim=True)) *
              (ch[:, :,  1:] - ch[:, :,  1:].mean(dim=-1, keepdim=True)))
        losses.append(rh.mean().pow(2))
        # Vertical
        rv = ((ch[:, :-1, :] - ch[:, :-1, :].mean(dim=-2, keepdim=True)) *
              (ch[:,  1:, :] - ch[:,  1:, :].mean(dim=-2, keepdim=True)))
        losses.append(rv.mean().pow(2))
    return torch.stack(losses).mean()
