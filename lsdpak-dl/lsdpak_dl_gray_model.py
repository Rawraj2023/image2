"""
LS-DPAK-DL grayscale -- native model architectures.
=====================================================================

All five images this project tests against are mode "L" -- genuinely
single-channel grayscale (verified: max|R-G| = max|R-B| = max|G-B| = 0 after
`.convert("RGB")`, i.e. every "RGB" result in this project's earlier work
was computed on a grayscale image replicated into three identical channels).
The RGB architecture in `lsdpak_dl.py` was built around that non-existent
3rd dimension: a 21-byte hash split into three 7-byte slices, a histogram
discriminator that fuses three channel blocks, a denoiser with 3 input/output
channels. None of that machinery was ever exercised on real colour content.

This module replaces it with single-channel architectures sized for what the
data actually is: a 256x256 grayscale plane, a 7-byte hash (v5's `hb`, not a
padded 21-byte one), one histogram, one channel. These are NEW networks
trained from scratch (see train_dl_gray.py) -- not nkg_gan.pt repurposed via
zero-padding, which is what the previous version of lsdpak_dl_gray.py did and
which this supersedes. lsdpak_dl.py and train.py are untouched; nothing here
depends on them or changes their behaviour.

NKGGray
-------
Same positional-chunk-decoder design as the RGB NKG, with the embedder's
input dimension changed from 21 to 7 (v5's actual hb length) and no
per-channel split -- one forward pass IS the whole keystream, because there
is only one channel.

HistogramDiscriminatorGray
---------------------------
Same per-bin MLP design as the RGB discriminator, operating on one
(B, 256) histogram instead of fusing three (B, 3, 256) ones.

UNetDenoiserGray
-----------------
Identical U-Net topology to the RGB denoiser (4-level encoder/decoder,
base_ch=16), with in/out channels changed from 3 to 1.

uniformity_loss_gray / correlation_loss_gray
----------------------------------------------
Single-plane versions of the RGB losses in lsdpak_dl.py, which hard-code a
3-way channel split (`HistogramDiscriminator.keystream_to_hist` asserts
`N % 3 == 0`, which 65536 does not satisfy anyway -- 2**16 has no factor of
3 -- so those RGB loss functions cannot be called on a single grayscale
plane without this rewrite).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")

IMG_SIZE = 256
HB_LEN = 7          # v5's actual transmitted metadata length -- no padding


# ============================================================================
# 1. NEURAL KEYSTREAM GENERATOR -- single channel
# ============================================================================

class NKGGray(nn.Module):
    """
    Generates N=65536 bytes (one 256x256 plane) from a 7-byte grayscale hb.

    Architecture identical in shape to lsdpak_dl.NKG except the embedder's
    input dimension is 7, not 21, and there is no per-channel hash slicing --
    one hb, one forward pass, one plane.
    """

    CHUNK_SIZE: int = 256
    N_CHUNKS: int = 256          # 256 x 256 = 65536 bytes = 256x256 image

    def __init__(self, embed_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.chunk_size = self.CHUNK_SIZE
        self.n_chunks = self.N_CHUNKS

        self.embedder = nn.Sequential(
            nn.Linear(HB_LEN, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
            nn.Tanh(),
        )

        pos = torch.arange(self.n_chunks).float() / self.n_chunks
        pos_enc = torch.stack([torch.sin(pos * 2 ** k * math.pi)
                               for k in range(8)], dim=1)
        self.register_buffer("pos_enc", pos_enc)

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

    def forward(self, hb_norm: torch.Tensor) -> torch.Tensor:
        """hb_norm: (B, 7) float32 in [-1,1]. Returns (B, N) float32 in [0,1]."""
        B = hb_norm.shape[0]
        embed = self.embedder(hb_norm)
        embed_exp = embed.unsqueeze(1).expand(-1, self.n_chunks, -1)
        pos_exp = self.pos_enc.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([embed_exp, pos_exp], dim=2)
        x_flat = x.reshape(B * self.n_chunks, self.embed_dim + 8)
        out_flat = self.decoder(x_flat)
        out = out_flat.reshape(B, self.n_chunks * self.chunk_size)
        return torch.sigmoid(out)

    @staticmethod
    def hash_to_tensor(hb7: bytes, device=DEVICE) -> torch.Tensor:
        """7-byte hb -> normalised [-1,1] float32 tensor (1,7). No padding --
        v5's hb IS 7 bytes; nothing is sliced or dropped."""
        import numpy as np
        assert len(hb7) == HB_LEN, f"expected {HB_LEN} bytes, got {len(hb7)}"
        arr = np.frombuffer(hb7, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
        return torch.from_numpy(arr).unsqueeze(0).to(device)

    def generate_keystream(self, hb7: bytes, size: int = IMG_SIZE):
        """(size,size) uint8 keystream plane from a 7-byte hb."""
        import numpy as np
        self.eval()
        with torch.no_grad():
            t = self.hash_to_tensor(hb7)
            ks = self(t).squeeze(0).numpy()
        ks_bytes = np.clip(np.round(ks * 255), 0, 255).astype(np.uint8)
        return ks_bytes[:size * size].reshape(size, size)


# ============================================================================
# 2. HISTOGRAM DISCRIMINATOR -- single channel
# ============================================================================

class HistogramDiscriminatorGray(nn.Module):
    """Classifies whether a single-channel keystream/cipher histogram looks
    uniform. Same training convention as the RGB version: D(uniform)->1,
    D(nkg)->0; NKG tries to fool D into outputting 1."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.ch_net = nn.Sequential(
            nn.Linear(256, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden // 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        """hist: (B, 256) normalised histogram. Returns (B, 1)."""
        feat = self.ch_net(hist)
        return self.classifier(feat)

    @staticmethod
    def keystream_to_hist(ks_float: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
        """Differentiable soft histogram (linear-interpolation binning) of a
        single-channel keystream. ks_float: (B, N) float32 in [0,1]."""
        B, N = ks_float.shape
        scaled = ks_float * (n_bins - 1)
        lo = scaled.floor().long().clamp(0, n_bins - 2)
        frac = scaled - lo.float()
        hist = torch.zeros(B, n_bins, device=ks_float.device)
        hist.scatter_add_(1, lo, 1.0 - frac)
        hist.scatter_add_(1, (lo + 1).clamp(max=n_bins - 1), frac)
        hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return hist

    @staticmethod
    def uniform_hist(B: int, n_bins: int = 256, device=DEVICE) -> torch.Tensor:
        h = torch.ones(B, n_bins, device=device) / n_bins
        noise = torch.randn_like(h) * 0.001
        h = (h + noise).abs()
        return h / h.sum(dim=1, keepdim=True)


# ============================================================================
# 3. U-NET DENOISER -- single channel
# ============================================================================

class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetDenoiserGray(nn.Module):
    """Same topology as lsdpak_dl.UNetDenoiser (4-level encoder/decoder,
    skip connections, base_ch=16), in/out channels = 1 instead of 3."""

    def __init__(self, base_ch: int = 16):
        super().__init__()
        c = base_ch
        self.enc1 = _DoubleConv(1, c)
        self.enc2 = _DoubleConv(c, c * 2)
        self.enc3 = _DoubleConv(c * 2, c * 4)
        self.enc4 = _DoubleConv(c * 4, c * 8)
        self.bottleneck = _DoubleConv(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = _DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = _DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = _DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = _DoubleConv(c * 2, c)
        self.head = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ============================================================================
# 4. LOSS UTILITIES -- single channel
# ============================================================================

def uniformity_loss_gray(ks_float: torch.Tensor) -> torch.Tensor:
    """ks_float: (B, N) float32 in [0,1]. Chi-square-style MSE to uniform."""
    hist = HistogramDiscriminatorGray.keystream_to_hist(ks_float)
    target = torch.ones_like(hist) / 256.0
    return F.mse_loss(hist, target)


def correlation_loss_gray(ks_float: torch.Tensor, size: int = IMG_SIZE) -> torch.Tensor:
    """ks_float: (B, N) float32, N = size*size. Mean squared H/V correlation
    on the single plane -- no 3-way channel loop, there is one channel."""
    B = ks_float.shape[0]
    ch = ks_float.reshape(B, size, size)
    rh = ((ch[:, :, :-1] - ch[:, :, :-1].mean(dim=-1, keepdim=True)) *
          (ch[:, :, 1:] - ch[:, :, 1:].mean(dim=-1, keepdim=True)))
    rv = ((ch[:, :-1, :] - ch[:, :-1, :].mean(dim=-2, keepdim=True)) *
          (ch[:, 1:, :] - ch[:, 1:, :].mean(dim=-2, keepdim=True)))
    return torch.stack([rh.mean().pow(2), rv.mean().pow(2)]).mean()
