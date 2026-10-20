"""
Generate additional test images for expanded evaluation.

Since we cannot download USC-SIPI images programmatically, we create
4 synthetic test images with diverse statistical properties that are
standard in image encryption literature:

1. gradient  - Smooth horizontal+vertical gradient (tests correlation breaking)
2. checker   - Checkerboard pattern (tests block-structured input)
3. noise_img - Random noise image (tests already-random input)
4. mandrill  - Mandrill/baboon-like texture (high-frequency content)

All images are 256x256 RGB, saved as TIFF.
"""

import os
import numpy as np
from PIL import Image

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "..", "test_images")


def make_gradient(size=256):
    """Smooth H+V gradient with channel offsets."""
    x = np.linspace(0, 255, size).astype(np.uint8)
    y = np.linspace(0, 255, size).astype(np.uint8)
    R = np.outer(y, np.ones(size, dtype=np.uint8))
    G = np.outer(np.ones(size, dtype=np.uint8), x)
    B = ((R.astype(int) + G.astype(int)) // 2).astype(np.uint8)
    return np.stack([R, G, B], axis=2)


def make_checker(size=256, block=16):
    """Checkerboard pattern with coloured squares."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(size):
        for c in range(size):
            if ((r // block) + (c // block)) % 2 == 0:
                img[r, c] = [200, 50, 50]
            else:
                img[r, c] = [50, 50, 200]
    return img


def make_noise_img(size=256):
    """Uniformly random RGB image."""
    return np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)


def make_mandrill_like(size=256):
    """High-frequency texture (sinusoidal patterns simulating fur/texture)."""
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)

    R = (127 + 127 * np.sin(xx * 0.3 + yy * 0.2) *
         np.cos(xx * 0.15 - yy * 0.1)).astype(np.uint8)
    G = (127 + 127 * np.sin(xx * 0.2 - yy * 0.25) *
         np.sin(yy * 0.12 + xx * 0.08)).astype(np.uint8)
    B = (127 + 127 * np.cos(xx * 0.18 + yy * 0.22) *
         np.sin(xx * 0.1 + yy * 0.3)).astype(np.uint8)
    return np.stack([R, G, B], axis=2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    images = {
        "gradient": make_gradient(),
        "checker":  make_checker(),
        "noise_img": make_noise_img(),
        "mandrill_synth": make_mandrill_like(),
    }

    for name, arr in images.items():
        path = os.path.join(OUT_DIR, f"{name}.tiff")
        Image.fromarray(arr, "RGB").save(path)
        print(f"  Saved {name:20s}  shape={arr.shape}  -> {path}")

    print(f"\nTotal images now in {OUT_DIR}:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
