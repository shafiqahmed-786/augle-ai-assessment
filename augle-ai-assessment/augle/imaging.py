"""Image I/O and hashing helpers shared by collection, annotation, dataset and validation.

Every component reads pixels through :func:`read_image_bgr` / :func:`decode_image`
so that EXIF-orientation handling is identical everywhere (OpenCV applies the EXIF
orientation on decode). Stored dimensions in the COCO file therefore always match
what the dataset and the annotator actually see.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


class ImageReadError(ValueError):
    """Raised when bytes or a file cannot be decoded into an image."""


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes into a BGR ``uint8`` array (H, W, 3)."""
    if not data:
        raise ImageReadError("empty byte payload")
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.size == 0:
        raise ImageReadError("payload is not a decodable image")
    return img


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read an image file as BGR ``uint8``; unicode-path safe (unlike ``cv2.imread`` on Windows)."""
    p = Path(path)
    if not p.is_file():
        raise ImageReadError(f"image file does not exist: {p}")
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
    except OSError as exc:  # pragma: no cover - filesystem specific
        raise ImageReadError(f"cannot read {p}: {exc}") from exc
    return decode_image(data.tobytes())


def encode_jpeg(img_bgr: np.ndarray, quality: int = 95) -> bytes:
    """Encode a BGR array as JPEG bytes (EXIF is not written, so orientation is baked in)."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ImageReadError("JPEG encoding failed")
    return buf.tobytes()


def resize_max_side(img_bgr: np.ndarray, max_side: int) -> np.ndarray:
    """Down-scale so the longest side is at most ``max_side`` (never upscales)."""
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img_bgr
    scale = max_side / float(longest)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sha256_hex(data: bytes) -> str:
    """Hex SHA-256 digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def dhash(img_bgr: np.ndarray, hash_size: int = 8) -> int:
    """64-bit (for ``hash_size=8``) difference hash used for near-duplicate detection."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two integer hashes."""
    return bin(a ^ b).count("1")
