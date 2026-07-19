"""
pose/normalizer.py  —  Keypoint Normalization
=============================================
Menormalisasi keypoint tubuh agar model tidak terpengaruh oleh:
  - Jarak anak dari kamera
  - Posisi anak di frame
  - Ukuran tubuh yang berbeda-beda antar anak

Pendekatan ini identik dengan normalisasi di repo referensi (ASL processing),
yang menormalisasi 21 titik tangan relatif terhadap bounding box tangan.
Di sini, kita normalisasi 33 titik tubuh relatif terhadap jarak bahu-ke-pinggul
sebagai unit referensi jarak tubuh.

Teknik:
  1. Translasi  → geser agar pusat bahu-pinggul berada di (0, 0, 0)
  2. Skalasi    → bagi dengan jarak bahu-ke-pinggul (ukuran torso)
  3. Hasil      → koordinat dalam "unit torso", tidak bergantung skala absolut
"""

import numpy as np


# Indeks landmark penting untuk normalisasi
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP       = 23
_RIGHT_HIP      = 24


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalisasi keypoint tubuh relatif terhadap torso.

    Args:
        keypoints: np.ndarray shape (33, 3)  — koordinat x, y, z dari MediaPipe

    Returns:
        np.ndarray shape (33, 3)  — koordinat ternormalisasi
    """
    kp = keypoints.copy()

    # ── 1. Hitung titik referensi ──────────────────────────────────────────
    # Pusat bahu
    shoulder_mid = (kp[_LEFT_SHOULDER] + kp[_RIGHT_SHOULDER]) / 2.0
    # Pusat pinggul
    hip_mid      = (kp[_LEFT_HIP]      + kp[_RIGHT_HIP])      / 2.0
    # Pusat torso (origin normalisasi)
    center       = (shoulder_mid + hip_mid) / 2.0

    # ── 2. Translasi ke origin ─────────────────────────────────────────────
    kp -= center

    # ── 3. Skalasi dengan panjang torso ────────────────────────────────────
    torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso_length > 1e-6:
        kp /= torso_length

    # ── 4. Clip agar tidak ada outlier ekstrem (misal jika pose partial) ──
    kp = np.clip(kp, -5.0, 5.0)

    return kp.astype(np.float32)


def normalize_batch(keypoints_batch: np.ndarray) -> np.ndarray:
    """
    Normalisasi batch keypoints.

    Args:
        keypoints_batch: shape (N, 33, 3)

    Returns:
        shape (N, 33, 3)
    """
    return np.stack([normalize_keypoints(kp) for kp in keypoints_batch])
