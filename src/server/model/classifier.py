"""
model/classifier.py  —  Real-Time Inference Pipeline
=====================================================
Mengelola sliding window frame dan menjalankan inferensi model
PointNet + LSTM untuk setiap window yang terbentuk.

Identik secara konsep dengan tahap "Classification" di repo referensi,
yang menjalankan PointNet pada setiap frame tangan yang masuk.
"""

import os
import numpy as np
import tensorflow as tf
from collections import deque
from typing import Optional, Dict

from model.pointnet  import CLASSES, N_CLASSES, N_POINTS, N_DIMS
from model.temporal  import WINDOW_SIZE, EMBED_DIM, build_pointnet_lstm
from pose.normalizer import normalize_keypoints


class StimmingClassifier:
    """
    Mengelola:
      1. Buffer sliding window (60 frame, step 15 = setiap 0.5 detik)
      2. Inferensi PointNet + LSTM
      3. Mengembalikan label + confidence
    """

    def __init__(
        self,
        weights_path:         str   = "weights/stimming_model.keras",
        window_size:          int   = WINDOW_SIZE,
        step:                 int   = 15,
        confidence_threshold: float = 0.65,
    ):
        self._window_size = window_size
        self._step        = step
        self._threshold   = confidence_threshold
        self._buffer      = deque(maxlen=window_size)
        self._frame_count = 0
        self._last_result = self._empty_result()

        # Muat model
        if os.path.exists(weights_path):
            print(f"[Classifier] Memuat weights: {weights_path}")
            self._model = tf.keras.models.load_model(weights_path)
        else:
            print(f"[Classifier] Weights tidak ditemukan. Membuat model baru.")
            print(f"[Classifier] Latih dengan: python train.py")
            self._model = build_pointnet_lstm()

    # ── Public ──────────────────────────────────────────────────────────────

    def predict(self, keypoints: np.ndarray) -> Dict:
        """
        Masukkan satu frame keypoints (33, 3) ke buffer.
        Setiap `step` frame, jalankan inferensi dan kembalikan hasil.

        Args:
            keypoints: np.ndarray shape (33, 3) — sudah dinormalisasi

        Returns:
            dict dengan keys: label, class_id, confidence, is_stimming, all_probs
        """
        self._buffer.append(keypoints)
        self._frame_count += 1

        # Jalankan inferensi saat buffer penuh dan sudah cukup frame baru
        if (
            len(self._buffer) == self._window_size
            and self._frame_count % self._step == 0
        ):
            self._last_result = self._run_inference()

        return self._last_result

    def reset(self):
        """Reset buffer (dipanggil saat koneksi WebSocket baru."""
        self._buffer.clear()
        self._frame_count = 0
        self._last_result = self._empty_result()

    # ── Private ─────────────────────────────────────────────────────────────

    def _run_inference(self) -> Dict:
        """Jalankan model pada window saat ini."""
        # Build tensor: (1, window_size, 33, 3)
        window = np.stack(list(self._buffer), axis=0)   # (60, 33, 3)
        window = window[np.newaxis, ...]                  # (1, 60, 33, 3)

        probs    = self._model.predict(window, verbose=0)[0]   # (N_CLASSES,)
        class_id = int(np.argmax(probs))
        conf     = float(probs[class_id])
        label    = CLASSES[class_id]

        return {
            "label":       label,
            "class_id":    class_id,
            "confidence":  conf,
            "is_stimming": (class_id != 0) and (conf >= self._threshold),
            "all_probs":   {c: float(p) for c, p in zip(CLASSES, probs)},
        }

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "label":       "menunggu",
            "class_id":    -1,
            "confidence":  0.0,
            "is_stimming": False,
            "all_probs":   {c: 0.0 for c in CLASSES},
        }
