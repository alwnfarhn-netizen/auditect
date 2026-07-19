"""
synthesis/synthesizer.py  —  Multi-Frame Synthesis & Error Correction
======================================================================
Menghaluskan hasil prediksi dari model agar tidak "flickering" (berubah-ubah
setiap frame). Tahap ini identik dengan tahap "Synthesis" di repo referensi,
yang mengakumulasi huruf-huruf yang terdeteksi sebelum dikirim ke LLM.

Teknik:
  1. Voting window  — ambil label mayoritas dari N prediksi terakhir
  2. Debounce       — abaikan deteksi yang tidak konsisten (< min_votes)
  3. Cooldown       — setelah alert stimming, tunggu cooldown_sec sebelum
                      alert berikutnya (mencegah spam notifikasi)
"""

import time
from collections import deque, Counter
from typing import Dict

from model.pointnet import CLASSES


class Synthesizer:
    """
    Smoothing multi-frame untuk mencegah false positive dan flickering.

    Args:
        window:               Jumlah prediksi yang di-voting
        confidence_threshold: Minimum confidence untuk menerima prediksi
        min_votes:            Minimum frame dalam window yang harus setuju
        cooldown_sec:         Jarak minimum (detik) antara dua alert stimming
    """

    def __init__(
        self,
        window:               int   = 20,
        confidence_threshold: float = 0.65,
        min_votes:            int   = 10,
        cooldown_sec:         float = 3.0,
    ):
        self._window    = window
        self._threshold = confidence_threshold
        self._min_votes = min_votes
        self._cooldown  = cooldown_sec

        self._history:       deque = deque(maxlen=window)
        self._last_alert_ts: float = 0.0

    def push(self, raw_result: Dict) -> Dict:
        """
        Masukkan satu hasil prediksi mentah.
        Kembalikan hasil yang sudah dihaluskan.
        """
        # Hanya masukkan ke history jika confidence cukup
        if raw_result["confidence"] >= self._threshold:
            self._history.append(raw_result["label"])

        # Jika history masih terlalu pendek, tunggu dulu
        if len(self._history) < self._min_votes:
            return {**raw_result, "is_stimming": False}

        # Voting: label mana yang paling sering muncul?
        vote_counts = Counter(self._history)
        voted_label, voted_count = vote_counts.most_common(1)[0]

        # Tolak jika tidak ada mayoritas yang cukup kuat
        if voted_count < self._min_votes:
            return {**raw_result, "label": "normal", "is_stimming": False}

        # Terapkan cooldown agar tidak spam alert
        is_stimming = voted_label != "normal"
        now = time.time()
        if is_stimming and (now - self._last_alert_ts) < self._cooldown:
            is_stimming = False   # masih dalam cooldown
        elif is_stimming:
            self._last_alert_ts = now   # update waktu alert terakhir

        return {
            **raw_result,
            "label":       voted_label,
            "is_stimming": is_stimming,
        }

    def reset(self):
        self._history.clear()
        self._last_alert_ts = 0.0
