"""
pose/extractor.py  —  MediaPipe Pose Extraction
================================================
Mengekstrak 33 keypoint tubuh dari setiap frame video menggunakan
Google MediaPipe Pose Landmark model.

Pendekatan ini identik dengan repo referensi yang menggunakan MediaPipe
Hand Landmark (21 titik tangan), namun diadaptasi ke MediaPipe Pose
(33 titik seluruh tubuh) untuk mendeteksi gerakan tubuh keseluruhan.
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ── Indeks landmark MediaPipe Pose yang relevan ─────────────────────────────
# Referensi: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LANDMARK_NAMES = {
    0:  "nose",
    11: "left_shoulder",   12: "right_shoulder",
    13: "left_elbow",      14: "right_elbow",
    15: "left_wrist",      16: "right_wrist",
    17: "left_pinky",      18: "right_pinky",
    19: "left_index",      20: "right_index",
    21: "left_thumb",      22: "right_thumb",
    23: "left_hip",        24: "right_hip",
    25: "left_knee",       26: "right_knee",
    27: "left_ankle",      28: "right_ankle",
}

N_LANDMARKS = 33


@dataclass
class PoseFrame:
    """Hasil ekstraksi pose dari satu frame."""
    keypoints:      np.ndarray          # shape (33, 3) → x, y, z (normalized 0–1)
    visibility:     np.ndarray          # shape (33,)   → confidence per landmark
    keypoints_list: List[Dict]          # format JSON-serializable untuk client
    timestamp:      float = 0.0


class PoseExtractor:
    """
    Wrapper tipis di atas MediaPipe Pose.
    Menerima frame BGR (OpenCV), mengembalikan PoseFrame.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
        model_complexity:         int   = 1,      # 0=lite, 1=full, 2=heavy
    ):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

    # ── Public API ──────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: float = 0.0,
    ) -> Optional[PoseFrame]:
        """
        Proses satu frame BGR.

        Returns:
            PoseFrame  jika pose terdeteksi dengan cukup keypoint yang visible
            None       jika tidak ada pose / kualitas rendah
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)
        rgb.flags.writeable = True

        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks.landmark

        # Raw keypoints: koordinat sudah dinormalisasi 0–1 oleh MediaPipe
        keypoints   = np.array([[l.x, l.y, l.z]         for l in lm], dtype=np.float32)
        visibility  = np.array([l.visibility              for l in lm], dtype=np.float32)

        # Buang frame jika torso tidak terlihat dengan jelas
        torso_indices = [11, 12, 23, 24]
        if visibility[torso_indices].mean() < 0.4:
            return None

        # Format JSON-serializable untuk dikirim ke browser client
        kp_list = [
            {"x": float(lm[i].x), "y": float(lm[i].y),
             "z": float(lm[i].z), "v": float(lm[i].visibility)}
            for i in range(N_LANDMARKS)
        ]

        return PoseFrame(
            keypoints=keypoints,
            visibility=visibility,
            keypoints_list=kp_list,
            timestamp=timestamp,
        )

    def draw_skeleton(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Gambar skeleton MediaPipe di atas frame untuk debug/visualisasi."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if result.pose_landmarks:
            self._mp_draw.draw_landmarks(
                frame_bgr,
                result.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
            )
        return frame_bgr

    def close(self):
        self._pose.close()
