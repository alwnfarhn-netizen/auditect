"""
main.py  —  AuDetect | Autism Behavior Detection System  |  Server Entry Point
===============================================================================
FastAPI server dengan WebSocket untuk komunikasi real-time antara
kamera browser (client) dan pipeline deteksi stimming (server).

Alur:
  Browser (kamera)  →[WebSocket frame]→  Server  →[WebSocket result]→  Browser (overlay)

Jalankan:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import json
import time
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from pose.extractor   import PoseExtractor
from pose.normalizer  import normalize_keypoints
from model.classifier import StimmingClassifier
from synthesis.synthesizer import Synthesizer
from database.logger  import DetectionLogger


# ── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AuDetect API",
    description="Autism Behavior Detection System — Real-time stimming detection via PointNet + LSTM",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (diinisialisasi sekali saat server start) ───────────────────
extractor   = PoseExtractor()
classifier  = StimmingClassifier(weights_path="weights/stimming_model.keras")
synthesizer = Synthesizer(window=20, confidence_threshold=0.65)
logger      = DetectionLogger(db_path="detections.db")


# ── REST Endpoints ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "online", "service": "AuDetect"}


@app.get("/history")
def get_history(limit: int = 100):
    """Ambil log deteksi terakhir untuk dashboard."""
    return logger.get_recent(limit)


@app.get("/stats")
def get_stats():
    """Statistik agregat per label."""
    return logger.get_stats()


# ── WebSocket Endpoint ──────────────────────────────────────────────────────
@app.websocket("/ws/detect")
async def detect_websocket(ws: WebSocket):
    """
    Protocol:
      Client  →  Server : JSON { "frame": "<base64 JPEG>" }
      Server  →  Client : JSON { "label", "confidence", "is_stimming",
                                 "all_probs", "landmarks", "timestamp" }
    """
    await ws.accept()
    print(f"[WS] Client terhubung: {ws.client}")

    synthesizer.reset()

    try:
        while True:
            # 1. Terima frame dari browser
            data = await ws.receive_text()
            payload = json.loads(data)

            if "frame" not in payload:
                continue

            # 2. Decode base64 → numpy array BGR
            img_bytes  = base64.b64decode(payload["frame"])
            img_array  = np.frombuffer(img_bytes, dtype=np.uint8)
            frame_bgr  = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                continue

            # 3. Ekstraksi pose keypoints (33 titik x 3)
            pose_result = extractor.process_frame(frame_bgr, time.time())

            if pose_result is None:
                await ws.send_text(json.dumps({
                    "label": "no_pose",
                    "confidence": 0.0,
                    "is_stimming": False,
                    "all_probs": {},
                    "landmarks": [],
                    "timestamp": time.time(),
                }))
                continue

            # 4. Normalisasi keypoints (relatif terhadap bahu-pinggul)
            normalized_kp = normalize_keypoints(pose_result.keypoints)

            # 5. Masukkan ke classifier (PointNet → LSTM)
            raw_result = classifier.predict(normalized_kp)

            # 6. Synthesis: smoothing multi-frame agar tidak flickering
            final_result = synthesizer.push(raw_result)

            # 7. Log ke database jika stimming terdeteksi
            if final_result["is_stimming"]:
                logger.log(final_result)

            # 8. Kirim hasil ke client
            response = {
                **final_result,
                "landmarks": pose_result.keypoints_list,   # list of {x,y,z,v}
                "timestamp": time.time(),
            }
            await ws.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print(f"[WS] Client terputus: {ws.client}")
    except Exception as e:
        print(f"[WS] Error: {e}")
