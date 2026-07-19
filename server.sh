#!/bin/bash
# server.sh — Jalankan Python server (FastAPI + WebSocket)
# Pastikan sudah: pip install -r requirements.txt

echo "╔══════════════════════════════════════════════╗"
echo "║   AuDetect — Autism Behavior Detection       ║"
echo "║              Python Server                   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

cd src/server

# Aktifkan virtual environment jika ada
if [ -d "../../.venv" ]; then
    source ../../.venv/bin/activate
    echo "[INFO] Virtual environment aktif"
fi

echo "[INFO] Memulai server di http://localhost:8000"
echo "[INFO] WebSocket endpoint: ws://localhost:8000/ws/detect"
echo "[INFO] Tekan Ctrl+C untuk berhenti"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
