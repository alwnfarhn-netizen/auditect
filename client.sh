#!/bin/bash
# client.sh — Jalankan React client
# Pastikan sudah: npm install (di folder src/client)

echo "╔══════════════════════════════════════════════╗"
echo "║   AuDetect — Autism Behavior Detection       ║"
echo "║              React Client                    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

cd src/client

if [ ! -d "node_modules" ]; then
    echo "[INFO] node_modules tidak ditemukan. Menjalankan npm install..."
    npm install
fi

echo "[INFO] Memulai client di http://localhost:3000"
echo "[INFO] Pastikan server.sh sudah berjalan terlebih dahulu"
echo ""

npm start
