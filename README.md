# 🧩 AuDetect
## Autism Behavior Detection System
### Deteksi Gerakan Berulang (Stimming) Anak Autis dalam Kelas Berbasis PointNet + LSTM

---

## Deskripsi Proyek

Sistem ini mendeteksi perilaku stimming (gerakan tubuh berulang) pada anak dengan Autism Spectrum Disorder (ASD) secara real-time menggunakan kamera kelas biasa. Sistem memanfaatkan **pose estimation** untuk mengekstrak keypoint tubuh, kemudian mengklasifikasikan pola gerakan temporal menggunakan arsitektur **PointNet + LSTM**.

Stimming yang dideteksi meliputi:
| Label | Perilaku |
|---|---|
| `body_rocking` | Goyang tubuh maju-mundur |
| `hand_flapping` | Kibas / tepuk tangan berulang |
| `head_banging` | Benturan kepala ke objek |
| `spinning` | Berputar di tempat |
| `normal` | Aktivitas normal (non-stimming) |

---

## Arsitektur Sistem

```
┌────────────────────────────────────────────────────────────────────────┐
│  Browser / Kamera Kelas                                                │
│  ┌──────────┐   WebSocket (frame)    ┌──────────────────────────────┐ │
│  │  React   │ ──────────────────────▶│  FastAPI Server (Python)     │ │
│  │  Client  │ ◀──────────────────────│                              │ │
│  │  (UI +   │   WebSocket (result)   │  MediaPipe Pose Extraction   │ │
│  │ Overlay) │                        │       ↓                      │ │
│  └──────────┘                        │  Normalisasi Keypoints       │ │
│                                      │       ↓                      │ │
│                                      │  PointNet Encoder            │ │
│                                      │  (per-frame embedding)       │ │
│                                      │       ↓                      │ │
│                                      │  LSTM Temporal Classifier    │ │
│                                      │       ↓                      │ │
│                                      │  Synthesis + Smoothing       │ │
│                                      │       ↓                      │ │
│                                      │  SQLite Logger               │ │
│                                      └──────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### Stack Teknologi

| Layer | Teknologi |
|---|---|
| Pose Estimation | MediaPipe Pose Landmark (33 keypoints) |
| Feature Extraction | PointNet (adaptasi dari PointNet paper, Qi et al. 2017) |
| Temporal Model | Bidirectional LSTM |
| Server | FastAPI + WebSocket (Python) |
| Client | React + TypeScript |
| Database | SQLite (logging deteksi) |

---

## Struktur Proyek

```
audetect/
│
├── src/
│   ├── server/                     ← Python backend
│   │   ├── main.py                 ← FastAPI + WebSocket server
│   │   ├── collect.py              ← Script pengumpulan data training
│   │   ├── train.py                ← Script training model
│   │   │
│   │   ├── pose/
│   │   │   ├── extractor.py        ← MediaPipe pose extraction
│   │   │   └── normalizer.py       ← Normalisasi keypoints (torso-relative)
│   │   │
│   │   ├── model/
│   │   │   ├── pointnet.py         ← Arsitektur PointNet (encoder + T-Net)
│   │   │   ├── temporal.py         ← PointNet + LSTM combined model
│   │   │   └── classifier.py       ← Sliding window inference pipeline
│   │   │
│   │   ├── synthesis/
│   │   │   └── synthesizer.py      ← Multi-frame voting + debounce
│   │   │
│   │   ├── database/
│   │   │   └── logger.py           ← SQLite detection logging
│   │   │
│   │   └── weights/                ← Folder model (.keras) — isi setelah training
│   │
│   └── client/                     ← React frontend
│       ├── src/
│       │   ├── App.tsx             ← Layout utama
│       │   ├── index.tsx           ← Entry point
│       │   ├── hooks/
│       │   │   └── useWebSocket.ts ← WebSocket hook (kirim frame, terima hasil)
│       │   └── components/
│       │       ├── Camera.tsx      ← Video feed + skeleton overlay
│       │       └── Dashboard.tsx   ← Probabilitas + riwayat deteksi
│       └── package.json
│
├── requirements.txt                ← Dependensi Python
├── server.sh                       ← Script jalankan server
├── client.sh                       ← Script jalankan client
└── README.md
```

---

## Instalasi & Menjalankan

### Prasyarat
- Python 3.10+
- Node.js 18+
- Webcam atau kamera IP

### 1. Clone & Install

```bash
git clone https://github.com/<username>/audetect.git
cd audetect

# Python dependencies
pip install -r requirements.txt

# Node.js dependencies
cd src/client && npm install && cd ../..
```

### 2. Kumpulkan Data Training

```bash
cd src/server

# Rekam setiap kelas (tekan SPASI untuk mulai/stop, Q untuk simpan)
python collect.py record --label normal        --samples 300
python collect.py record --label body_rocking  --samples 300
python collect.py record --label hand_flapping --samples 300
python collect.py record --label head_banging  --samples 200
python collect.py record --label spinning      --samples 200

# Gabungkan semua rekaman
python collect.py merge --dir dataset/raw --out dataset.npz
```

### 3. Training Model

```bash
cd src/server
python train.py --dataset dataset.npz --epochs 80
# Model tersimpan di: weights/stimming_model.keras
```

### 4. Jalankan Sistem

```bash
# Terminal 1 — Server
bash server.sh

# Terminal 2 — Client
bash client.sh
```

Buka browser: **http://localhost:3000**

---

## Pipeline Teknis

### A. Pose Estimation (MediaPipe)
Sistem menggunakan **MediaPipe Pose Landmark** yang menghasilkan 33 titik keypoint 3D per frame secara real-time. Berbeda dari pendekatan berbasis gambar (image classification), penggunaan keypoint koordinat membuat sistem:
- **Tidak bergantung latar belakang** — hanya memproses koordinat, bukan piksel
- **Tidak bergantung warna kulit** — tidak ada bias warna
- **Efisien secara komputasi** — hanya memproses 33×3 = 99 angka per frame

### B. Normalisasi Keypoints
Keypoints dinormalisasi relatif terhadap **jarak bahu-ke-pinggul** (panjang torso) sebagai unit referensi, sehingga sistem tidak terpengaruh jarak anak dari kamera atau ukuran tubuh yang berbeda.

### C. PointNet Encoder
PointNet (Qi et al., 2017) dirancang untuk memproses **point cloud 3D**. Arsitektur T-Net di dalamnya mempelajari matriks transformasi yang membuat model **invariant terhadap rotasi dan translasi**. Output adalah vektor embedding 256 dimensi yang merepresentasikan postur tubuh dalam satu frame.

### D. LSTM Temporal Classifier
Sequence 60 frame (±2 detik) dari embedding PointNet diproses oleh **2-layer LSTM** untuk menangkap pola temporal — ritme, kecepatan, dan pengulangan gerakan — yang menjadi ciri khas stimming.

### E. Synthesis & Smoothing
Hasil prediksi per-window dihaluskan dengan **majority voting** pada 20 prediksi terakhir dan **cooldown** 3 detik untuk mencegah false positive dan notifikasi berulang.

---

## Referensi

- Qi, C. R., et al. (2017). *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*. CVPR 2017. https://arxiv.org/abs/1612.00593
- Thomas, K. J. (2024). *ASL Fingerspell Recognition and Semantic Pose Retrieval*. arXiv:2408.09311. https://arxiv.org/abs/2408.09311
- Lugaresi, C., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. arXiv:1906.08172.
- American Psychiatric Association. (2013). *Diagnostic and Statistical Manual of Mental Disorders, 5th edition (DSM-5)*.

---

## Label Kelas & Panduan Pengumpulan Data

| Label | Deskripsi | Tips Rekaman |
|---|---|---|
| `normal` | Duduk, berdiri, berjalan normal | Rekam berbagai aktivitas kelas biasa |
| `body_rocking` | Goyang maju-mundur atau kiri-kanan | Duduk di kursi, gerakan ritmis berulang |
| `hand_flapping` | Kibas tangan cepat di samping tubuh | Tangan di bawah bahu, gerakan cepat |
| `head_banging` | Bentur kepala ke permukaan | Simulasikan dengan gerakan mendekat ke meja |
| `spinning` | Berputar di tempat | Berdiri, putar tubuh penuh |

**Minimum sampel yang disarankan:** 200–300 per kelas  
**Variasi yang dianjurkan:** rekam dari beberapa orang dan beberapa sudut kamera

---

*AuDetect — Autism Behavior Detection System | dibuat dengan MediaPipe, PointNet, FastAPI, dan React*
