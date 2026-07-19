"""
collect.py  —  Data Collection Tool
=====================================
Script interaktif untuk merekam dan memberi label data training.
Gunakan SPASI untuk mulai/stop rekam, Q untuk keluar dan simpan.

Cara pakai:
    python collect.py --label body_rocking --source 0 --samples 300
    python collect.py --label hand_flapping --source 0 --samples 300
    python collect.py --label normal        --source 0 --samples 300
"""

import argparse
import os
import pickle
import time
import cv2
import numpy as np

from pose.extractor  import PoseExtractor
from pose.normalizer import normalize_keypoints
from model.pointnet  import CLASSES
from model.temporal  import WINDOW_SIZE


def collect(source, label: str, output_dir: str, max_samples: int = 300):
    assert label in CLASSES, f"Label harus salah satu dari: {CLASSES}"
    os.makedirs(output_dir, exist_ok=True)

    extractor  = PoseExtractor()
    buffer     = []      # menyimpan normalized keypoints per frame
    samples_X  = []      # windows yang sudah terkumpul
    samples_y  = []
    label_idx  = CLASSES.index(label)
    recording  = False
    step       = 15      # step yang sama dengan classifier

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka: {source}")

    print(f"\n[INFO] Label: '{label}'  (idx={label_idx})")
    print("[INFO] SPASI = mulai/stop  |  Q = keluar & simpan\n")

    while len(samples_X) < max_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame_disp = frame.copy()
        result = extractor.process_frame(frame, time.time())

        if result is not None:
            extractor.draw_skeleton(frame_disp)

            if recording:
                kp_norm = normalize_keypoints(result.keypoints)
                buffer.append(kp_norm)

                # Bentuk window saat buffer cukup
                if (
                    len(buffer) >= WINDOW_SIZE
                    and len(buffer) % step == 0
                ):
                    window = np.stack(buffer[-WINDOW_SIZE:], axis=0)
                    samples_X.append(window)
                    samples_y.append(label_idx)
                    print(f"  ✓ Sampel #{len(samples_X)}/{max_samples}")

        # UI overlay
        status = "● REKAM" if recording else "■ PAUSE"
        color  = (0, 0, 220) if recording else (120, 120, 120)
        cv2.rectangle(frame_disp, (0, 0), (640, 52), (20, 20, 20), -1)
        cv2.putText(frame_disp, f"  {status}   |  Label: {label}   |  Sampel: {len(samples_X)}/{max_samples}",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

        cv2.imshow("Pengumpulan Data — AuDetect", frame_disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
            buffer.clear()
            print(f"[INFO] Rekam {'MULAI ▶' if recording else 'BERHENTI ■'}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    if samples_X:
        fname = f"{label}_{int(time.time())}.pkl"
        path  = os.path.join(output_dir, fname)
        with open(path, "wb") as f:
            pickle.dump({"X": np.array(samples_X), "y": np.array(samples_y)}, f)
        print(f"\n[SIMPAN] {len(samples_X)} sampel → {path}")
    else:
        print("\n[WARN] Tidak ada sampel yang direkam.")


def merge(dataset_dir: str, save_path: str = "dataset.npz"):
    """Gabungkan semua .pkl menjadi satu file .npz untuk training."""
    all_X, all_y = [], []
    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(dataset_dir, fname), "rb") as f:
            d = pickle.load(f)
        all_X.append(d["X"])
        all_y.append(d["y"])
        label = CLASSES[d["y"][0]] if len(d["y"]) else "?"
        print(f"  {fname:40s}  {len(d['X']):4d} sampel  [{label}]")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    np.savez_compressed(save_path, X=X, y=y)
    print(f"\n[SIMPAN] Dataset gabungan: {X.shape} → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Collection — Autism Stimming")
    sub    = parser.add_subparsers(dest="cmd")

    # python collect.py record ...
    rec = sub.add_parser("record", help="Rekam data untuk satu label")
    rec.add_argument("--label",   required=True, choices=CLASSES)
    rec.add_argument("--source",  default="0")
    rec.add_argument("--output",  default="dataset/raw")
    rec.add_argument("--samples", default=300, type=int)

    # python collect.py merge ...
    mrg = sub.add_parser("merge", help="Gabungkan semua file pkl")
    mrg.add_argument("--dir",  default="dataset/raw")
    mrg.add_argument("--out",  default="dataset.npz")

    args = parser.parse_args()

    if args.cmd == "record":
        collect(args.source, args.label, args.output, args.samples)
    elif args.cmd == "merge":
        merge(args.dir, args.out)
    else:
        parser.print_help()
