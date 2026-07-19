/**
 * components/Camera.tsx
 * ──────────────────────
 * Komponen kamera:
 *   1. Akses webcam browser via getUserMedia
 *   2. Tampilkan video live
 *   3. Gambar skeleton overlay dari landmarks yang dikembalikan server
 *   4. Tampilkan status deteksi (label + confidence) di atas video
 */

import React, { useEffect, useRef } from "react";
import { DetectionResult, LABEL_ID } from "../hooks/useWebSocket";

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  result:   DetectionResult | null;
}

// ── Koneksi MediaPipe landmark (pasangan landmark yang dihubungkan) ──────
const CONNECTIONS: [number, number][] = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],[23,25],[24,26],
  [25,27],[26,28],[0,11],[0,12],
];

const LABEL_COLORS: Record<string, string> = {
  normal:        "#22c55e",
  body_rocking:  "#f97316",
  hand_flapping: "#ef4444",
  head_banging:  "#a855f7",
  spinning:      "#3b82f6",
  menunggu:      "#6b7280",
  no_pose:       "#6b7280",
};

export default function Camera({ videoRef, result }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // ── Mulai stream kamera ──────────────────────────────────────────────────
  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 } })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      })
      .catch((e) => console.error("Kamera tidak bisa diakses:", e));
  }, [videoRef]);

  // ── Gambar skeleton + label di canvas setiap kali result berubah ─────────
  useEffect(() => {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video || !result) return;

    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const lm    = result.landmarks;
    const color = LABEL_COLORS[result.label] || "#ffffff";
    const W     = canvas.width;
    const H     = canvas.height;

    // Gambar koneksi skeleton
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.globalAlpha = 0.75;
    for (const [a, b] of CONNECTIONS) {
      if (!lm[a] || !lm[b]) continue;
      if (lm[a].v < 0.4 || lm[b].v < 0.4) continue;
      ctx.beginPath();
      ctx.moveTo(lm[a].x * W, lm[a].y * H);
      ctx.lineTo(lm[b].x * W, lm[b].y * H);
      ctx.stroke();
    }

    // Gambar titik landmark
    ctx.fillStyle   = "#ffffff";
    ctx.globalAlpha = 0.9;
    for (const p of lm) {
      if (p.v < 0.4) continue;
      ctx.beginPath();
      ctx.arc(p.x * W, p.y * H, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1.0;
  }, [result, videoRef]);

  const label = result ? (LABEL_ID[result.label] ?? result.label) : "Memuat...";
  const conf  = result ? (result.confidence * 100).toFixed(1) : "0.0";
  const color = LABEL_COLORS[result?.label ?? "menunggu"] ?? "#6b7280";
  const isStimming = result?.is_stimming ?? false;

  return (
    <div style={{ position: "relative", borderRadius: 16, overflow: "hidden",
                  border: isStimming ? "3px solid #ef4444" : "2px solid #334155" }}>
      {/* Video feed */}
      <video
        ref={videoRef}
        style={{ width: "100%", display: "block", transform: "scaleX(-1)" }}
        muted
        playsInline
      />

      {/* Skeleton overlay canvas */}
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
          transform: "scaleX(-1)", pointerEvents: "none",
        }}
      />

      {/* Status badge */}
      <div style={{
        position: "absolute", top: 12, left: 12,
        background: "rgba(0,0,0,0.72)", borderRadius: 10,
        padding: "8px 16px", display: "flex", alignItems: "center", gap: 8,
      }}>
        <span style={{
          width: 10, height: 10, borderRadius: "50%",
          background: color, display: "inline-block",
          boxShadow: isStimming ? `0 0 8px ${color}` : "none",
        }} />
        <span style={{ color, fontWeight: 600, fontSize: 15 }}>{label}</span>
        <span style={{ color: "#94a3b8", fontSize: 13 }}>{conf}%</span>
      </div>

      {/* Alert banner */}
      {isStimming && (
        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0,
          background: "rgba(239,68,68,0.85)", color: "#fff",
          padding: "8px 16px", textAlign: "center", fontWeight: 600,
          fontSize: 14, letterSpacing: 1,
        }}>
          ⚠ STIMMING TERDETEKSI — {label.toUpperCase()}
        </div>
      )}
    </div>
  );
}
