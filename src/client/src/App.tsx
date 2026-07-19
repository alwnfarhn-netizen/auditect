/**
 * App.tsx  —  AuDetect  |  Main Interface
 * ─────────────────────────────────────────────────────────
 * Layout utama: kamera (kiri) + dashboard deteksi (kanan)
 */

import React, { useRef } from "react";
import Camera    from "./components/Camera";
import Dashboard from "./components/Dashboard";
import { useWebSocket } from "./hooks/useWebSocket";

const WS_URL = "ws://localhost:8000/ws/detect";

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const { result, connected, error } = useWebSocket(WS_URL, videoRef);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0f172a",
      color: "#e2e8f0",
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
    }}>

      {/* Header */}
      <header style={{
        borderBottom: "1px solid #1e293b",
        padding: "16px 32px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#f1f5f9" }}>
            🧩 AuDetect
          </h1>
          <p style={{ margin: "2px 0 0", fontSize: 12, color: "#64748b" }}>
            Autism Behavior Detection System — PointNet + LSTM
          </p>
        </div>
        <div style={{
          fontSize: 12, color: connected ? "#22c55e" : "#ef4444",
          display: "flex", alignItems: "center", gap: 6,
        }}>
          <span style={{
            width: 8, height: 8, borderRadius: "50%",
            background: connected ? "#22c55e" : "#ef4444",
            display: "inline-block",
          }}/>
          {connected ? "Server Online" : "Menghubungkan..."}
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div style={{
          background: "#7f1d1d", color: "#fca5a5",
          padding: "10px 32px", fontSize: 13,
        }}>
          ⚠ {error}
        </div>
      )}

      {/* Main content */}
      <main style={{
        display: "grid",
        gridTemplateColumns: "1fr 340px",
        gap: 24,
        padding: 24,
        maxWidth: 1280,
        margin: "0 auto",
      }}>
        {/* Kamera + skeleton overlay */}
        <Camera videoRef={videoRef} result={result} />

        {/* Panel deteksi */}
        <Dashboard result={result} wsConnected={connected} />
      </main>

      {/* Footer */}
      <footer style={{
        textAlign: "center", padding: "16px 32px",
        color: "#334155", fontSize: 12, borderTop: "1px solid #1e293b",
      }}>
        AuDetect — Autism Behavior Detection System | MediaPipe × PointNet × LSTM
      </footer>
    </div>
  );
}
