/**
 * components/Dashboard.tsx
 * ─────────────────────────
 * Panel kanan: probabilitas semua kelas + riwayat deteksi real-time.
 */

import React, { useState, useEffect } from "react";
import { DetectionResult, LABEL_ID } from "../hooks/useWebSocket";

interface Props {
  result: DetectionResult | null;
  wsConnected: boolean;
}

interface HistoryEntry {
  label:      string;
  confidence: number;
  time:       string;
}

const LABEL_COLORS: Record<string, string> = {
  normal:        "#22c55e",
  body_rocking:  "#f97316",
  hand_flapping: "#ef4444",
  head_banging:  "#a855f7",
  spinning:      "#3b82f6",
};

export default function Dashboard({ result, wsConnected }: Props) {
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    if (result?.is_stimming) {
      setHistory((prev) => [
        {
          label:      result.label,
          confidence: result.confidence,
          time:       new Date().toLocaleTimeString("id-ID"),
        },
        ...prev.slice(0, 49),   // simpan 50 terakhir
      ]);
    }
  }, [result]);

  const probs = result?.all_probs ?? {};

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      {/* Status server */}
      <div style={{
        background: "#1e293b", borderRadius: 12, padding: "12px 16px",
        display: "flex", alignItems: "center", gap: 10,
      }}>
        <span style={{
          width: 10, height: 10, borderRadius: "50%",
          background: wsConnected ? "#22c55e" : "#ef4444",
          display: "inline-block",
        }}/>
        <span style={{ color: "#94a3b8", fontSize: 13 }}>
          {wsConnected ? "Terhubung ke server" : "Mencoba terhubung..."}
        </span>
      </div>

      {/* Probabilitas semua kelas */}
      <div style={{ background: "#1e293b", borderRadius: 12, padding: 16 }}>
        <h3 style={{ color: "#e2e8f0", margin: "0 0 14px", fontSize: 14, fontWeight: 600 }}>
          Probabilitas Deteksi
        </h3>
        {Object.entries(LABEL_ID)
          .filter(([k]) => k in (probs as object) || k === "normal")
          .slice(0, 5)
          .map(([key, name]) => {
            const p     = (probs[key] ?? 0) * 100;
            const color = LABEL_COLORS[key] ?? "#6b7280";
            return (
              <div key={key} style={{ marginBottom: 10 }}>
                <div style={{
                  display: "flex", justifyContent: "space-between",
                  color: "#cbd5e1", fontSize: 13, marginBottom: 4,
                }}>
                  <span>{name}</span>
                  <span style={{ color }}>{p.toFixed(1)}%</span>
                </div>
                <div style={{ background: "#0f172a", borderRadius: 4, height: 6 }}>
                  <div style={{
                    width: `${p}%`, height: "100%", borderRadius: 4,
                    background: color,
                    transition: "width 0.3s ease",
                  }} />
                </div>
              </div>
            );
          })}
      </div>

      {/* Riwayat stimming */}
      <div style={{ background: "#1e293b", borderRadius: 12, padding: 16, flex: 1 }}>
        <h3 style={{ color: "#e2e8f0", margin: "0 0 12px", fontSize: 14, fontWeight: 600 }}>
          Riwayat Stimming
        </h3>
        {history.length === 0 ? (
          <p style={{ color: "#475569", fontSize: 13, margin: 0 }}>
            Belum ada deteksi stimming...
          </p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 280, overflowY: "auto" }}>
            {history.map((h, i) => (
              <div key={i} style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                background: "#0f172a", borderRadius: 8, padding: "8px 12px",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: "50%",
                    background: LABEL_COLORS[h.label] ?? "#6b7280",
                    display: "inline-block",
                  }} />
                  <span style={{ color: "#e2e8f0", fontSize: 13 }}>
                    {LABEL_ID[h.label] ?? h.label}
                  </span>
                </div>
                <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                  <span style={{ color: "#64748b", fontSize: 12 }}>
                    {(h.confidence * 100).toFixed(0)}%
                  </span>
                  <span style={{ color: "#475569", fontSize: 11 }}>{h.time}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

    </div>
  );
}
