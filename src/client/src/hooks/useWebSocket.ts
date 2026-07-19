/**
 * hooks/useWebSocket.ts
 * ─────────────────────
 * Hook yang mengelola koneksi WebSocket ke server,
 * mengirim frame kamera, dan menerima hasil deteksi.
 *
 * Identik dengan pola WebSocket di repo referensi yang
 * mengirim frame → menerima klasifikasi + data animasi.
 */

import { useEffect, useRef, useState, useCallback } from "react";

export interface DetectionResult {
  label:       string;
  class_id:    number;
  confidence:  number;
  is_stimming: boolean;
  all_probs:   Record<string, number>;
  landmarks:   { x: number; y: number; z: number; v: number }[];
  timestamp:   number;
}

const LABEL_ID: Record<string, string> = {
  normal:        "Normal",
  body_rocking:  "Goyang Tubuh",
  hand_flapping: "Kibas Tangan",
  head_banging:  "Bentur Kepala",
  spinning:      "Berputar",
  menunggu:      "Menunggu...",
  no_pose:       "Pose Tidak Terdeteksi",
};

export { LABEL_ID };

export function useWebSocket(wsUrl: string, videoRef: React.RefObject<HTMLVideoElement>) {
  const wsRef               = useRef<WebSocket | null>(null);
  const animFrameRef        = useRef<number>(0);
  const offscreenCanvasRef  = useRef<HTMLCanvasElement>(document.createElement("canvas"));

  const [result,    setResult]    = useState<DetectionResult | null>(null);
  const [connected, setConnected] = useState(false);
  const [error,     setError]     = useState<string | null>(null);

  const connect = useCallback(() => {
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
      console.log("[WS] Terhubung ke server");
      startSendingFrames();
    };

    ws.onmessage = (event) => {
      try {
        const data: DetectionResult = JSON.parse(event.data);
        setResult(data);
      } catch (e) {
        console.error("[WS] Parse error:", e);
      }
    };

    ws.onerror = () => setError("Koneksi WebSocket gagal. Pastikan server berjalan.");

    ws.onclose = () => {
      setConnected(false);
      cancelAnimationFrame(animFrameRef.current);
      // Auto-reconnect setelah 3 detik
      setTimeout(connect, 3000);
    };
  }, [wsUrl]);

  const startSendingFrames = useCallback(() => {
    const canvas = offscreenCanvasRef.current;
    canvas.width  = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d")!;

    const sendFrame = () => {
      const video = videoRef.current;
      if (video && video.readyState === 4 && wsRef.current?.readyState === WebSocket.OPEN) {
        ctx.drawImage(video, 0, 0, 640, 480);
        // Kirim sebagai JPEG base64 (kualitas 0.7 untuk kecepatan)
        const dataUrl  = canvas.toDataURL("image/jpeg", 0.7);
        const b64Frame = dataUrl.split(",")[1];
        wsRef.current.send(JSON.stringify({ frame: b64Frame }));
      }
      animFrameRef.current = requestAnimationFrame(sendFrame);
    };

    animFrameRef.current = requestAnimationFrame(sendFrame);
  }, [videoRef]);

  useEffect(() => {
    connect();
    return () => {
      cancelAnimationFrame(animFrameRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { result, connected, error };
}
