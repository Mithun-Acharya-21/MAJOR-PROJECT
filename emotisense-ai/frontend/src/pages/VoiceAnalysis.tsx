import { useCallback, useState } from "react";
import { analyzeVoice } from "@/api/client";
import { useAnalysis } from "@/hooks/useAnalysis";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import { Disclaimer } from "@/components/Disclaimer";
import { RiskBadge } from "@/components/RiskBadge";
import { QualityBadge } from "@/components/QualityBadge";
import { EmotionChart } from "@/components/EmotionChart";
import { ConsentGate } from "@/components/ConsentGate";
import type { RiskTier } from "@/types/api";

export default function VoiceAnalysis() {
  const [consented, setConsented] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const { state, audioBlob, error: recError, start, stop, reset } = useAudioRecorder();
  const { data, loading, error, run } = useAnalysis(analyzeVoice);

  const handleAnalyse = useCallback(() => {
    const file =
      uploadFile ??
      (audioBlob ? new File([audioBlob], "recording.webm", { type: audioBlob.type }) : null);
    if (file) run(file);
  }, [uploadFile, audioBlob, run]);

  const hasInput = !!uploadFile || !!audioBlob;

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: "32px 16px" }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 16 }}>Voice Analysis</h2>
      <Disclaimer />

      {!consented ? (
        <ConsentGate modality="audio_recording" onConsent={() => setConsented(true)} />
      ) : (
        <>
          {/* Record */}
          <section aria-label="Record audio" style={{ marginBottom: 24 }}>
            <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 10 }}>Record Audio</h3>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              {state !== "recording" ? (
                <button
                  onClick={start}
                  aria-label="Start recording"
                  style={btnStyle("#dc2626")}
                >
                  ⏺ Start Recording
                </button>
              ) : (
                <button
                  onClick={stop}
                  aria-label="Stop recording"
                  style={btnStyle("#374151")}
                >
                  ⏹ Stop Recording
                </button>
              )}
              {(state === "stopped" || state === "error") && (
                <button onClick={reset} style={btnStyle("#6b7280", true)}>
                  Reset
                </button>
              )}
              <span style={{ fontSize: 13, color: state === "recording" ? "#dc2626" : "#6b7280" }}>
                {state === "recording" ? "● Recording…" : state === "stopped" ? "Recording ready" : state === "error" ? recError : "Idle"}
              </span>
            </div>
          </section>

          {/* Upload */}
          <section aria-label="Upload audio file" style={{ marginBottom: 24 }}>
            <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 8 }}>Or Upload File</h3>
            <input
              type="file"
              accept="audio/*"
              aria-label="Upload audio file"
              onChange={(e) => {
                setUploadFile(e.target.files?.[0] ?? null);
                reset();
              }}
              style={{ fontSize: 14 }}
            />
          </section>

          <button
            onClick={handleAnalyse}
            disabled={!hasInput || loading}
            style={btnStyle(!hasInput || loading ? "#9ca3af" : "#2563eb")}
          >
            {loading ? "Analysing…" : "Analyse"}
          </button>

          {error && (
            <p role="alert" style={{ color: "#ef4444", marginTop: 12, fontSize: 14 }}>
              Error: {error}
            </p>
          )}

          {data && (
            <section aria-label="Voice analysis results" style={{ marginTop: 24 }}>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
                <QualityBadge
                  score={data.audio_quality?.quality_score}
                  label="Audio quality"
                  rejectionReasons={data.audio_quality?.rejection_reasons}
                />
                <QualityBadge
                  score={data.asr_quality?.quality_score}
                  label="ASR quality"
                  rejectionReasons={data.asr_quality?.rejection_reasons}
                />
                {data.combined_quality_score != null && (
                  <QualityBadge score={data.combined_quality_score} label="Combined" />
                )}
              </div>
              {data.transcript && (
                <div style={{ background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 8, padding: "10px 14px", marginBottom: 16 }}>
                  <strong style={{ fontSize: 13 }}>Transcript ({data.language ?? "unknown language"}):</strong>
                  <p style={{ margin: "6px 0 0", fontSize: 14 }}>{data.transcript}</p>
                </div>
              )}
              <RiskBadge
                tier={data.risk_tier as RiskTier}
                score={data.risk_score}
              />
              {data.emotion_profile && (
                <div style={{ marginTop: 20 }}>
                  <EmotionChart profile={data.emotion_profile} />
                </div>
              )}
              <Disclaimer text={data.disclaimer} />
            </section>
          )}
        </>
      )}
    </main>
  );
}

function btnStyle(bg: string, small = false): React.CSSProperties {
  return {
    padding: small ? "6px 14px" : "8px 22px",
    background: bg,
    color: "#fff",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    fontWeight: 600,
    fontSize: 14,
  };
}
