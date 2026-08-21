import { useCallback, useState } from "react";
import { analyzeMultimodal } from "@/api/client";
import { useAnalysis } from "@/hooks/useAnalysis";
import { Disclaimer } from "@/components/Disclaimer";
import { RiskBadge } from "@/components/RiskBadge";
import { EmotionChart } from "@/components/EmotionChart";
import { SuggestionList } from "@/components/SuggestionList";
import { ConsentGate } from "@/components/ConsentGate";
import type { RiskTier } from "@/types/api";

export default function MultimodalAnalysis() {
  const [consented, setConsented] = useState(false);
  const [text, setText] = useState("");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const { data, loading, error, run } = useAnalysis(analyzeMultimodal);

  const handleSubmit = useCallback(() => {
    run({ text: text.trim() || undefined, audioFile: audioFile ?? undefined, imageFile: imageFile ?? undefined });
  }, [text, audioFile, imageFile, run]);

  const hasAny = !!text.trim() || !!audioFile || !!imageFile;

  return (
    <main style={{ maxWidth: 800, margin: "0 auto", padding: "32px 16px" }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 16 }}>Multimodal Analysis</h2>
      <Disclaimer />

      {!consented ? (
        <ConsentGate modality="image_upload" onConsent={() => setConsented(true)} />
      ) : (
        <>
          <div style={{ display: "grid", gap: 20, marginBottom: 20 }}>
            {/* Text */}
            <div>
              <label htmlFor="mm-text" style={{ display: "block", fontWeight: 600, marginBottom: 6 }}>
                Text (optional)
              </label>
              <textarea
                id="mm-text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                maxLength={10000}
                placeholder="Enter text here…"
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  borderRadius: 8,
                  border: "1px solid #cbd5e1",
                  fontSize: 14,
                  resize: "vertical",
                  boxSizing: "border-box",
                }}
              />
            </div>

            {/* Audio */}
            <div>
              <label style={{ display: "block", fontWeight: 600, marginBottom: 6 }}>
                Audio (optional)
              </label>
              <input
                type="file"
                accept="audio/*"
                aria-label="Upload audio file for multimodal analysis"
                onChange={(e) => setAudioFile(e.target.files?.[0] ?? null)}
                style={{ fontSize: 14 }}
              />
            </div>

            {/* Image */}
            <div>
              <label style={{ display: "block", fontWeight: 600, marginBottom: 6 }}>
                Image (optional)
              </label>
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                aria-label="Upload image file for multimodal analysis"
                onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
                style={{ fontSize: 14 }}
              />
            </div>
          </div>

          <button
            onClick={handleSubmit}
            disabled={!hasAny || loading}
            style={{
              padding: "8px 24px",
              background: hasAny && !loading ? "#2563eb" : "#9ca3af",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              fontWeight: 600,
              fontSize: 14,
              cursor: !hasAny || loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Analysing…" : "Analyse"}
          </button>

          {error && (
            <p role="alert" style={{ color: "#ef4444", marginTop: 12, fontSize: 14 }}>
              Error: {error}
            </p>
          )}

          {data && (
            <section aria-label="Multimodal results" style={{ marginTop: 28 }}>
              {/* Modality summary */}
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
                {data.used_modalities.map((m) => (
                  <ModalityPill key={m} label={m} present />
                ))}
                {data.missing_modalities.map((m) => (
                  <ModalityPill key={m} label={m} present={false} />
                ))}
              </div>

              <RiskBadge
                tier={data.risk_tier as RiskTier}
                score={data.risk_score}
                uncertainty={data.uncertainty}
              />

              {/* Gate contributions */}
              {data.gate_contributions.length > 0 && (
                <div style={{ marginTop: 20 }}>
                  <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
                    Modality Gate Contributions
                    <span style={{ fontSize: 12, fontWeight: 400, color: "#6b7280", marginLeft: 8 }}>
                      (diagnostic signal, not causal proof)
                    </span>
                  </h4>
                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                    {data.gate_contributions.map((g) => (
                      <div
                        key={g.modality}
                        style={{
                          padding: "6px 14px",
                          borderRadius: 8,
                          background: g.active ? "#eff6ff" : "#f9fafb",
                          border: `1px solid ${g.active ? "#bfdbfe" : "#e2e8f0"}`,
                          fontSize: 13,
                          opacity: g.active ? 1 : 0.5,
                        }}
                      >
                        <strong style={{ textTransform: "capitalize" }}>{g.modality}</strong>:{" "}
                        {g.contribution_pct.toFixed(1)} %
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {data.emotion_profile && (
                <div style={{ marginTop: 20 }}>
                  <EmotionChart profile={data.emotion_profile} />
                </div>
              )}

              <SuggestionList
                suggestions={data.suggestions}
                isCrisis={data.risk_tier === "crisis"}
                policyDisclaimer={data.policy_disclaimer}
              />

              <Disclaimer text={data.disclaimer} />
            </section>
          )}
        </>
      )}
    </main>
  );
}

function ModalityPill({ label, present }: { label: string; present: boolean }) {
  return (
    <span
      style={{
        padding: "3px 10px",
        borderRadius: 20,
        background: present ? "#dcfce7" : "#f1f5f9",
        border: `1px solid ${present ? "#86efac" : "#cbd5e1"}`,
        fontSize: 13,
        textTransform: "capitalize",
        color: present ? "#15803d" : "#6b7280",
      }}
    >
      {present ? "✓" : "–"} {label}
    </span>
  );
}
