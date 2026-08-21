import { useCallback, useState } from "react";
import { analyzeText } from "@/api/client";
import { useAnalysis } from "@/hooks/useAnalysis";
import { Disclaimer } from "@/components/Disclaimer";
import { RiskBadge } from "@/components/RiskBadge";
import { ConsentGate } from "@/components/ConsentGate";
import type { RiskTier } from "@/types/api";

export default function TextAnalysis() {
  const [consented, setConsented] = useState(false);
  const [text, setText] = useState("");
  const { data, loading, error, run } = useAnalysis(analyzeText);

  const handleSubmit = useCallback(() => {
    if (text.trim()) run(text.trim());
  }, [text, run]);

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: "32px 16px" }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 16 }}>Text Analysis</h2>
      <Disclaimer />

      {!consented ? (
        <ConsentGate modality="text" onConsent={() => setConsented(true)} />
      ) : (
        <>
          <label htmlFor="text-input" style={{ display: "block", marginBottom: 6, fontWeight: 600 }}>
            Enter text to analyse
          </label>
          <textarea
            id="text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={6}
            maxLength={10000}
            placeholder="Type or paste text here…"
            aria-label="Text input for analysis"
            style={{
              width: "100%",
              padding: "10px 12px",
              borderRadius: 8,
              border: "1px solid #cbd5e1",
              fontSize: 14,
              resize: "vertical",
              boxSizing: "border-box",
            }}
          />
          <div style={{ display: "flex", gap: 10, marginTop: 10, alignItems: "center" }}>
            <button
              onClick={handleSubmit}
              disabled={loading || !text.trim()}
              style={{
                padding: "8px 22px",
                background: !loading && text.trim() ? "#2563eb" : "#9ca3af",
                color: "#fff",
                border: "none",
                borderRadius: 6,
                cursor: loading || !text.trim() ? "not-allowed" : "pointer",
                fontWeight: 600,
                fontSize: 14,
              }}
            >
              {loading ? "Analysing…" : "Analyse"}
            </button>
            <span style={{ fontSize: 13, color: "#6b7280" }}>
              {text.length} / 10 000 characters
            </span>
          </div>

          {error && (
            <p role="alert" style={{ color: "#ef4444", marginTop: 12, fontSize: 14 }}>
              Error: {error}
            </p>
          )}

          {data && (
            <section aria-label="Analysis results" style={{ marginTop: 24 }}>
              <RiskBadge
                tier={data.risk_tier as RiskTier}
                score={data.risk_score}
                uncertainty={data.uncertainty}
              />
              {!data.model_available && (
                <p style={{ marginTop: 12, fontSize: 14, color: "#6b7280" }}>{data.note}</p>
              )}
              {data.top_tokens.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <h4 style={{ fontSize: 14, marginBottom: 6 }}>Top tokens (model attribution)</h4>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {data.top_tokens.map((t) => (
                      <span
                        key={t}
                        style={{
                          background: "#eff6ff",
                          border: "1px solid #bfdbfe",
                          borderRadius: 4,
                          padding: "2px 8px",
                          fontSize: 13,
                        }}
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                  <p style={{ fontSize: 12, color: "#6b7280", marginTop: 6 }}>
                    Model-observed tokens — not clinical indicators.
                  </p>
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
