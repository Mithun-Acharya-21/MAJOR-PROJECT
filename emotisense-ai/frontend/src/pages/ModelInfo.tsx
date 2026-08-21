import { useEffect, useState } from "react";
import { fetchModelInfo } from "@/api/client";
import type { ModelInfoResponse } from "@/types/api";
import { Disclaimer } from "@/components/Disclaimer";

const PHASES = [
  { n: "1–5",  label: "Text pipeline: DeBERTa-v3 + BiLSTM + attention" },
  { n: "6",    label: "Ablation evaluation: text head comparison" },
  { n: "7–8",  label: "Vision: face detection + transfer learning + Grad-CAM" },
  { n: "9–10", label: "Voice: acoustic features + speech emotion CNN-BiLSTM" },
  { n: "11",   label: "ASR: offline Whisper integration" },
  { n: "12",   label: "Aligned multimodal dataset builder (DAIC-WOZ format)" },
  { n: "13",   label: "Fusion baselines: concat, mean-pool, weighted-sum" },
  { n: "14",   label: "Reliability-aware attention fusion + temperature calibration" },
  { n: "15",   label: "Full evaluation: metrics, ablation, fairness, domain-shift" },
  { n: "16",   label: "Controlled support policy engine (crisis-first)" },
  { n: "17",   label: "Explainability: LIME, SHAP, Grad-CAM, gate views" },
  { n: "18",   label: "FastAPI backend: all routes + privacy controls" },
  { n: "19",   label: "React/TypeScript frontend + Docker" },
  { n: "20",   label: "Final benchmark, model card, viva materials" },
];

export default function ModelInfo() {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModelInfo().then(setInfo).catch((e: Error) => setError(e.message));
  }, []);

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: "32px 16px" }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 16 }}>Model Information</h2>
      <Disclaimer />

      {error && (
        <p style={{ color: "#ef4444", fontSize: 14 }}>Could not load model info: {error}</p>
      )}

      {info && (
        <div
          style={{
            background: "#fff",
            border: "1px solid #e2e8f0",
            borderRadius: 10,
            padding: 20,
            marginBottom: 24,
          }}
        >
          <InfoRow label="Version"    value={info.model_version} />
          <InfoRow label="Readiness"  value={info.readiness.replace(/_/g, " ")} />
          <InfoRow
            label="Modalities"
            value={info.supported_modalities.join(", ")}
          />
          <InfoRow
            label="Raw input stored by default"
            value={info.raw_input_storage_default ? "Yes" : "No"}
          />
          <div style={{ marginTop: 14, padding: "10px 12px", background: "#eff6ff", borderRadius: 8, fontSize: 13, color: "#1e40af" }}>
            {info.disclaimer}
          </div>
        </div>
      )}

      <h3 style={{ fontSize: 17, fontWeight: 700, marginBottom: 12 }}>
        20-Phase Implementation Roadmap
      </h3>
      <ol style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 6 }}>
        {PHASES.map(({ n, label }) => (
          <li key={n} style={{ display: "flex", gap: 12, alignItems: "flex-start", fontSize: 14 }}>
            <span
              style={{
                minWidth: 40,
                textAlign: "right",
                fontWeight: 600,
                color: "#2563eb",
                flexShrink: 0,
              }}
            >
              {n}
            </span>
            <span>{label}</span>
          </li>
        ))}
      </ol>
    </main>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", gap: 12, padding: "6px 0", borderBottom: "1px solid #f1f5f9", fontSize: 14 }}>
      <span style={{ width: 180, fontWeight: 600, color: "#374151", flexShrink: 0 }}>{label}</span>
      <span style={{ color: "#64748b" }}>{value}</span>
    </div>
  );
}
