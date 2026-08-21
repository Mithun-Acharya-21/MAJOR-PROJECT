import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchHealth, fetchModelInfo } from "@/api/client";
import type { HealthResponse, ModelInfoResponse } from "@/types/api";
import { Disclaimer } from "@/components/Disclaimer";

const CARDS = [
  { to: "/text",       icon: "💬", title: "Text Analysis",      desc: "Analyse written text through the research NLP pipeline." },
  { to: "/voice",      icon: "🎙️", title: "Voice Analysis",     desc: "Record or upload audio for speech and acoustic analysis." },
  { to: "/facial",     icon: "🖼️",  title: "Facial Analysis",    desc: "Upload an image for facial expression analysis." },
  { to: "/multimodal", icon: "🔀", title: "Multimodal",         desc: "Combine text, voice, and image for fusion analysis." },
  { to: "/explain",    icon: "🔍", title: "Explainability",     desc: "Inspect LIME, SHAP, Grad-CAM, and gate attribution views." },
  { to: "/suggest",    icon: "💚", title: "Suggestions",        desc: "View supportive wellness suggestions based on signals." },
  { to: "/model",      icon: "📊", title: "Model Info",         desc: "Current model version, readiness, and supported modalities." },
  { to: "/ethics",     icon: "🔒", title: "Ethics & Privacy",   desc: "Limitations, privacy controls, and crisis resources." },
];

export default function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
    fetchModelInfo().then(setInfo).catch(() => setInfo(null));
  }, []);

  return (
    <main style={{ maxWidth: 960, margin: "0 auto", padding: "32px 16px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 4 }}>
        EmotiSense<span style={{ color: "#2563eb" }}>AI</span>
      </h1>
      <p style={{ color: "#6b7280", marginBottom: 24 }}>
        Multimodal early-risk research screening tool
      </p>

      <Disclaimer />

      <div style={{ display: "flex", gap: 12, marginBottom: 24, flexWrap: "wrap" }}>
        <StatusPill
          label="Backend"
          ok={health?.status === "healthy"}
          value={health ? `${health.environment}` : "checking…"}
        />
        <StatusPill
          label="Model"
          ok={!!info}
          value={info ? info.model_version : "checking…"}
        />
        {info && (
          <StatusPill
            label="Readiness"
            ok={false}
            value={info.readiness.replace(/_/g, " ")}
          />
        )}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: 16,
        }}
      >
        {CARDS.map(({ to, icon, title, desc }) => (
          <Link
            key={to}
            to={to}
            style={{
              textDecoration: "none",
              display: "flex",
              flexDirection: "column",
              gap: 8,
              padding: 18,
              borderRadius: 10,
              border: "1px solid #e2e8f0",
              background: "#fff",
              transition: "box-shadow 0.15s",
            }}
            onMouseEnter={(e) =>
              ((e.currentTarget as HTMLAnchorElement).style.boxShadow =
                "0 4px 16px rgba(0,0,0,0.10)")
            }
            onMouseLeave={(e) =>
              ((e.currentTarget as HTMLAnchorElement).style.boxShadow = "none")
            }
          >
            <span style={{ fontSize: 28 }} role="img" aria-hidden="true">{icon}</span>
            <span style={{ fontWeight: 600, color: "#1e293b", fontSize: 15 }}>{title}</span>
            <span style={{ fontSize: 13, color: "#64748b", lineHeight: 1.4 }}>{desc}</span>
          </Link>
        ))}
      </div>
    </main>
  );
}

function StatusPill({ label, ok, value }: { label: string; ok: boolean; value: string }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "4px 12px",
        borderRadius: 20,
        background: ok ? "#dcfce7" : "#fef3c7",
        border: `1px solid ${ok ? "#86efac" : "#fcd34d"}`,
        fontSize: 13,
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: ok ? "#22c55e" : "#f59e0b",
        }}
      />
      <strong>{label}:</strong> {value}
    </div>
  );
}
