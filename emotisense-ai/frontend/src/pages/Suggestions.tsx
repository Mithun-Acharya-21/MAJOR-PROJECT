import { Disclaimer } from "@/components/Disclaimer";

const CATEGORIES = [
  {
    id: "sleep_hygiene",
    icon: "🌙",
    label: "Sleep Hygiene",
    suggestion: "Consider keeping a consistent sleep schedule, even on weekends.",
  },
  {
    id: "social_connection",
    icon: "🤝",
    label: "Social Connection",
    suggestion: "Reaching out to a trusted friend or family member, even briefly, can help.",
  },
  {
    id: "stress_management",
    icon: "🧘",
    label: "Stress Management",
    suggestion: "Short breaks during demanding tasks — even 5 minutes — can help reset focus.",
  },
  {
    id: "physical_activity",
    icon: "🚶",
    label: "Physical Activity",
    suggestion: "A short walk, even indoors, can gently lift mood for some people.",
  },
  {
    id: "mindfulness_grounding",
    icon: "💨",
    label: "Mindfulness & Grounding",
    suggestion:
      "Slow, deliberate breathing — breathing in for 4 counts and out for 6 — is a simple grounding tool.",
  },
  {
    id: "professional_support",
    icon: "🏥",
    label: "Professional Support",
    suggestion: "Speaking with a mental health professional can provide personalised support.",
  },
];

export default function Suggestions() {
  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: "32px 16px" }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 8 }}>
        Supportive Suggestions
      </h2>
      <p style={{ color: "#6b7280", marginBottom: 16, fontSize: 14 }}>
        These are general wellness suggestions based on model-observed signals.
        They are drawn from a fixed, approved set of categories.
      </p>
      <Disclaimer text="These are general supportive suggestions. They are not medical advice, diagnosis, or treatment. Please consult a qualified professional for personal support." />

      {/* Crisis banner — always visible */}
      <div
        role="note"
        aria-label="Crisis support information"
        style={{
          background: "#7c3aed",
          color: "#fff",
          borderRadius: 8,
          padding: "14px 16px",
          marginBottom: 20,
        }}
      >
        <strong style={{ fontSize: 15 }}>Crisis Support</strong>
        <p style={{ margin: "6px 0 0", fontSize: 14, lineHeight: 1.5 }}>
          If you are in immediate danger or at risk of harming yourself or others, please contact
          emergency services (999, 911, or your local emergency number) immediately.
          For non-urgent support, visit{" "}
          <a
            href="https://findahelpline.com"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "#c4b5fd" }}
          >
            findahelpline.com
          </a>{" "}
          to find a local crisis line.
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
          gap: 14,
        }}
      >
        {CATEGORIES.map(({ id, icon, label, suggestion }) => (
          <div
            key={id}
            style={{
              background: "#f0fdf4",
              border: "1px solid #bbf7d0",
              borderRadius: 10,
              padding: 16,
            }}
          >
            <div style={{ fontSize: 28, marginBottom: 8 }} aria-hidden="true">
              {icon}
            </div>
            <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 6, color: "#166534" }}>
              {label}
            </h3>
            <p style={{ fontSize: 14, margin: 0, color: "#374151", lineHeight: 1.5 }}>
              {suggestion}
            </p>
          </div>
        ))}
      </div>
    </main>
  );
}
