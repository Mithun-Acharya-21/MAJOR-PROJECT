/**
 * Renders supportive suggestions from the policy engine.
 * Crisis suggestions rendered with high-contrast styling.
 * Disclaimer always shown.
 */

interface Suggestion {
  category: string;
  text: string;
}

interface SuggestionListProps {
  suggestions: Suggestion[];
  isCrisis?: boolean;
  policyDisclaimer?: string | null;
}

const CRISIS_DISCLAIMER =
  "If you are in immediate danger, please contact emergency services or a crisis line immediately.";

export function SuggestionList({
  suggestions,
  isCrisis = false,
  policyDisclaimer,
}: SuggestionListProps) {
  if (suggestions.length === 0) return null;

  return (
    <section aria-label="Supportive suggestions" style={{ marginTop: 24 }}>
      {isCrisis && (
        <div
          role="alert"
          style={{
            background: "#7c3aed",
            color: "#fff",
            borderRadius: 8,
            padding: "12px 16px",
            marginBottom: 12,
            fontWeight: 700,
            fontSize: 15,
          }}
        >
          {CRISIS_DISCLAIMER}
        </div>
      )}
      <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 10 }}>
        {isCrisis ? "Crisis Support Resources" : "Supportive Suggestions"}
      </h3>
      <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 8 }}>
        {suggestions.map((s, i) => (
          <li
            key={i}
            style={{
              background: isCrisis ? "#f3f0ff" : "#f0fdf4",
              border: `1px solid ${isCrisis ? "#c4b5fd" : "#bbf7d0"}`,
              borderRadius: 8,
              padding: "10px 14px",
              fontSize: 14,
            }}
          >
            <span
              style={{
                display: "inline-block",
                fontSize: 11,
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color: "#6b7280",
                marginBottom: 4,
              }}
            >
              {s.category.replace(/_/g, " ")}
            </span>
            <p style={{ margin: 0 }}>{s.text}</p>
          </li>
        ))}
      </ul>
      {policyDisclaimer && (
        <p style={{ fontSize: 12, color: "#6b7280", marginTop: 10, fontStyle: "italic" }}>
          {policyDisclaimer}
        </p>
      )}
    </section>
  );
}
