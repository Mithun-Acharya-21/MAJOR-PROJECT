/**
 * Safety disclaimer banner — must appear on every screen that shows
 * model output. Never hidden, never dismissible permanently.
 */

interface DisclaimerProps {
  text?: string;
  variant?: "info" | "crisis";
}

const DEFAULT_TEXT =
  "This is an AI research screening result, not a medical diagnosis, psychiatric " +
  "assessment, or treatment recommendation. If you are concerned about your " +
  "wellbeing, consider speaking with a qualified mental-health professional.";

export function Disclaimer({ text = DEFAULT_TEXT, variant = "info" }: DisclaimerProps) {
  const bg = variant === "crisis" ? "#7c3aed" : "#1e40af";
  return (
    <aside
      role="note"
      aria-label="Research disclaimer"
      style={{
        background: bg,
        color: "#fff",
        borderRadius: 8,
        padding: "12px 16px",
        marginBottom: 16,
        fontSize: 14,
        lineHeight: 1.5,
      }}
    >
      <strong>Note: </strong>{text}
    </aside>
  );
}
