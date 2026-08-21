/**
 * Small quality-score indicator used next to each modality result.
 */

interface QualityBadgeProps {
  score: number | null | undefined;
  label?: string;
  rejectionReasons?: string[];
}

export function QualityBadge({ score, label, rejectionReasons }: QualityBadgeProps) {
  if (score === null || score === undefined) return null;

  const pct = Math.round(score * 100);
  const color = pct >= 70 ? "#22c55e" : pct >= 40 ? "#f59e0b" : "#ef4444";

  return (
    <div
      aria-label={`${label ?? "Quality"}: ${pct}%`}
      title={rejectionReasons?.length ? `Issues: ${rejectionReasons.join(", ")}` : undefined}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "3px 10px",
        borderRadius: 20,
        background: `${color}22`,
        border: `1px solid ${color}`,
        fontSize: 12,
        color,
        fontWeight: 600,
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: color,
          flexShrink: 0,
        }}
      />
      {label && <span style={{ color: "#374151", fontWeight: 400 }}>{label}: </span>}
      {pct}%
    </div>
  );
}
