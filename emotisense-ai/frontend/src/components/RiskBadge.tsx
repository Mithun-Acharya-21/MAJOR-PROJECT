import type { RiskTier } from "@/types/api";
import { riskTierColor, riskTierLabel } from "@/types/api";

interface RiskBadgeProps {
  tier: RiskTier;
  score?: number | null;
  uncertainty?: number | null;
}

export function RiskBadge({ tier, score, uncertainty }: RiskBadgeProps) {
  const color = riskTierColor(tier);
  const label = riskTierLabel(tier);
  return (
    <div
      role="status"
      aria-label={`Risk assessment: ${label}`}
      style={{
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 4,
        padding: "12px 24px",
        borderRadius: 12,
        background: color,
        color: "#fff",
        minWidth: 160,
      }}
    >
      <span style={{ fontSize: 13, opacity: 0.85 }}>Research Signal</span>
      <span style={{ fontSize: 20, fontWeight: 700 }}>{label}</span>
      {score !== null && score !== undefined && (
        <span style={{ fontSize: 13 }}>
          Score: {(score * 100).toFixed(1)} %
        </span>
      )}
      {uncertainty !== null && uncertainty !== undefined && (
        <span style={{ fontSize: 11, opacity: 0.8 }}>
          Uncertainty: {(uncertainty * 100).toFixed(1)} %
        </span>
      )}
      <span style={{ fontSize: 11, opacity: 0.75, marginTop: 4, textAlign: "center" }}>
        Not a clinical diagnosis
      </span>
    </div>
  );
}
