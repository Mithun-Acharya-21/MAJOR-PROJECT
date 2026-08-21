/**
 * Simple horizontal bar chart for emotion probabilities.
 * Labelled "Emotion Signal" — never depression indicator.
 */

import type { EmotionProfile } from "@/types/api";

interface EmotionChartProps {
  profile: EmotionProfile;
}

const EMOTION_COLORS: Record<string, string> = {
  happy:    "#22c55e",
  neutral:  "#6b7280",
  sad:      "#3b82f6",
  angry:    "#ef4444",
  fear:     "#8b5cf6",
  disgust:  "#f59e0b",
  surprise: "#06b6d4",
};

export function EmotionChart({ profile }: EmotionChartProps) {
  const entries = Object.entries(profile.probabilities).sort(
    ([, a], [, b]) => b - a
  );

  return (
    <section aria-label="Emotion signal chart">
      <h3 style={{ fontSize: 15, marginBottom: 10, fontWeight: 600 }}>
        Emotion Signals
        <span style={{ fontSize: 12, fontWeight: 400, color: "#6b7280", marginLeft: 8 }}>
          (emotion model — not a depression indicator)
        </span>
      </h3>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {entries.map(([emotion, prob]) => (
          <div key={emotion} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span
              style={{
                width: 80,
                fontSize: 13,
                textTransform: "capitalize",
                fontWeight: profile.predicted_emotion === emotion ? 700 : 400,
              }}
            >
              {emotion}
            </span>
            <div
              style={{
                flex: 1,
                background: "#f1f5f9",
                borderRadius: 4,
                height: 14,
                overflow: "hidden",
              }}
              role="progressbar"
              aria-valuenow={Math.round(prob * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`${emotion}: ${Math.round(prob * 100)}%`}
            >
              <div
                style={{
                  width: `${(prob * 100).toFixed(1)}%`,
                  height: "100%",
                  background: EMOTION_COLORS[emotion] ?? "#6b7280",
                  transition: "width 0.4s ease",
                }}
              />
            </div>
            <span style={{ width: 42, fontSize: 12, color: "#374151", textAlign: "right" }}>
              {(prob * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
