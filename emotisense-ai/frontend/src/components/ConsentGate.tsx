/**
 * Explicit consent gate — user must acknowledge before any data is processed.
 * Consent is per-session per-modality and is never assumed.
 */

import { useState } from "react";

interface ConsentGateProps {
  modality: "text" | "audio_recording" | "image_upload" | "video_upload";
  onConsent: () => void;
}

const CONSENT_TEXT: Record<ConsentGateProps["modality"], string> = {
  text: "I understand that the text I enter will be analysed by a research AI model. This is not a clinical service and my text will not be stored.",
  audio_recording: "I consent to my voice being recorded and processed in memory by this research system. No audio file will be stored after analysis.",
  image_upload: "I consent to uploading an image for facial-expression research analysis. The image will not be stored after processing.",
  video_upload: "I consent to uploading a video for research analysis. The video will not be stored after processing.",
};

export function ConsentGate({ modality, onConsent }: ConsentGateProps) {
  const [checked, setChecked] = useState(false);

  return (
    <div
      style={{
        background: "#f8fafc",
        border: "1px solid #cbd5e1",
        borderRadius: 10,
        padding: 20,
        marginBottom: 24,
      }}
    >
      <h3 style={{ margin: "0 0 10px", fontSize: 16 }}>Consent Required</h3>
      <p style={{ margin: "0 0 14px", fontSize: 14, color: "#374151" }}>
        {CONSENT_TEXT[modality]}
      </p>
      <label style={{ display: "flex", alignItems: "flex-start", gap: 8, cursor: "pointer" }}>
        <input
          type="checkbox"
          aria-label="I give my consent"
          checked={checked}
          onChange={(e) => setChecked(e.target.checked)}
          style={{ marginTop: 2 }}
        />
        <span style={{ fontSize: 14 }}>
          I have read and understood the above, and I give my consent.
        </span>
      </label>
      <button
        onClick={onConsent}
        disabled={!checked}
        aria-disabled={!checked}
        style={{
          marginTop: 16,
          padding: "8px 22px",
          background: checked ? "#2563eb" : "#9ca3af",
          color: "#fff",
          border: "none",
          borderRadius: 6,
          cursor: checked ? "pointer" : "not-allowed",
          fontSize: 14,
          fontWeight: 600,
        }}
      >
        Continue
      </button>
    </div>
  );
}
