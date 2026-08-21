import { Disclaimer } from "@/components/Disclaimer";

const LIMITATIONS = [
  "Depression-risk labels from research datasets are not equivalent to clinical diagnoses.",
  "Emotion recognition is noisy and influenced by cultural and contextual factors.",
  "Performance on benchmark interview datasets may not generalise to casual recordings.",
  "A face, voice, or transcript alone cannot establish mental state.",
  "Explanations show model behaviour — they are not proof of clinical causality.",
  "The system cannot replace qualified professional assessment or emergency care.",
  "Model performance on clinical populations has not been established.",
  "Dataset biases and demographic imbalance affect all outputs.",
];

const PRIVACY_CONTROLS = [
  "No raw audio, images, or video is persisted by default.",
  "Text input is processed in memory and discarded after the request.",
  "Request IDs are pseudonymous (UUID4) — no user identity is attached.",
  "Consent is required before any biometric data is processed.",
  "Uploads are size-limited and MIME-type validated before processing.",
  "Rate limiting prevents bulk data submission.",
  "All data is transmitted over HTTPS in production.",
  "Store only minimal derived metadata if experiment tracking is enabled.",
];

export default function EthicsPrivacy() {
  return (
    <main style={{ maxWidth: 760, margin: "0 auto", padding: "32px 16px" }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 16 }}>Ethics, Limitations & Privacy</h2>
      <Disclaimer />

      <Section title="System Limitations">
        <ul style={{ paddingLeft: 20, margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
          {LIMITATIONS.map((l, i) => (
            <li key={i} style={{ fontSize: 14, color: "#374151", lineHeight: 1.6 }}>{l}</li>
          ))}
        </ul>
      </Section>

      <Section title="Privacy Controls">
        <ul style={{ paddingLeft: 20, margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
          {PRIVACY_CONTROLS.map((p, i) => (
            <li key={i} style={{ fontSize: 14, color: "#374151", lineHeight: 1.6 }}>{p}</li>
          ))}
        </ul>
      </Section>

      <Section title="Safety Policy">
        <p style={{ fontSize: 14, lineHeight: 1.7, margin: 0 }}>
          This system implements a crisis-first safety policy. Any signal matching
          self-harm or immediate-danger language triggers an immediate crisis response
          that supersedes all other suggestions. Generic, non-location-specific crisis
          resource text is displayed; jurisdiction-specific lines can be injected by
          the operator at deployment time. The system will never diagnose, prescribe,
          or recommend medication changes.
        </p>
      </Section>

      <Section title="Intended Use">
        <p style={{ fontSize: 14, lineHeight: 1.7, margin: 0 }}>
          This is a <strong>research screening tool</strong>. It is designed for use
          by researchers and developers studying multimodal early-risk indicators.
          It is not intended for use by individuals as a substitute for professional
          mental-health assessment, and must not be deployed in clinical settings
          without regulatory approval and appropriate clinical oversight.
        </p>
      </Section>

      <Section title="Dataset and Licensing">
        <p style={{ fontSize: 14, lineHeight: 1.7, margin: 0 }}>
          Multimodal depression-risk training requires an approved research agreement
          for DAIC-WOZ / E-DAIC. Emotion pretraining datasets (RAVDESS, FER-2013,
          CREMA-D) are used under their respective licences. RAVDESS uses CC BY-NC-SA
          4.0. No restricted dataset is included in this repository. All data cards
          must be completed before any dataset is preprocessed.
        </p>
      </Section>
    </main>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section
      aria-labelledby={`section-${title.replace(/\s+/g, "-")}`}
      style={{
        marginBottom: 28,
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        padding: 20,
      }}
    >
      <h3
        id={`section-${title.replace(/\s+/g, "-")}`}
        style={{ fontSize: 16, fontWeight: 700, marginBottom: 12, marginTop: 0 }}
      >
        {title}
      </h3>
      {children}
    </section>
  );
}
