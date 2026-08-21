import { Disclaimer } from "@/components/Disclaimer";

const SECTIONS = [
  {
    title: "Text — LIME Token Attribution",
    description:
      "LIME perturbs the input text and measures how each word's presence affects the risk score. Higher absolute values indicate stronger influence on the model output.",
    caveat: "LIME shows model behaviour, not clinical causality. These are not diagnostic markers.",
  },
  {
    title: "Text — SHAP Token Attribution",
    description:
      "SHAP (SHapley Additive exPlanations) fairly distributes the prediction score across tokens based on their marginal contributions across all possible token subsets.",
    caveat: "SHAP values explain the model, not the user's mental state.",
  },
  {
    title: "Text — Attention Weights",
    description:
      "DeBERTa attention weights show which token positions the transformer attended to. Displayed as a supplementary diagnostic view only.",
    caveat: "Attention weights are a supplementary signal, not a causal explanation.",
  },
  {
    title: "Voice — Acoustic Feature Attribution",
    description:
      "Permutation importance measures how much accuracy drops when each MFCC coefficient row is randomly shuffled. Higher drop = more influential feature.",
    caveat: "Acoustic attributions are model-observed patterns, not clinical evidence.",
  },
  {
    title: "Face — Grad-CAM Heatmap",
    description:
      "Gradient-weighted Class Activation Mapping highlights which spatial regions of the face image most influenced the emotion prediction.",
    caveat: "Grad-CAM is an emotion-model visualisation, not a depression-risk indicator.",
  },
  {
    title: "Fusion — Gate Contributions",
    description:
      "Learned modality gates weight each branch by quality and contextual reliability. The contribution percentage shows relative gate values, not causal importance.",
    caveat: "Gate weights are diagnostic signals, not causal proof of which modality matters clinically.",
  },
  {
    title: "Fusion — Cross-Modal Attention",
    description:
      "The cross-modal attention matrix shows inter-modality alignment — how much each modality 'attended' to every other modality during fusion.",
    caveat: "Attention alignment is an architectural signal, not evidence of clinical co-occurrence.",
  },
];

export default function Explainability() {
  return (
    <main style={{ maxWidth: 800, margin: "0 auto", padding: "32px 16px" }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 8 }}>Explainability Views</h2>
      <p style={{ color: "#6b7280", marginBottom: 16, fontSize: 14 }}>
        All explanations show how the model processed inputs. None prove clinical causality.
      </p>
      <Disclaimer text="These are model-derived explanations for a research screening system. They must not be interpreted as clinical evidence of any mental-health condition." />

      <div style={{ display: "flex", flexDirection: "column", gap: 16, marginTop: 20 }}>
        {SECTIONS.map(({ title, description, caveat }) => (
          <div
            key={title}
            style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 10,
              padding: 18,
            }}
          >
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>{title}</h3>
            <p style={{ fontSize: 14, color: "#374151", marginBottom: 8, lineHeight: 1.6 }}>
              {description}
            </p>
            <p style={{ fontSize: 13, color: "#6b7280", fontStyle: "italic", margin: 0 }}>
              ⚠ {caveat}
            </p>
          </div>
        ))}
      </div>

      <div
        style={{
          marginTop: 28,
          padding: 16,
          background: "#fffbeb",
          border: "1px solid #fcd34d",
          borderRadius: 8,
          fontSize: 14,
          color: "#374151",
        }}
      >
        <strong>Performance note:</strong> LIME and SHAP are computed on demand and
        may take 10–30 seconds. Grad-CAM and fusion explanations are fast (single
        forward pass). All explanations are cached per request.
      </div>
    </main>
  );
}
