import { useCallback, useState } from "react";
import { analyzeImage } from "@/api/client";
import { useAnalysis } from "@/hooks/useAnalysis";
import { Disclaimer } from "@/components/Disclaimer";
import { EmotionChart } from "@/components/EmotionChart";
import { QualityBadge } from "@/components/QualityBadge";
import { ConsentGate } from "@/components/ConsentGate";

export default function FacialAnalysis() {
  const [consented, setConsented] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const { data, loading, error, run } = useAnalysis(analyzeImage);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    if (f) {
      const reader = new FileReader();
      reader.onload = (ev) => setPreview(ev.target?.result as string);
      reader.readAsDataURL(f);
    } else {
      setPreview(null);
    }
  }, []);

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: "32px 16px" }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 16 }}>Facial Expression Analysis</h2>
      <Disclaimer text="Facial expression analysis detects emotion signals only. It is not a depression indicator and must not be used for clinical assessment." />

      {!consented ? (
        <ConsentGate modality="image_upload" onConsent={() => setConsented(true)} />
      ) : (
        <>
          <label htmlFor="image-upload" style={{ display: "block", fontWeight: 600, marginBottom: 8 }}>
            Upload Image (JPEG, PNG, WebP)
          </label>
          <input
            id="image-upload"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleFileChange}
            style={{ marginBottom: 14, fontSize: 14 }}
          />

          {preview && (
            <div style={{ marginBottom: 14 }}>
              <img
                src={preview}
                alt="Uploaded face preview"
                style={{
                  maxWidth: 240,
                  maxHeight: 240,
                  borderRadius: 8,
                  border: "1px solid #e2e8f0",
                  objectFit: "cover",
                }}
              />
            </div>
          )}

          <button
            onClick={() => file && run(file)}
            disabled={!file || loading}
            style={{
              padding: "8px 22px",
              background: file && !loading ? "#2563eb" : "#9ca3af",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              cursor: !file || loading ? "not-allowed" : "pointer",
              fontWeight: 600,
              fontSize: 14,
            }}
          >
            {loading ? "Analysing…" : "Analyse"}
          </button>

          {error && (
            <p role="alert" style={{ color: "#ef4444", marginTop: 12, fontSize: 14 }}>
              Error: {error}
            </p>
          )}

          {data && (
            <section aria-label="Facial analysis results" style={{ marginTop: 24 }}>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 12 }}>
                <div
                  style={{
                    padding: "4px 12px",
                    borderRadius: 20,
                    background: data.face_detected ? "#dcfce7" : "#fee2e2",
                    border: `1px solid ${data.face_detected ? "#86efac" : "#fca5a5"}`,
                    fontSize: 13,
                  }}
                >
                  {data.face_detected ? `✓ Face detected (${data.n_faces})` : "✗ No face detected"}
                </div>
                <QualityBadge
                  score={data.face_quality?.quality_score}
                  label="Face quality"
                  rejectionReasons={data.face_quality?.rejection_reasons}
                />
              </div>

              {/* Grad-CAM overlay */}
              {data.gradcam_heatmap_base64 && (
                <div style={{ marginBottom: 16 }}>
                  <h4 style={{ fontSize: 14, marginBottom: 6 }}>
                    Grad-CAM Heatmap
                    <span style={{ fontSize: 12, fontWeight: 400, color: "#6b7280", marginLeft: 8 }}>
                      (emotion model — not clinical evidence)
                    </span>
                  </h4>
                  <img
                    src={`data:image/png;base64,${data.gradcam_heatmap_base64}`}
                    alt="Grad-CAM emotion model heatmap"
                    style={{ maxWidth: 240, borderRadius: 8, border: "1px solid #e2e8f0" }}
                  />
                </div>
              )}

              {data.emotion_profile && <EmotionChart profile={data.emotion_profile} />}
              {!data.model_available && (
                <p style={{ fontSize: 14, color: "#6b7280", marginTop: 12 }}>{data.note}</p>
              )}
              <Disclaimer text={data.disclaimer} />
            </section>
          )}
        </>
      )}
    </main>
  );
}
