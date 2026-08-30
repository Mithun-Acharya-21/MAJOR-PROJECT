import React, { useState, useEffect } from "react";
import {
  FileText,
  Camera,
  Video,
  Mic,
  BarChart3,
  ChevronRight,
  ChevronLeft,
  RotateCcw,
  AlertTriangle,
  CheckCircle2,
  Info,
  Shield,
  Sparkles,
  Brain,
  Eye,
  Activity,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  analyzeText,
  analyzePhoto,
  analyzeVoice,
  analyzeMultimodal,
  startSession,
  fetchBaselines,
  type TextResult,
  type PhotoResult,
  type VoiceResult,
  type MultimodalResult,
  type ModalityResult,
  type BaselinesResult,
  type ModelMetric,
} from "@/lib/api-client";
import { WebcamCapture } from "@/components/WebcamCapture";
import { AudioRecorder } from "@/components/AudioRecorder";

const STEPS = [
  { id: "text", label: "Text Analysis", icon: FileText },
  { id: "photo", label: "Photo Analysis", icon: Camera },
  { id: "face", label: "Live Face", icon: Video },
  { id: "voice", label: "Voice/Speech", icon: Mic },
  { id: "result", label: "Final Result", icon: BarChart3 },
] as const;

type StepId = (typeof STEPS)[number]["id"];

const SAMPLE_TEXTS = [
  "I have been feeling tired lately. I don't enjoy activities as much as before and sometimes find it difficult to concentrate.",
  "Had a great day today! Went for a walk, laughed with friends, and feeling optimistic about the week ahead.",
];

function riskColor(level: string): string {
  if (level === "high") return "var(--risk)";
  if (level === "moderate") return "var(--attention)";
  return "var(--calm)";
}

function riskLabel(level: string): string {
  if (level === "high") return "Higher Depression Risk";
  if (level === "moderate") return "Moderate Risk Indicators";
  return "Lower Depression Risk";
}

// ────────────────────────────────────────────────
// Modality Result Card
// ────────────────────────────────────────────────
function ModalityCard({
  name,
  result,
  icon: Icon,
}: {
  name: string;
  result: ModalityResult;
  icon: React.ComponentType<{ className?: string; style?: React.CSSProperties }>;
}) {
  const color = riskColor(result.risk_level);
  return (
    <div className="rounded-xl border border-border bg-background/40 p-4">
      <div className="flex items-center gap-2">
        <Icon className="size-4" style={{ color }} />
        <span className="font-semibold capitalize">{name}</span>
        <Badge variant="outline" style={{ color, borderColor: color }} className="ml-auto">
          {riskLabel(result.risk_level)}
        </Badge>
      </div>
      <div className="mt-3 flex items-end gap-3">
        <span className="text-2xl font-bold font-mono" style={{ color }}>
          {result.probability.toFixed(1)}%
        </span>
        <span className="text-xs text-muted-foreground mb-1">
          confidence: {result.confidence.toFixed(0)}%
        </span>
      </div>
      <Progress value={result.probability} className="mt-2 h-1.5" />
      {result.error && (
        <p className="mt-2 text-xs text-destructive">{result.error}</p>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────
// Modality Contribution Bars
// ────────────────────────────────────────────────
const FusionViz = (props: { result: MultimodalResult }) => {
  const contributions = props.result.explanation?.contributions;
  if (!contributions || contributions.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        Individual modality contribution data is unavailable.
      </p>
    );
  }

  const maxContrib = Math.max(...contributions.map((c) => c.weight * 100), 1);

  return (
    <div className="space-y-3">
      <div className="space-y-2">
        {contributions.map((c) => {
          const barWidth = (c.weight * 100) / maxContrib;
          const color = c.probability >= 55 ? "var(--risk)" : c.probability >= 45 ? "var(--attention)" : "var(--calm)";
          return (
            <div key={c.modality} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="capitalize font-medium">{c.modality}</span>
                <span className="text-muted-foreground font-mono text-xs">
                  {c.probability.toFixed(1)}% — weight: {(c.weight * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-2 w-full rounded-full bg-muted/30 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${barWidth * 100}%`, backgroundColor: color }}
                />
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-muted-foreground">
        The final result is based on the combined model outputs from all available modalities
        using weighted probability fusion.
        {props.result.explanation?.agreement_score != null && (
          <> Agreement score: {(props.result.explanation.agreement_score * 100).toFixed(0)}%.</>
        )}
      </p>
    </div>
  );
};

// ────────────────────────────────────────────────
// LIME / Explainability Section
// ────────────────────────────────────────────────
function LimeExplainability({
  textResult,
  photoResult,
  voiceResult,
  finalResult,
}: {
  textResult: TextResult | null;
  photoResult: PhotoResult | null;
  voiceResult: VoiceResult | null;
  finalResult: MultimodalResult | null;
}) {
  // Determine which tabs to show based on available data
  const hasText = !!textResult?.explanation;
  const hasPhoto = !!photoResult?.explanation?.gradcam;
  const hasVoice = !!voiceResult?.explanation;
  const hasOverall = !!finalResult?.explanation?.contributions;

  const defaultTab = hasOverall ? "overall" : hasText ? "text" : hasPhoto ? "photo" : "voice";

  return (
    <div>
      <Tabs defaultValue={defaultTab} className="w-full">
        <TabsList className="w-full justify-start">
          {hasOverall && <TabsTrigger value="overall">Overall</TabsTrigger>}
          {hasText && <TabsTrigger value="text">Text</TabsTrigger>}
          {hasPhoto && <TabsTrigger value="photo">Photo</TabsTrigger>}
          {hasVoice && <TabsTrigger value="voice">Voice</TabsTrigger>}
        </TabsList>

        {/* Overall Tab */}
        {hasOverall && (
          <TabsContent value="overall">                <FusionViz result={finalResult!} />
          </TabsContent>
        )}

        {/* Text Tab */}
        {hasText && (
          <TabsContent value="text">
            <LimeTextExplanation result={textResult!} />
          </TabsContent>
        )}

        {/* Photo Tab */}
        {hasPhoto && (
          <TabsContent value="photo">
            <LimePhotoExplanation result={photoResult!} />
          </TabsContent>
        )}

        {/* Voice Tab */}
        {hasVoice && (
          <TabsContent value="voice">
            <LimeVoiceExplanation result={voiceResult!} />
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
}

function LimeTextExplanation({ result }: { result: TextResult }) {
  const explanation = result.explanation;

  // Handle LIME array format
  if (Array.isArray(explanation)) {
    return (
      <div className="space-y-3">
        {explanation.map((f: { feature: string; weight: number; direction: string; contribution?: number }, i: number) => (
          <div key={i} className="flex items-center gap-3">
            <span className="font-mono text-sm min-w-[140px]">"{f.feature}"</span>
            <div className="flex-1 h-2 rounded-full bg-muted/30 overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${Math.min(f.weight * 100, 100)}%`,
                  backgroundColor: f.direction === "risk_increasing" ? "var(--risk)" : "var(--calm)",
                }}
              />
            </div>
            <Badge
              variant="outline"
              style={{
                color: f.direction === "risk_increasing" ? "var(--risk)" : "var(--calm)",
                borderColor: f.direction === "risk_increasing" ? "var(--risk)" : "var(--calm)",
              }}
            >
              {f.direction === "risk_increasing" ? "↑ risk" : "↓ protective"}
            </Badge>
          </div>
        ))}
      </div>
    );
  }

  // Handle attention_tokens format
  if (explanation && typeof explanation === "object" && "attention_tokens" in explanation) {
    const tokens = (explanation as { attention_tokens: Array<{ token: string; weight: number; color?: string }> }).attention_tokens;
    if (tokens && tokens.length > 0) {
      return (
        <div className="flex flex-wrap gap-1">
          {tokens.map((t, i) => (
            <span
              key={i}
              className="rounded px-1.5 py-0.5 font-mono text-sm"
              style={{
                backgroundColor: t.color ?? undefined,
                color: "var(--foreground)",
                opacity: 0.5 + t.weight * 0.5,
              }}
            >
              {t.token}
            </span>
          ))}
        </div>
      );
    }
  }

  // Handle top_features format
  if (explanation && typeof explanation === "object" && "top_features" in explanation) {
    const features = (explanation as { top_features: Array<{ feature: string; weight: number; direction: string }> }).top_features;
    if (features && features.length > 0) {
      return (
        <div className="space-y-2">
          {features.map((f, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="font-mono text-sm min-w-[140px]">"{f.feature}"</span>
              <div className="flex-1 h-2 rounded-full bg-muted/30 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${Math.min(f.weight * 100, 100)}%`,
                    backgroundColor: f.direction === "risk_increasing" ? "var(--risk)" : "var(--calm)",
                  }}
                />
              </div>
              <span className="text-xs text-muted-foreground font-mono">
                {f.weight.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      );
    }
  }

  return (
    <p className="text-sm text-muted-foreground">
      Text explanation is not available for this analysis.
    </p>
  );
}

function LimePhotoExplanation({ result }: { result: PhotoResult }) {
  const explanation = result.explanation;
  if (!explanation) {
    return (
      <p className="text-sm text-muted-foreground">
        Image explanation is not available for this analysis.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        {explanation.gradcam && (
          <div className="rounded-lg border border-border bg-background/30 p-2">
            <p className="mb-1 text-[10px] text-muted-foreground">Grad-CAM Overlay</p>
            <img src={explanation.gradcam} alt="Grad-CAM" className="w-full rounded" />
          </div>
        )}
        {explanation.heatmap && (
          <div className="rounded-lg border border-border bg-background/30 p-2">
            <p className="mb-1 text-[10px] text-muted-foreground">Heatmap</p>
            <img src={explanation.heatmap} alt="Heatmap" className="w-full rounded" />
          </div>
        )}
      </div>
      {!explanation.gradcam && !explanation.heatmap && (
        <p className="text-sm text-muted-foreground">
          Image explanation is not available for this analysis.
        </p>
      )}
    </div>
  );
}

function LimeVoiceExplanation({ result }: { result: VoiceResult }) {
  const explanation = result.explanation;
  if (!explanation || typeof explanation !== "object") {
    return (
      <p className="text-sm text-muted-foreground">
        Voice explanation is not available for this analysis.
      </p>
    );
  }

  const exp = explanation as Record<string, unknown>;
  const topFeatures = (exp as Record<string, unknown>)['top_features'] as Array<{ feature: string; weight: number; direction: string }> | undefined;
  const method = (exp as Record<string, unknown>)['method'] as string | undefined;

  return (
    <div className="space-y-3">
      {method && (
        <p className="text-xs text-muted-foreground">
          Analysis method: <span className="font-mono">{method}</span>
        </p>
      )}
      {topFeatures && topFeatures.length > 0 ? (
        <div className="space-y-2">
          {topFeatures.map((f, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="font-mono text-sm min-w-[140px]">{f.feature}</span>
              <div className="flex-1 h-2 rounded-full bg-muted/30 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${Math.min(f.weight * 100, 100)}%`,
                    backgroundColor: f.direction === "risk_increasing" ? "var(--risk)" : "var(--calm)",
                  }}
                />
              </div>
              <Badge
                variant="outline"
                style={{
                  color: f.direction === "risk_increasing" ? "var(--risk)" : "var(--calm)",
                  borderColor: f.direction === "risk_increasing" ? "var(--risk)" : "var(--calm)",
                }}
              >
                {f.direction === "risk_increasing" ? "↑" : "↓"}
              </Badge>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">
          No feature-level explanation available for this analysis.
        </p>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────
// Model Performance Section
// ────────────────────────────────────────────────
function ModelPerformance({ baselines }: { baselines: BaselinesResult | null }) {
  const [expanded, setExpanded] = useState(false);

  if (!baselines) {
    return (
      <p className="text-sm text-muted-foreground">Loading model performance data...</p>
    );
  }

  // Find the "ours" model (multimodal fusion) and other models
  const oursModel = baselines.models.find((m) => m.type === "multimodal");
  const otherModels = baselines.models.filter((m) => m.type !== "multimodal");

  return (
    <div className="space-y-4">
      {/* Our model highlight */}
      {oursModel && (
        <div className="rounded-xl border border-primary/30 bg-primary/5 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="size-4 text-primary" />
            <span className="font-semibold text-sm">{oursModel.name}</span>
            <Badge variant="outline" className="ml-auto text-[10px]">Our Model</Badge>
          </div>
          <PerformanceBars metrics={oursModel} />
        </div>
      )}

      {/* Baseline comparison */}
      <div className="space-y-2">
        {otherModels.map((model) => (
          <div key={model.name} className="rounded-lg border border-border bg-background/40 p-3">
            <p className="text-xs font-medium text-muted-foreground">{model.name}</p>
            <div className="mt-2 grid grid-cols-5 gap-2">
              {(["accuracy", "precision", "recall", "f1", "auc"] as const).map((metric) => (
                <div key={metric} className="text-center">
                  <p className="font-mono text-sm font-bold" style={{ color: metric === "accuracy" ? "var(--calm)" : undefined }}>
                    {metric === "auc" ? model[metric].toFixed(2) : `${(model[metric] * 100).toFixed(0)}%`}
                  </p>
                  <p className="text-[10px] text-muted-foreground capitalize">{metric === "auc" ? "ROC-AUC" : metric}</p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Technical details */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        {expanded ? "▾ Hide" : "▸ Show"} Technical Details
      </button>
      {expanded && baselines.description && (
        <div className="rounded-lg bg-secondary/20 p-3 text-xs text-muted-foreground">
          <p>{baselines.description}</p>
          {baselines.notes && <p className="mt-2 italic">{baselines.notes}</p>}
          {baselines.datasets && (
            <div className="mt-2">
              <p className="font-medium">Datasets:</p>
              <ul className="mt-1 list-disc pl-4">
                {Object.entries(baselines.datasets).map(([type, names]) => (
                  <li key={type}>
                    <span className="capitalize">{type}:</span> {names.join(", ")}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function PerformanceBars({ metrics }: { metrics: ModelMetric }) {
  const items = [
    { label: "Accuracy", value: metrics.accuracy, color: "var(--calm)" },
    { label: "Precision", value: metrics.precision, color: "var(--text-branch)" },
    { label: "Recall", value: metrics.recall, color: "var(--attention)" },
    { label: "F1 Score", value: metrics.f1, color: "var(--fusion-branch)" },
    { label: "ROC-AUC", value: metrics.auc, color: "var(--face-branch)" },
  ];

  return (
    <div className="space-y-2">
      {items.map((item) => (
        <div key={item.label} className="space-y-0.5">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">{item.label}</span>
            <span className="font-mono font-bold" style={{ color: item.color }}>
              {item.label === "ROC-AUC" ? item.value.toFixed(2) : `${(item.value * 100).toFixed(1)}%`}
            </span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-muted/30 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{ width: `${item.value * 100}%`, backgroundColor: item.color }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// ────────────────────────────────────────────────
// Recommendations Section
// ────────────────────────────────────────────────
function RecommendationsSection({ recommendations }: { recommendations: Record<string, unknown> }) {
  const rec = recommendations as {
    headline?: string;
    safety_concern?: boolean;
    what_you_can_do?: {
      today?: string[];
      this_week?: string[];
      when_to_seek_help?: string[];
    };
    explainable_signals?: Array<{ signal: string; description: string; recommendation: string }>;
  };

  return (
    <div className="space-y-3">
      {rec.safety_concern && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          <AlertTriangle className="mr-1 inline size-4" />
          If you are in immediate danger, please contact your local emergency services.
        </div>
      )}
      {rec.headline && (
        <p className="text-sm font-medium">{rec.headline}</p>
      )}
      {rec.explainable_signals && rec.explainable_signals.length > 0 && (
        <div className="space-y-2">
          {rec.explainable_signals.map((s, i) => (
            <div key={i} className="rounded bg-secondary/20 p-2 text-xs">
              <span className="font-medium">{s.signal}:</span>{" "}
              <span className="text-muted-foreground">{s.recommendation}</span>
            </div>
          ))}
        </div>
      )}
      {rec.what_you_can_do && (
        <div className="space-y-3 text-xs">
          {rec.what_you_can_do.today && rec.what_you_can_do.today.length > 0 && (
            <div>
              <p className="font-medium text-foreground">Today:</p>
              <ul className="mt-1 space-y-1 pl-4">
                {rec.what_you_can_do.today.map((item, i) => (
                  <li key={i} className="flex items-start gap-1">
                    <CheckCircle2 className="mt-0.5 size-3 shrink-0 text-primary" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {rec.what_you_can_do.when_to_seek_help && rec.what_you_can_do.when_to_seek_help.length > 0 && (
            <div>
              <p className="font-medium text-foreground">When to seek help:</p>
              <ul className="mt-1 space-y-1 pl-4">
                {rec.what_you_can_do.when_to_seek_help.map((item, i) => (
                  <li key={i} className="flex items-start gap-1">
                    <CheckCircle2 className="mt-0.5 size-3 shrink-0 text-amber-500" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────
// Main ScreeningWizard
// ────────────────────────────────────────────────
export function ScreeningWizard() {
  const [currentStep, setCurrentStep] = useState<StepId>("text");
  const [sessionId, setSessionId] = useState<string | null>(null);

  // Results per modality
  const [textResult, setTextResult] = useState<TextResult | null>(null);
  const [photoResult, setPhotoResult] = useState<PhotoResult | null>(null);
  const [faceResult, setFaceResult] = useState<ModalityResult | null>(null);
  const [voiceResult, setVoiceResult] = useState<VoiceResult | null>(null);
  const [finalResult, setFinalResult] = useState<MultimodalResult | null>(null);

  // Model performance data
  const [baselines, setBaselines] = useState<BaselinesResult | null>(null);

  // Loading states
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Input state
  const [textInput, setTextInput] = useState("");
  const [photoFile, setPhotoFile] = useState<File | null>(null);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  const stepIndex = STEPS.findIndex((s) => s.id === currentStep);

  // Fetch baselines when reaching result step
  useEffect(() => {
    if (currentStep === "result" && !baselines) {
      fetchBaselines()
        .then(setBaselines)
        .catch(() => setBaselines(null));
    }
  }, [currentStep, baselines]);

  const goToStep = (step: StepId) => {
    setError(null);
    setCurrentStep(step);
  };

  const canGoNext = () => {
    if (currentStep === "text") return !!textResult;
    if (currentStep === "photo") return !!photoResult;
    if (currentStep === "face") return !!faceResult;
    if (currentStep === "voice") return !!voiceResult;
    return false;
  };

  const goNext = () => {
    const nextIndex = stepIndex + 1;
    if (nextIndex < STEPS.length) {
      goToStep(STEPS[nextIndex]!.id);
    }
  };

  const goPrev = () => {
    const prevIndex = stepIndex - 1;
    if (prevIndex >= 0) {
      goToStep(STEPS[prevIndex]!.id);
    }
  };

  // --- Analysis handlers ---

  const handleAnalyzeText = async () => {
    if (!textInput.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const result = await analyzeText(textInput.trim());
      setTextResult(result);
      // Initialize session
      const session = await startSession();
      setSessionId(session.session_id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Text analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzePhoto = async () => {
    if (!photoFile) return;
    setLoading(true);
    setError(null);
    try {
      const result = await analyzePhoto(photoFile);
      setPhotoResult(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Photo analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeFace = async (frames: Blob[]) => {
    if (frames.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      frames.forEach((frame, i) => {
        formData.append("frames", frame, `frame_${i}.jpg`);
      });
      const res = await fetch("http://localhost:8000/api/analyze/face", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Face analysis failed");
      const result = await res.json();
      setFaceResult(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Face analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeVoice = async () => {
    if (!audioBlob) return;
    setLoading(true);
    setError(null);
    try {
      const result = await analyzeVoice(audioBlob, sessionId ?? undefined);
      setVoiceResult(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Voice analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const handleFinalFusion = async () => {
    setLoading(true);
    setError(null);
    try {
      const params: { text?: string; photo?: File; audio?: Blob; session_id?: string } = {};
      const trimmed = textInput.trim();
      if (trimmed) params.text = trimmed;
      if (photoFile) params.photo = photoFile;
      if (audioBlob) params.audio = audioBlob;
      if (sessionId) params.session_id = sessionId;
      const result = await analyzeMultimodal(params);
      setFinalResult(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Fusion analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setTextResult(null);
    setPhotoResult(null);
    setFaceResult(null);
    setVoiceResult(null);
    setFinalResult(null);
    setBaselines(null);
    setTextInput("");
    setPhotoFile(null);
    setPhotoPreview(null);
    setAudioBlob(null);
    setError(null);
    setSessionId(null);
    setCurrentStep("text");
  };

  return (
    <div className="mx-auto max-w-4xl">
      {/* Step Progress */}
      <div className="mb-8 flex items-center justify-between">
        {STEPS.map((step, i) => {
          const Icon = step.icon;
          const isActive = step.id === currentStep;
          const isCompleted =
            (step.id === "text" && !!textResult) ||
            (step.id === "photo" && !!photoResult) ||
            (step.id === "face" && !!faceResult) ||
            (step.id === "voice" && !!voiceResult) ||
            (step.id === "result" && !!finalResult);
          return (
            <div key={step.id} className="flex items-center">
              <button
                onClick={() => goToStep(step.id)}
                className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors ${
                  isActive
                    ? "bg-primary/10 text-primary font-medium"
                    : isCompleted
                      ? "text-green-500"
                      : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {isCompleted ? (
                  <CheckCircle2 className="size-4" />
                ) : (
                  <Icon className="size-4" />
                )}
                <span className="hidden sm:inline">{step.label}</span>
              </button>
              {i < STEPS.length - 1 && (
                <ChevronRight className="mx-1 size-3 text-muted-foreground" />
              )}
            </div>
          );
        })}
      </div>

      {/* Step Content */}
      <div className="panel p-6">
        {error && (
          <div className="mb-4 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            <AlertTriangle className="mr-1 inline size-4" />
            {error}
          </div>
        )}

        {/* STEP 1: Text */}
        {currentStep === "text" && (
          <div>
            <h2 className="text-xl font-semibold">Step 1: Text Analysis</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Describe your thoughts, feelings, daily activities, mood, sleep, motivation, and concentration.
            </p>
            <Textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              rows={6}
              className="mt-4 resize-none bg-input/40 font-mono text-sm"
              placeholder="I have been feeling tired lately. I don't enjoy activities as much as before..."
            />
            <div className="mt-1 text-right text-xs text-muted-foreground">
              {textInput.length} characters
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              {SAMPLE_TEXTS.map((s, i) => (
                <Button key={i} variant="secondary" size="sm" onClick={() => setTextInput(s)}>
                  Sample {i + 1}
                </Button>
              ))}
            </div>
            <Button
              className="mt-4"
              onClick={handleAnalyzeText}
              disabled={loading || !textInput.trim()}
            >
              {loading ? "Analyzing..." : "Analyze Text"}
            </Button>
            {textResult && (
              <div className="mt-4">
                <ModalityCard name="Text" result={textResult} icon={FileText} />
                <div className="mt-3 rounded-lg bg-secondary/20 p-3 text-xs">
                  <p className="font-medium">Explanation:</p>
                  {Array.isArray(textResult.explanation) && textResult.explanation.length > 0 ? (
                    <div className="mt-2 space-y-1">
                      {textResult.explanation.slice(0, 8).map((f, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <span
                            className="size-2 rounded-full"
                            style={{
                              background:
                                f.direction === "risk_increasing"
                                  ? "var(--risk)"
                                  : "var(--calm)",
                            }}
                          />
                          <span className="font-mono">{f.feature}</span>
                          <span className="text-muted-foreground">
                            {f.direction === "risk_increasing" ? "+" : "-"}{f.weight.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-1 text-muted-foreground">Token-level explanation available.</p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* STEP 2: Photo */}
        {currentStep === "photo" && (
          <div>
            <h2 className="text-xl font-semibold">Step 2: Photo Analysis</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Upload a facial photograph for expression analysis.
            </p>
            <div className="mt-4">
              <input
                type="file"
                accept="image/*"
                className="hidden"
                id="photo-upload"
                onChange={(e) => {
                  const file = e.target.files?.[0] ?? null;
                  setPhotoFile(file);
                  if (file) {
                    const reader = new FileReader();
                    reader.onload = () => setPhotoPreview(reader.result as string);
                    reader.readAsDataURL(file);
                  } else {
                    setPhotoPreview(null);
                  }
                }}
              />
              <label
                htmlFor="photo-upload"
                className="flex cursor-pointer items-center justify-center rounded-lg border-2 border-dashed border-border p-8 text-center text-muted-foreground hover:border-primary/50"
              >
                {photoPreview ? (
                  <img
                    src={photoPreview}
                    alt="Selected face"
                    className="h-40 rounded-lg object-cover"
                  />
                ) : (
                  <div>
                    <Camera className="mx-auto size-8" />
                    <p className="mt-2 text-sm">Click to upload a facial photo</p>
                  </div>
                )}
              </label>
            </div>
            <Button
              className="mt-4"
              onClick={handleAnalyzePhoto}
              disabled={loading || !photoFile}
            >
              {loading ? "Analyzing..." : "Analyze Photo"}
            </Button>
            {photoResult && (
              <div className="mt-4">
                <ModalityCard name="Photo" result={photoResult} icon={Camera} />
                {photoResult.explanation?.gradcam && (
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <div className="rounded-lg border border-border bg-background/30 p-2">
                      <p className="mb-1 text-[10px] text-muted-foreground">Grad-CAM Overlay</p>
                      <img src={photoResult.explanation.gradcam} alt="Grad-CAM" className="w-full rounded" />
                    </div>
                    {photoResult.explanation.heatmap && (
                      <div className="rounded-lg border border-border bg-background/30 p-2">
                        <p className="mb-1 text-[10px] text-muted-foreground">Heatmap</p>
                        <img src={photoResult.explanation.heatmap} alt="Heatmap" className="w-full rounded" />
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* STEP 3: Live Face */}
        {currentStep === "face" && (
          <div>
            <h2 className="text-xl font-semibold">Step 3: Live Face Analysis</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Record a short video clip for real-time facial expression analysis.
            </p>
            <WebcamCapture
              onCapture={handleAnalyzeFace}
              loading={loading}
            />
            {faceResult && (
              <div className="mt-4">
                <ModalityCard name="Live Face" result={faceResult} icon={Video} />
                {(() => {
                  const dist = faceResult.features?.['expression_distribution'] as Record<string, number> | undefined;
                  if (!dist) return null;
                  return (
                    <div className="mt-3 rounded-lg bg-secondary/20 p-3 text-xs">
                      <p className="font-medium mb-2">Expression Distribution:</p>
                      <div className="grid grid-cols-3 gap-2">
                        {Object.entries(dist).map(([expr, pct]) => (
                          <div key={expr} className="text-center">
                            <span className="text-muted-foreground capitalize">{expr}</span>
                            <p className="font-mono font-bold">{pct}%</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        )}

        {/* STEP 4: Voice */}
        {currentStep === "voice" && (
          <div>
            <h2 className="text-xl font-semibold">Step 4: Voice/Speech Analysis</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Record yourself speaking for 10-30 seconds about how you've been feeling.
            </p>
            <AudioRecorder
              onRecord={(blob) => setAudioBlob(blob)}
              onAnalyze={handleAnalyzeVoice}
              loading={loading}
              hasRecording={!!audioBlob}
            />
            {voiceResult && (
              <div className="mt-4">
                <ModalityCard name="Voice" result={voiceResult} icon={Mic} />
                {voiceResult.transcript && (
                  <div className="mt-3 rounded-lg border border-border bg-background/30 p-3">
                    <p className="text-xs font-medium text-muted-foreground">Transcript:</p>
                    <p className="mt-1 text-sm text-foreground/80">{voiceResult.transcript}</p>
                  </div>
                )}
                {voiceResult.voice_features && (
                  <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                    {Object.entries(voiceResult.voice_features).map(([k, v]) => (
                      <div key={k} className="rounded bg-secondary/30 p-2">
                        <span className="text-muted-foreground">{k.replace(/_/g, " ")}: </span>
                        <span className="font-mono">{String(v)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* STEP 5: Final Result — COMPLETE */}
        {currentStep === "result" && (
          <div>
            <h2 className="text-xl font-semibold">Step 5: Multimodal Result</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Combined analysis from all available modalities.
            </p>

            {!finalResult ? (
              <div className="mt-6 space-y-4">
                <p className="text-sm text-muted-foreground">
                  Available modalities:
                </p>
                <div className="flex flex-wrap gap-2">
                  {textResult && <Badge variant="secondary">Text</Badge>}
                  {photoResult && <Badge variant="secondary">Photo</Badge>}
                  {faceResult && <Badge variant="secondary">Live Face</Badge>}
                  {voiceResult && <Badge variant="secondary">Voice</Badge>}
                </div>
                <Button
                  onClick={handleFinalFusion}
                  disabled={loading}
                  className="w-full"
                >
                  {loading ? "Combining multimodal results..." : "Generate Final Multimodal Result"}
                </Button>
              </div>
            ) : (
              <div className="mt-4 space-y-6">
                {/* ─── Overall Result Card ─── */}
                <div
                  className="rounded-xl border-2 p-6 text-center"
                  style={{ borderColor: riskColor(finalResult.risk_level) }}
                >
                  <Shield className="mx-auto size-8" style={{ color: riskColor(finalResult.risk_level) }} />
                  <h3
                    className="mt-2 text-2xl font-bold"
                    style={{ color: riskColor(finalResult.risk_level) }}
                  >
                    {riskLabel(finalResult.risk_level)}
                  </h3>
                  <div className="mt-3 flex justify-center gap-8">
                    <div>
                      <p className="text-xs text-muted-foreground">Risk Probability</p>
                      <p className="text-3xl font-bold font-mono" style={{ color: riskColor(finalResult.risk_level) }}>
                        {finalResult.probability.toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Confidence</p>
                      <p className="text-3xl font-bold font-mono">
                        {finalResult.confidence.toFixed(0)}%
                      </p>
                    </div>
                  </div>
                  <Progress value={finalResult.probability} className="mx-auto mt-4 h-3 max-w-md" />
                </div>

                {/* ─── Individual Modality Cards ─── */}
                <div>
                  <h4 className="text-sm font-semibold text-muted-foreground mb-3">Individual Analysis</h4>
                  <div className="grid gap-3 sm:grid-cols-2">
                    {finalResult.individual_results['text'] != null ? (
                      <ModalityCard name="Text" result={finalResult.individual_results['text']!} icon={FileText} />
                    ) : null}
                    {finalResult.individual_results['photo'] != null ? (
                      <ModalityCard name="Photo" result={finalResult.individual_results['photo']!} icon={Camera} />
                    ) : null}
                    {finalResult.individual_results['face'] != null ? (
                      <ModalityCard name="Live Face" result={finalResult.individual_results['face']!} icon={Video} />
                    ) : null}
                    {finalResult.individual_results['voice'] != null ? (
                      <ModalityCard name="Voice" result={finalResult.individual_results['voice']!} icon={Mic} />
                    ) : null}
                  </div>
                </div>

                {/* ─── Multimodal Fusion Visualization ─── */}
                <FusionViz result={finalResult!} />

                {/* ─── LIME / Explainability Section ─── */}
                <div>
                  <h4 className="text-sm font-semibold text-muted-foreground mb-3 flex items-center gap-2">
                    <Brain className="size-4" />
                    Why did the model reach this result?
                  </h4>
                  <LimeExplainability
                    textResult={textResult}
                    photoResult={photoResult}
                    voiceResult={voiceResult}
                    finalResult={finalResult}
                  />
                  <p className="mt-3 text-xs text-muted-foreground italic">
                    LIME shows which model features influenced the prediction. These features represent
                    model behavior and should not be interpreted as medical causes or proof of depression.
                  </p>
                </div>

                {/* ─── Model Performance ─── */}
                <div>
                  <h4 className="text-sm font-semibold text-muted-foreground mb-3 flex items-center gap-2">
                    <Activity className="size-4" />
                    Model Performance
                  </h4>
                  <ModelPerformance baselines={baselines} />
                </div>

                {/* ─── Suggestions / Recommendations ─── */}
                {finalResult.recommendations != null && typeof finalResult.recommendations === 'object' ? (
                  <div>
                    <h4 className="text-sm font-semibold text-muted-foreground mb-3">
                      What you can do next
                    </h4>
                    <RecommendationsSection recommendations={finalResult.recommendations as Record<string, unknown>} />
                  </div>
                ) : null}

                {/* ─── Disclaimer ─── */}
                <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4">
                  <div className="flex items-start gap-2">
                    <Info className="mt-0.5 size-4 text-amber-500" />
                    <div className="text-xs text-muted-foreground">
                      <p className="font-medium text-foreground">Important Notice</p>
                      <p className="mt-1">
                        NeuroScreen is a research/educational screening tool and does NOT provide a
                        medical diagnosis. AI-generated screening results should not be treated as
                        clinical conclusions. If you are concerned about your mental health, please
                        consider speaking with a qualified mental-health professional.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="mt-4 flex justify-between">
        <Button
          variant="outline"
          onClick={goPrev}
          disabled={stepIndex === 0}
        >
          <ChevronLeft className="size-4" /> Previous
        </Button>
        <div className="flex gap-2">
          <Button variant="ghost" onClick={handleReset}>
            <RotateCcw className="size-4" /> Reset
          </Button>
          {currentStep !== "result" && canGoNext() && (
            <Button onClick={goNext}>
              Next <ChevronRight className="size-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
