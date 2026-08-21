/**
 * TypeScript types mirroring the FastAPI response schemas (Phase 18).
 * Keep in sync with backend/app/schemas/analysis.py.
 */

export interface ModalityQuality {
  quality_score: number;
  is_usable: boolean;
  rejection_reasons: string[];
}

export interface EmotionProfile {
  probabilities: Record<string, number>;
  predicted_emotion: string;
}

export interface GateSummary {
  modality: string;
  gate_value: number;
  contribution_pct: number;
  quality_score: number | null;
  active: boolean;
}

// ---- Responses ----

export interface HealthResponse {
  status: "healthy";
  service: string;
  environment: string;
}

export interface ModelInfoResponse {
  model_version: string;
  readiness: string;
  supported_modalities: string[];
  raw_input_storage_default: boolean;
  disclaimer: string;
}

export interface ConsentResponse {
  consent_type: string;
  accepted: boolean;
  note: string;
}

export interface TextAnalysisResponse {
  request_id: string;
  risk_score: number | null;
  risk_tier: string | null;
  uncertainty: number | null;
  top_tokens: string[];
  model_available: boolean;
  disclaimer: string;
  note: string;
}

export interface VoiceAnalysisResponse {
  request_id: string;
  transcript: string | null;
  language: string | null;
  asr_quality: ModalityQuality | null;
  audio_quality: ModalityQuality | null;
  risk_score: number | null;
  risk_tier: string | null;
  emotion_profile: EmotionProfile | null;
  combined_quality_score: number | null;
  model_available: boolean;
  disclaimer: string;
}

export interface ImageAnalysisResponse {
  request_id: string;
  face_detected: boolean;
  n_faces: number;
  face_quality: ModalityQuality | null;
  emotion_profile: EmotionProfile | null;
  gradcam_heatmap_base64: string | null;
  model_available: boolean;
  disclaimer: string;
  note: string;
}

export interface VideoAnalysisResponse {
  request_id: string;
  n_frames_sampled: number;
  n_frames_with_face: number;
  mean_face_quality: number | null;
  reliable_frame_ratio: number | null;
  emotion_profile: EmotionProfile | null;
  model_available: boolean;
  disclaimer: string;
}

export interface MultimodalAnalysisResponse {
  request_id: string;
  used_modalities: string[];
  missing_modalities: string[];
  risk_score: number | null;
  risk_tier: string | null;
  uncertainty: number | null;
  gate_contributions: GateSummary[];
  emotion_profile: EmotionProfile | null;
  suggestions: Array<{ category: string; text: string }>;
  policy_disclaimer: string | null;
  model_available: boolean;
  disclaimer: string;
}

// ---- Risk tier helpers ----

export type RiskTier = "low" | "moderate" | "high" | "crisis" | null;

export function riskTierColor(tier: RiskTier): string {
  switch (tier) {
    case "low":      return "#22c55e"; // green
    case "moderate": return "#f59e0b"; // amber
    case "high":     return "#ef4444"; // red
    case "crisis":   return "#7c3aed"; // violet
    default:         return "#6b7280"; // gray
  }
}

export function riskTierLabel(tier: RiskTier): string {
  switch (tier) {
    case "low":      return "Low Concern";
    case "moderate": return "Moderate Concern";
    case "high":     return "High Concern";
    case "crisis":   return "Crisis — Seek Immediate Support";
    default:         return "No Score Yet";
  }
}
