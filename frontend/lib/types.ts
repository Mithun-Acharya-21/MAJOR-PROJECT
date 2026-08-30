// Shared API contract types for NeuroScreen.
// These mirror the expected backend responses. Where the live backend differs,
// adapt the client mapping in lib/api.ts rather than fabricating data.

export type RiskLevel = 'low' | 'moderate' | 'high'

export type ModalityKind = 'text' | 'photo' | 'face' | 'voice'

/** A single feature/token and how strongly it pushed the prediction. */
export interface FeatureContribution {
  /** Human-readable feature or token, e.g. "difficulty concentrating". */
  label: string
  /** Signed influence weight. Positive = risk-increasing, negative = risk-decreasing. */
  weight: number
  /** Optional normalized magnitude 0..1 for bar rendering. Derived if absent. */
  magnitude?: number
}

export interface LimeExplanation {
  modality: ModalityKind | 'overall'
  features: FeatureContribution[]
  /** Optional data URL / URL for an image explanation overlay (photo/face). */
  imageOriginalUrl?: string
  imageOverlayUrl?: string
}

/** Fields common to every per-modality result. */
export interface ModalityResultBase {
  modality: ModalityKind
  riskLevel: RiskLevel
  /** Probability 0..1. */
  probability: number
  /** Confidence 0..1. */
  confidence: number
  lime?: LimeExplanation
}

export interface TextResult extends ModalityResultBase {
  modality: 'text'
  features: FeatureContribution[]
}

export interface PhotoResult extends ModalityResultBase {
  modality: 'photo'
  faceDetected: boolean
}

export interface FaceExpression {
  label: string
  /** Proportion 0..1. */
  value: number
}

export interface FaceResult extends ModalityResultBase {
  modality: 'face'
  expressions: FaceExpression[]
  framesAnalyzed: number
}

export interface VoiceFeatures {
  speakingRate: number // syllables/sec
  pauseRatio: number // 0..1
  pitchMean: number // Hz
  pitchVariation: number // Hz
  energy: number // 0..1
  durationSeconds: number
}

export interface VoiceResult extends ModalityResultBase {
  modality: 'voice'
  transcript: string
  features: VoiceFeatures
}

/** Weighted contribution of a modality to the fused result. */
export interface ModalityWeight {
  modality: ModalityKind
  /** Weight 0..1 as returned by the backend fusion model. */
  weight: number
}

export interface MultimodalResult {
  riskLevel: RiskLevel
  probability: number
  confidence: number
  /** Actual fusion weights from the backend. Never hard-coded on the client. */
  weights: ModalityWeight[]
  /** Which modalities were actually included in the fusion. */
  modalitiesUsed: ModalityKind[]
  /** Optional safety flag raised by the backend. */
  safetyConcern?: boolean
  /** Optional region-specific crisis resources supplied by the backend. */
  crisisResources?: { label: string; value: string }[]
  lime?: LimeExplanation
  /** Backend-generated wellness recommendations when available. */
  recommendations?: unknown
  /** Per-modality summaries from fusion (backend individual_results). */
  individualResults?: Partial<
    Record<ModalityKind, { probability: number; confidence: number; riskLevel: RiskLevel }>
  >
}

export interface ModelMetrics {
  accuracy: number
  precision: number
  recall: number
  f1: number
  rocAuc: number
}

export interface Recommendation {
  category: string
  items: string[]
}

export interface HealthStatus {
  status: 'ok' | 'degraded' | 'down'
  version?: string
}

export type AnyModalityResult = TextResult | PhotoResult | FaceResult | VoiceResult
