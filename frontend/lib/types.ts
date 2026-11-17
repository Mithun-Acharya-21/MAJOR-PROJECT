// Shared API contract types for EmotionDep.
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
  dominantExpression?: string
  uncertaintyLabel?: string
  expressionProminence?: number
  observedMovement?: Record<string, number>
  expressionDistribution?: Record<string, number>
  expressions?: FaceExpression[]
  modelName?: string
  modelType?: string
  processingStatus?: string
  confidenceSource?: string
  isDepressionModel?: boolean
  note?: string
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

// ---------------------------------------------------------------------------
// Adaptive screening interview
// ---------------------------------------------------------------------------

export interface AdaptiveProgress {
  questions_answered: number
  min_questions: number
  max_questions: number
  progress_percent: number
  required_dimensions_covered: boolean
  dimension_coverage: Record<string, number>
  missing_information: string[]
}

export interface AdaptiveQuestion {
  id: string
  text: string
  target_dimensions: string[]
  /** Question-SPECIFIC options — each question measures a different
   * construct, so option sets differ per question (never a shared scale). */
  options: { id: string; label: string }[]
}

/** One modality entry of the canonical four-modality contract (§17). */
export interface ModalityContractEntry {
  modality: 'text' | 'face' | 'live_face' | 'voice'
  score: number | null
  confidence: number | null
  signal_type: string | null
  signal_type_label: string | null
  available: boolean
  risk_level?: string | null
  evidence?: Record<string, unknown>
}

export interface ScreeningFusionSummary {
  score: number
  confidence: number
  uncertainty: number
  uncertainty_label?: string
  agreement?: number | null
  fusion_type?: string
  validated: boolean
  validated_modalities: string[]
  weights: Record<string, number>
  note: string
}

export interface ScreeningProfileSummary {
  overall_screening_level: string
  dimensions: Record<string, { severity: number; confidence: number }>
  functional_impact: string
  duration: string
  uncertainty: number
  modalities_used: string[]
}

export interface LimeExplanationPayload {
  label: string
  features: { feature: string; weight: number; direction: string; contribution?: number }[]
}

/** Canonical screening result response (§42) — one shape for the whole page. */
export interface CanonicalScreeningResult {
  text: ModalityContractEntry
  face: ModalityContractEntry
  live_face: ModalityContractEntry
  voice: ModalityContractEntry
  fusion: ScreeningFusionSummary
  screening: {
    category: string
    category_key?: string
    non_diagnostic: boolean
    functional_impact?: string
    duration?: string
    safety_override?: boolean
  }
  explainability: {
    lime: LimeExplanationPayload | null
    modality_contributions: Record<string, number | null>
    agreement_note: string
  }
  profile: ScreeningProfileSummary
  suggestions: PersonalizedRecommendation[]
  recommendations?: PersonalizedRecommendation[]
  message?: string | null
  crisis_resources?: CrisisResource[] | null
  disclaimer: string
  model_version: string
}

export interface ScreeningStartResponse {
  session_id: string
  question: AdaptiveQuestion
  progress: AdaptiveProgress
  status: string
}

export interface CrisisResource {
  label: string
  value: string
}

export interface ScreeningNextResponse {
  session_id: string
  safety_concern: boolean
  question: AdaptiveQuestion | null
  done: boolean
  stop_reason?: string
  message?: string
  crisis_resources?: CrisisResource[]
  progress: AdaptiveProgress
  status: string
}

export interface ScreeningStateResponse extends Record<string, unknown> {
  session_id: string
  questions: { id: string; text: string }[]
  features: Record<string, { severity: number; confidence: number }>
  question_count: number
  status: string
  progress: AdaptiveProgress
}

/** A personalized recommendation returned by the screening engine. */
export interface PersonalizedRecommendation {
  id: string
  title: string
  reason: string
  action: string
  category: string
  /** Screening area that drove this suggestion (spec §27). */
  matched_area?: string
  priority: number
  evidence_level?: string
}

export interface ScreeningResultResponse extends Partial<CanonicalScreeningResult> {
  /** Backwards-compatible nested summary. */
  screening_result?: {
    category: string
    category_key?: string
    safety_override?: boolean
    probability: number | null
    uncertainty: number | null
    uncertainty_label?: string
    dimensions: Record<string, { severity: number; confidence: number }>
    dimension_coverage?: Record<string, number>
    functional_impact?: string
    duration?: string
    modalities_used: string[]
    model_version: string
    calibration_method?: string
    fusion?: {
      weights: Record<string, number>
      individual_results?: Record<string, unknown>
    }
    questions_answered?: number
    model_failure?: string[]
    low_evidence_note?: string
  }
  message?: string | null
  recommendations: PersonalizedRecommendation[]
  crisis_resources?: CrisisResource[] | null
  disclaimer: string
}
