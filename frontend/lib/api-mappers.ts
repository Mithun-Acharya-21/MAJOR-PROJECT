/**
 * Maps backend snake_case responses (probabilities 0–100) to frontend types (0–1).
 * The backend is the source of truth — no values are fabricated here.
 */

import type {
  FaceResult,
  FeatureContribution,
  LimeExplanation,
  ModalityKind,
  ModelMetrics,
  MultimodalResult,
  PhotoResult,
  Recommendation,
  RiskLevel,
  TextResult,
  VoiceResult,
} from './types'

function toUnit(value: number | undefined): number {
  if (value == null || Number.isNaN(value)) return 0
  return value > 1 ? value / 100 : value
}

function toRiskLevel(level: string | undefined): RiskLevel {
  if (level === 'high' || level === 'moderate' || level === 'low') return level
  return 'low'
}

function mapFeatures(raw: unknown): FeatureContribution[] {
  if (!raw) return []

  if (Array.isArray(raw)) {
    return raw.map((item) => {
      const rec = item as Record<string, unknown>
      const label = String(rec.feature ?? rec.token ?? rec.label ?? '')
      const weightRaw = Number(rec.contribution ?? rec.weight ?? 0)
      const direction = String(rec.direction ?? '')
      const signed =
        direction === 'risk_decreasing' || weightRaw < 0 ? -Math.abs(weightRaw) : Math.abs(weightRaw)
      return { label, weight: signed }
    })
  }

  if (typeof raw === 'object') {
    const obj = raw as Record<string, unknown>
    if (Array.isArray(obj.top_features)) return mapFeatures(obj.top_features)
    if (Array.isArray(obj.attention_tokens)) {
      return (obj.attention_tokens as Array<{ token: string; weight: number }>).map((t) => ({
        label: t.token,
        weight: t.weight,
      }))
    }
  }

  return []
}

function mapLime(
  modality: ModalityKind | 'overall',
  explanation: unknown,
  imageOriginal?: string,
): LimeExplanation | undefined {
  const features = mapFeatures(explanation)
  const obj = (typeof explanation === 'object' && explanation ? explanation : {}) as Record<
    string,
    unknown
  >
  const overlay = (obj.gradcam ?? obj.overlay ?? obj.heatmap) as string | undefined
  const original = (obj.original ?? imageOriginal) as string | undefined

  if (!features.length && !overlay) return undefined

  return {
    modality,
    features,
    imageOriginalUrl: original,
    imageOverlayUrl: overlay,
  }
}

export function mapTextResult(raw: Record<string, unknown>): TextResult {
  const explanation = raw.explanation
  const features = mapFeatures(explanation)
  const lime =
    mapLime('text', explanation) ??
    (features.length ? { modality: 'text' as const, features } : undefined)

  return {
    modality: 'text',
    riskLevel: toRiskLevel(raw.risk_level as string),
    probability: toUnit(raw.probability as number),
    confidence: toUnit(raw.confidence as number),
    features,
    lime,
  }
}

export function mapPhotoResult(raw: Record<string, unknown>): PhotoResult {
  const explanation = (raw.explanation ?? {}) as Record<string, unknown>
  const featuresObj = (raw.features ?? {}) as Record<string, unknown>
  const faceDetected = Boolean(explanation.face_detected ?? featuresObj.face_detected ?? true)
  const lime = mapLime('photo', explanation)

  return {
    modality: 'photo',
    riskLevel: toRiskLevel(raw.risk_level as string),
    probability: toUnit(raw.probability as number),
    confidence: toUnit(raw.confidence as number),
    faceDetected,
    lime,
  }
}

export function mapFaceResult(raw: Record<string, unknown>): FaceResult {
  const explanation = (raw.explanation ?? {}) as Record<string, unknown>
  const features = (raw.features ?? {}) as Record<string, unknown>
  const dist =
    (explanation.expression_distribution as Record<string, number>) ??
    (features.expression_distribution as Record<string, number>) ??
    {}

  const expressions = Object.entries(dist).map(([label, value]) => ({
    label: label.charAt(0).toUpperCase() + label.slice(1),
    value: toUnit(value),
  }))

  const limeFeatures = mapFeatures(explanation)
  const lime = limeFeatures.length
    ? { modality: 'face' as const, features: limeFeatures }
    : undefined

  return {
    modality: 'face',
    riskLevel: toRiskLevel(raw.risk_level as string),
    probability: toUnit(raw.probability as number),
    confidence: toUnit(raw.confidence as number),
    expressions,
    framesAnalyzed: Number(explanation.frames_analyzed ?? features.analyzed_count ?? 0),
    lime,
  }
}

export function mapVoiceResult(raw: Record<string, unknown>): VoiceResult {
  const vf = (raw.voice_features ?? raw.features ?? {}) as Record<string, number>
  const explanation = raw.explanation
  const features = mapFeatures(explanation)
  const lime = features.length ? { modality: 'voice' as const, features } : undefined

  return {
    modality: 'voice',
    riskLevel: toRiskLevel(raw.risk_level as string),
    probability: toUnit(raw.probability as number),
    confidence: toUnit(raw.confidence as number),
    transcript: String(raw.transcript ?? ''),
    features: {
      speakingRate: vf.speaking_rate ?? 0,
      pauseRatio: vf.silence_ratio ?? 0,
      pitchMean: vf.pitch_mean ?? 0,
      pitchVariation: vf.pitch_variation ?? 0,
      energy: vf.rms_energy ?? 0,
      durationSeconds: vf.duration_seconds ?? 0,
    },
    lime,
  }
}

export function mapMultimodalResult(raw: Record<string, unknown>): MultimodalResult {
  const fusionWeights = (raw.fusion_weights ?? {}) as Record<string, number>
  const explanation = (raw.explanation ?? {}) as Record<string, unknown>
  const contributions = (explanation.contributions ?? []) as Array<{
    modality: string
    weight: number
  }>

  const weights =
    contributions.length > 0
      ? contributions.map((c) => ({
          modality: c.modality as ModalityKind,
          weight: c.weight,
        }))
      : Object.entries(fusionWeights).map(([modality, weight]) => ({
          modality: modality as ModalityKind,
          weight,
        }))

  const modalitiesUsed = (explanation.available_modalities as ModalityKind[]) ??
    Object.keys(fusionWeights) as ModalityKind[]

  const recs = raw.recommendations as Record<string, unknown> | undefined
  const safetyConcern = Boolean(recs?.safety_concern)

  const textExpl = (raw.individual_results as Record<string, unknown>)?.text
  const overallLime = textExpl
    ? mapLime('overall', (textExpl as Record<string, unknown>).explanation)
    : undefined

  return {
    riskLevel: toRiskLevel(raw.risk_level as string),
    probability: toUnit(raw.probability as number),
    confidence: toUnit(raw.confidence as number),
    weights,
    modalitiesUsed,
    safetyConcern,
    lime: overallLime,
    recommendations: recs,
  }
}

export function mapModelMetrics(raw: Record<string, unknown>): ModelMetrics {
  return {
    accuracy: Number(raw.accuracy ?? 0),
    precision: Number(raw.precision ?? 0),
    recall: Number(raw.recall ?? 0),
    f1: Number(raw.f1 ?? 0),
    rocAuc: Number(raw.roc_auc ?? raw.auc ?? 0),
  }
}

export function mapRecommendations(raw: unknown): Recommendation[] {
  if (!raw || typeof raw !== 'object') return getDefaultRecommendations()

  const obj = raw as Record<string, unknown>
  const items: Recommendation[] = []

  const whatYouCanDo = obj.what_you_can_do as Record<string, string[]> | undefined
  if (whatYouCanDo?.today?.length) {
    items.push({ category: 'Daily wellbeing', items: whatYouCanDo.today })
  }
  if (whatYouCanDo?.this_week?.length) {
    items.push({ category: 'This week', items: whatYouCanDo.this_week })
  }
  if (whatYouCanDo?.when_to_seek_help?.length) {
    items.push({ category: 'Consider professional support', items: whatYouCanDo.when_to_seek_help })
  }

  if (Array.isArray(obj.coping_strategies) && obj.coping_strategies.length) {
    items.push({ category: 'Coping strategies', items: obj.coping_strategies as string[] })
  }

  if (items.length === 0) return getDefaultRecommendations()
  return items
}

export function getDefaultRecommendations(): Recommendation[] {
  return [
    {
      category: 'Daily wellbeing',
      items: [
        'Maintain a regular daily routine',
        'Get adequate rest',
        'Take regular breaks',
        'Stay connected with supportive people',
        'Consider physical activity',
        'Spend time doing meaningful activities',
      ],
    },
    {
      category: 'Consider professional support',
      items: [
        'If these concerns persist or interfere with your daily life, consider speaking with a qualified mental-health professional.',
      ],
    },
  ]
}
