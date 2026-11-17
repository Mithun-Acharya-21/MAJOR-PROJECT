// Centralized API client for EmotionDep.
// All model inference happens on the backend. The client never fabricates
// predictions, weights, or metrics — it only calls the real endpoints below.

import {
  getDefaultRecommendations,
  mapFaceResult,
  mapModelMetrics,
  mapMultimodalResult,
  mapPhotoResult,
  mapRecommendations,
  mapTextResult,
  mapVoiceResult,
} from './api-mappers'
import type {
  FaceResult,
  HealthStatus,
  ModelMetrics,
  MultimodalResult,
  PersonalizedRecommendation,
  PhotoResult,
  Recommendation,
  ScreeningNextResponse,
  ScreeningResultResponse,
  ScreeningStartResponse,
  ScreeningStateResponse,
  TextResult,
  VoiceResult,
} from './types'

const BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000').replace(/\/$/, '')

export class ApiError extends Error {
  status?: number
  constructor(message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export const isApiConfigured = () => BASE_URL.length > 0

function endpoint(path: string) {
  if (!isApiConfigured()) {
    throw new ApiError(
      'The screening backend is not configured. Set NEXT_PUBLIC_API_URL to connect the model service.',
    )
  }
  return `${BASE_URL}${path}`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(endpoint(path), {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.headers ?? {}),
      },
    })
  } catch {
    throw new ApiError(
      'Unable to reach the screening service. Check your connection and try again.',
    )
  }

  if (!res.ok) {
    let detail = ''
    try {
      const body = await res.json()
      detail = body?.message ?? body?.detail ?? ''
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail || `Request failed (${res.status}).`, res.status)
  }

  try {
    return (await res.json()) as T
  } catch {
    throw new ApiError('Received an unexpected response from the screening service.')
  }
}

function formRequest<T>(path: string, form: FormData): Promise<T> {
  return request<T>(path, { method: 'POST', body: form })
}

export const api = {
  health: async (): Promise<HealthStatus> => {
    const raw = await request<{ status: string; models_loaded?: boolean }>('/api/health')
    return {
      status: raw.status === 'healthy' ? 'ok' : 'degraded',
      version: raw.models_loaded ? 'models-loaded' : 'models-pending',
    }
  },

  analyzeText: async (text: string): Promise<TextResult> => {
    const form = new FormData()
    form.append('text', text)
    const raw = await formRequest<Record<string, unknown>>('/api/analyze/text', form)
    return mapTextResult(raw)
  },

  analyzePhoto: async (image: Blob): Promise<PhotoResult> => {
    const form = new FormData()
    form.append('file', image, 'photo.jpg')
    const raw = await formRequest<Record<string, unknown>>('/api/analyze/photo', form)
    return mapPhotoResult(raw)
  },

  analyzeFace: async (frames: Blob[]): Promise<FaceResult> => {
    const form = new FormData()
    frames.forEach((frame, i) => form.append('frames', frame, `frame-${i}.jpg`))
    const raw = await formRequest<Record<string, unknown>>('/api/analyze/face', form)
    return mapFaceResult(raw)
  },

  analyzeVoice: async (audio: Blob, sessionId?: string): Promise<VoiceResult> => {
    const form = new FormData()
    form.append('audio', audio, 'recording.webm')
    if (sessionId) form.append('session_id', sessionId)
    const raw = await formRequest<Record<string, unknown>>('/api/analyze/voice', form)
    return mapVoiceResult(raw)
  },

  analyzeMultimodal: async (payload: {
    text?: string
    photo?: Blob
    faceFrames?: Blob[]
    audio?: Blob
    sessionId?: string
  }): Promise<MultimodalResult & { raw?: Record<string, unknown> }> => {
    const form = new FormData()
    if (payload.text?.trim()) form.append('text', payload.text.trim())
    if (payload.photo) form.append('photo', payload.photo, 'photo.jpg')
    if (payload.faceFrames?.length) {
      payload.faceFrames.forEach((frame, i) =>
        form.append('frames', frame, `frame-${i}.jpg`),
      )
    }
    if (payload.audio) form.append('audio', payload.audio, 'recording.webm')
    if (payload.sessionId) form.append('session_id', payload.sessionId)

    const raw = await formRequest<Record<string, unknown>>('/api/analyze/multimodal', form)
    const mapped = mapMultimodalResult(raw)
    if (raw.recommendations) {
      mapped.recommendations = raw.recommendations
    }
    return { ...mapped, raw }
  },

  metrics: async (): Promise<ModelMetrics & { source?: string; note?: string }> => {
    try {
      const raw = await request<Record<string, unknown>>('/api/model/metrics')
      return { ...mapModelMetrics(raw), source: String(raw.source ?? ''), note: String(raw.note ?? '') }
    } catch {
      const baselines = await request<{ models: Array<Record<string, unknown>>; notes?: string }>(
        '/api/baselines',
      )
      const fusion =
        baselines.models.find((m) => m.type === 'multimodal') ?? baselines.models.at(-1)!
      return {
        ...mapModelMetrics(fusion),
        source: 'baselines',
        note: baselines.notes,
      }
    }
  },

  recommendations: async (): Promise<Recommendation[]> => {
    try {
      const raw = await request<unknown>('/api/recommendations')
      if (Array.isArray(raw)) return raw as Recommendation[]
      return mapRecommendations(raw)
    } catch {
      return getDefaultRecommendations()
    }
  },

  startSession: () =>
    request<{ session_id: string; status: string }>('/session/start', { method: 'POST' }),

  // ------------------------------------------------------------------
  // Adaptive screening interview (new backend routes)
  // ------------------------------------------------------------------

  screeningStart: async (): Promise<ScreeningStartResponse> =>
    request<ScreeningStartResponse>('/api/screening/start', { method: 'POST' }),

  screeningResponse: async (
    sessionId: string,
    answer: string,
  ): Promise<ScreeningNextResponse> =>
    request<ScreeningNextResponse>(`/api/screening/${sessionId}/response`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answer }),
    }),

  /** Submit a structured option selection (with optional free-text extra).
   * The backend maps the option through the question-specific evidence bank. */
  screeningResponseOption: async (
    sessionId: string,
    answer: string,
    optionId?: string,
  ): Promise<ScreeningNextResponse> =>
    request<ScreeningNextResponse>(`/api/screening/${sessionId}/response`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(
        optionId ? { answer, option_id: optionId } : { answer },
      ),
    }),

  screeningState: async (sessionId: string): Promise<ScreeningStateResponse> =>
    request<ScreeningStateResponse>(`/api/screening/${sessionId}/state`),

  screeningResult: async (sessionId: string): Promise<ScreeningResultResponse> =>
    request<ScreeningResultResponse>(`/api/screening/${sessionId}/result`),

  screeningRecommendations: async (
    sessionId: string,
  ): Promise<{ recommendations: PersonalizedRecommendation[]; safety_override: boolean }> =>
    request<{ recommendations: PersonalizedRecommendation[]; safety_override: boolean }>(
      `/api/screening/${sessionId}/recommendations`,
    ),

  screeningComplete: async (payload: {
    sessionId: string
    photo?: Blob
    faceFrames?: Blob[]
    audio?: Blob
  }): Promise<ScreeningResultResponse> => {
    const form = new FormData()
    form.append('session_id', payload.sessionId)
    if (payload.photo) form.append('photo', payload.photo, 'photo.jpg')
    if (payload.faceFrames?.length) {
      payload.faceFrames.forEach((frame, i) => form.append('frames', frame, `frame-${i}.jpg`))
    }
    if (payload.audio) form.append('audio', payload.audio, 'recording.webm')
    return formRequest<ScreeningResultResponse>(
      `/api/screening/${payload.sessionId}/complete`,
      form,
    )
  },

  /** Research/debug trace for a completed session (spec §31/§32). */
  screeningResearchTrace: async (
    sessionId: string,
  ): Promise<Record<string, unknown>> =>
    request<Record<string, unknown>>(`/api/screening/${sessionId}/research-trace`),
}
