'use client'

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import type {
  AdaptiveProgress,
  AdaptiveQuestion,
  CrisisResource,
  FaceResult,
  ModalityKind,
  MultimodalResult,
  PhotoResult,
  TextResult,
  VoiceResult,
} from './types'

/** In-progress (or completed) adaptive-interview snapshot. Kept in the
 * store — not component state — so the text module survives free
 * navigation between steps (returning to Text restores the interview
 * exactly where the user left off instead of restarting it). */
export interface InterviewSnapshot {
  sessionId: string
  question: AdaptiveQuestion | null
  progress: AdaptiveProgress
  history: { q: string; a: string }[]
  done: boolean
  safety: boolean
  message?: string
  crisisResources?: CrisisResource[]
  /** Currently drafted answer + selected option (restored on return). */
  draftAnswer?: string
  draftOptionId?: string | null
}

interface ScreeningState {
  text?: TextResult
  photo?: PhotoResult
  face?: FaceResult
  voice?: VoiceResult
  multimodal?: MultimodalResult
  textInput?: string
  sessionId?: string
  /** The adaptive interview (text branch) was completed. Persisted so the
   * completion indicator survives reloads; navigation never depends on it. */
  interviewDone?: boolean
  /** Live adaptive-interview state (see InterviewSnapshot). */
  interview?: InterviewSnapshot
  completedSteps: ModalityKind[]
}

/** Raw inputs kept in memory only (not persisted to localStorage). */
interface RawInputs {
  photoBlob?: Blob
  faceFrames?: Blob[]
  voiceBlob?: Blob
}

interface ScreeningContextValue extends ScreeningState {
  setText: (result: TextResult, input: string) => void
  setPhoto: (result: PhotoResult, blob: Blob) => void
  setFace: (result: FaceResult, frames: Blob[]) => void
  setVoice: (result: VoiceResult, blob: Blob) => void
  setMultimodal: (result: MultimodalResult) => void
  setSessionId: (id: string) => void
  /** Record a completed adaptive interview (its screening session id). */
  setInterviewSession: (id: string) => void
  /** Persist the live adaptive-interview snapshot (question, answers, drafts). */
  setInterview: (snapshot: InterviewSnapshot) => void
  /** Discard the interview snapshot + its completion marks (retake flow). */
  clearInterview: () => void
  getRawInputs: () => RawInputs
  isComplete: (modality: ModalityKind) => boolean
  reset: () => void
}

const ScreeningContext = createContext<ScreeningContextValue | null>(null)

const STORAGE_KEY = 'emotiondep-session'

const EMPTY: ScreeningState = { completedSteps: [] }

export function ScreeningProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<ScreeningState>(EMPTY)
  const rawInputsRef = useRef<RawInputs>({})
  const hydrated = useRef(false)

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (raw) setState(JSON.parse(raw) as ScreeningState)
    } catch {
      /* ignore */
    }
    hydrated.current = true
  }, [])

  useEffect(() => {
    if (!hydrated.current) return
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
    } catch {
      /* ignore */
    }
  }, [state])

  const markComplete = useCallback((modality: ModalityKind, prev: ScreeningState) => {
    const completedSteps = prev.completedSteps.includes(modality)
      ? prev.completedSteps
      : [...prev.completedSteps, modality]
    return completedSteps
  }, [])

  const invalidateFusion = useCallback(() => ({ multimodal: undefined as MultimodalResult | undefined }), [])

  const setText = useCallback(
    (result: TextResult, input: string) =>
      setState((prev) => ({
        ...prev,
        text: result,
        textInput: input,
        ...invalidateFusion(),
        completedSteps: markComplete('text', prev),
      })),
    [markComplete, invalidateFusion],
  )

  const setPhoto = useCallback(
    (result: PhotoResult, blob: Blob) => {
      rawInputsRef.current.photoBlob = blob
      setState((prev) => ({
        ...prev,
        photo: result,
        ...invalidateFusion(),
        completedSteps: markComplete('photo', prev),
      }))
    },
    [markComplete, invalidateFusion],
  )

  const setFace = useCallback(
    (result: FaceResult, frames: Blob[]) => {
      rawInputsRef.current.faceFrames = frames
      setState((prev) => ({
        ...prev,
        face: result,
        ...invalidateFusion(),
        completedSteps: markComplete('face', prev),
      }))
    },
    [markComplete, invalidateFusion],
  )

  const setVoice = useCallback(
    (result: VoiceResult, blob: Blob) => {
      rawInputsRef.current.voiceBlob = blob
      setState((prev) => ({
        ...prev,
        voice: result,
        ...invalidateFusion(),
        completedSteps: markComplete('voice', prev),
      }))
    },
    [markComplete, invalidateFusion],
  )

  const setMultimodal = useCallback(
    (result: MultimodalResult) => setState((prev) => ({ ...prev, multimodal: result })),
    [],
  )

  const setSessionId = useCallback((id: string) => {
    setState((prev) => ({ ...prev, sessionId: id }))
  }, [])

  const setInterviewSession = useCallback(
    (id: string) =>
      setState((prev) => ({
        ...prev,
        sessionId: id,
        interviewDone: true,
        completedSteps: markComplete('text', prev),
      })),
    [markComplete],
  )

  const setInterview = useCallback(
    (snapshot: InterviewSnapshot) =>
      setState((prev) => ({
        ...prev,
        interview: snapshot,
        // A finished (or safety-override) interview also records completion,
        // mirroring setInterviewSession so both paths stay consistent.
        ...(snapshot.done || snapshot.safety
          ? {
              sessionId: snapshot.sessionId,
              interviewDone: true,
              completedSteps: markComplete('text', prev),
            }
          : {}),
      })),
    [markComplete],
  )

  const clearInterview = useCallback(
    () =>
      setState((prev) => ({
        ...prev,
        interview: undefined,
        interviewDone: false,
        sessionId: undefined,
        completedSteps: prev.completedSteps.filter((m) => m !== 'text'),
        // Text evidence changed — any fused result is stale.
        ...invalidateFusion(),
      })),
    [invalidateFusion],
  )

  const getRawInputs = useCallback(() => ({ ...rawInputsRef.current }), [])

  const reset = useCallback(() => {
    setState(EMPTY)
    rawInputsRef.current = {}
    try {
      localStorage.removeItem(STORAGE_KEY)
    } catch {
      /* ignore */
    }
  }, [])

  const isComplete = useCallback(
    (modality: ModalityKind) => state.completedSteps.includes(modality),
    [state.completedSteps],
  )

  const value = useMemo<ScreeningContextValue>(
    () => ({
      ...state,
      setText,
      setPhoto,
      setFace,
      setVoice,
      setMultimodal,
      setSessionId,
      setInterviewSession,
      setInterview,
      clearInterview,
      getRawInputs,
      isComplete,
      reset,
    }),
    [
      state,
      setText,
      setPhoto,
      setFace,
      setVoice,
      setMultimodal,
      setSessionId,
      setInterviewSession,
      setInterview,
      clearInterview,
      getRawInputs,
      isComplete,
      reset,
    ],
  )

  return <ScreeningContext.Provider value={value}>{children}</ScreeningContext.Provider>
}

export function useScreening() {
  const ctx = useContext(ScreeningContext)
  if (!ctx) throw new Error('useScreening must be used within ScreeningProvider')
  return ctx
}
