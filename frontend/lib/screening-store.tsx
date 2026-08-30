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
  FaceResult,
  ModalityKind,
  MultimodalResult,
  PhotoResult,
  TextResult,
  VoiceResult,
} from './types'

interface ScreeningState {
  text?: TextResult
  photo?: PhotoResult
  face?: FaceResult
  voice?: VoiceResult
  multimodal?: MultimodalResult
  textInput?: string
  sessionId?: string
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
  getRawInputs: () => RawInputs
  isComplete: (modality: ModalityKind) => boolean
  reset: () => void
}

const ScreeningContext = createContext<ScreeningContextValue | null>(null)

const STORAGE_KEY = 'neuroscreen-session'

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
