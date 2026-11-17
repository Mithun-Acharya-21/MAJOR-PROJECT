'use client'

import { useCallback, useMemo, useState } from 'react'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'
import { ScreeningHeader } from '@/components/layout/screening-header'
import { SiteFooter } from '@/components/layout/site-footer'
import { ScreeningStepper } from '@/components/screening/screening-stepper'
import { StepHeader } from '@/components/screening/step-header'
import { AdaptiveInterview } from '@/components/screening/adaptive-interview'
import { PhotoAnalysis } from '@/components/screening/photo-analysis'
import { LiveFaceAnalysis } from '@/components/screening/live-face-analysis'
import { VoiceAnalysis } from '@/components/screening/voice-analysis'
import { ScreeningResultView } from '@/components/screening/screening-result-view'
import { STEPS } from '@/lib/constants'
import { useScreening } from '@/lib/screening-store'

/**
 * Unified multimodal screening flow (spec §3):
 *   1. Text / assignment-based adaptive interview (modality 1)
 *   2. Face detection (photo)
 *   3. Live face analysis (multi-frame)
 *   4. Voice analysis
 *   5. Multimodal fusion → final result → explanations → personalized suggestions
 *
 * Navigation is FREE: every step (including Results) can be opened at any
 * time, in any order. Assessment completion is tracked separately and only
 * drives the completion indicators + the Results content — never access.
 */
export default function ScreeningPage() {
  const { isComplete, interviewDone, setInterviewSession } = useScreening()
  const [activeStep, setActiveStep] = useState(1)

  /** Completion indicators only — never used to restrict navigation. */
  const completedIds = useMemo(
    () =>
      STEPS.filter(
        (s) =>
          s.key !== 'result' &&
          (s.key === 'text'
            ? Boolean(interviewDone)
            : s.key === 'face_live'
              ? isComplete('face')
              : isComplete(s.key as 'photo' | 'voice')),
      ).map((s) => s.id),
    [isComplete, interviewDone],
  )

  const step = STEPS.find((s) => s.id === activeStep) ?? STEPS[0]

  function goNext() {
    setActiveStep((s) => Math.min(s + 1, STEPS.length))
  }

  function goBack() {
    setActiveStep((s) => Math.max(s - 1, 1))
  }

  /** Interview finished (or safety override) → record completion + session
   * id. The result step runs the multimodal pipeline on demand. */
  const handleInterviewComplete = useCallback(
    (sessionId: string) => {
      setInterviewSession(sessionId)
      goNext()
    },
    [setInterviewSession],
  )

  return (
    <div className="flex min-h-dvh flex-col">
      <ScreeningHeader />
      <main className="mx-auto flex w-full max-w-[1216px] flex-1 flex-col gap-8 px-4 py-8 sm:px-6">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="inline-flex h-7 items-center gap-1.5 rounded-lg px-2.5 text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            <ArrowLeft className="size-4" />
            Home
          </Link>
        </div>

        <ScreeningStepper active={activeStep} completed={completedIds} onSelect={setActiveStep} />

        {step.key !== 'result' ? <StepHeader step={step} /> : null}

        {activeStep === 1 ? (
          <AdaptiveInterview
            onComplete={handleInterviewComplete}
            onSafety={handleInterviewComplete}
            onViewResults={() => setActiveStep(5)}
          />
        ) : null}
        {activeStep === 2 ? <PhotoAnalysis onBack={goBack} onNext={goNext} /> : null}
        {activeStep === 3 ? <LiveFaceAnalysis onBack={goBack} onNext={goNext} /> : null}
        {activeStep === 4 ? <VoiceAnalysis onBack={goBack} onNext={goNext} /> : null}
        {activeStep === 5 ? (
          <ScreeningResultView
            onRestart={() => setActiveStep(1)}
            onGoToStep={(id) => setActiveStep(id)}
          />
        ) : null}
      </main>
      <SiteFooter />
    </div>
  )
}
