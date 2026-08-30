'use client'

import { useCallback, useMemo, useState } from 'react'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'
import { ScreeningHeader } from '@/components/layout/screening-header'
import { SiteFooter } from '@/components/layout/site-footer'
import { ScreeningStepper } from '@/components/screening/screening-stepper'
import { StepHeader } from '@/components/screening/step-header'
import { TextAnalysis } from '@/components/screening/text-analysis'
import { PhotoAnalysis } from '@/components/screening/photo-analysis'
import { LiveFaceAnalysis } from '@/components/screening/live-face-analysis'
import { VoiceAnalysis } from '@/components/screening/voice-analysis'
import { MultimodalResult } from '@/components/screening/multimodal-result'
import { STEPS } from '@/lib/constants'
import { useScreening } from '@/lib/screening-store'

export default function ScreeningPage() {
  const { completedSteps, isComplete } = useScreening()
  const [activeStep, setActiveStep] = useState(1)

  const completedIds = useMemo(
    () =>
      STEPS.filter((s) => s.key !== 'result' && isComplete(s.key as 'text' | 'photo' | 'face' | 'voice')).map(
        (s) => s.id,
      ),
    [isComplete, completedSteps],
  )

  const isReachable = useCallback(
    (id: number) => {
      if (id === 1) return true
      const prev = STEPS.find((s) => s.id === id - 1)
      if (!prev || prev.key === 'result') return false
      return isComplete(prev.key as 'text' | 'photo' | 'face' | 'voice') || completedIds.includes(id - 1)
    },
    [isComplete, completedIds],
  )

  const step = STEPS.find((s) => s.id === activeStep) ?? STEPS[0]

  function goNext() {
    setActiveStep((s) => Math.min(s + 1, STEPS.length))
  }

  function goBack() {
    setActiveStep((s) => Math.max(s - 1, 1))
  }

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

        <ScreeningStepper
          active={activeStep}
          completed={completedIds}
          onSelect={setActiveStep}
          isReachable={isReachable}
        />

        {step.key !== 'result' ? <StepHeader step={step} /> : null}

        {activeStep === 1 ? <TextAnalysis onNext={goNext} /> : null}
        {activeStep === 2 ? <PhotoAnalysis onBack={goBack} onNext={goNext} /> : null}
        {activeStep === 3 ? <LiveFaceAnalysis onBack={goBack} onNext={goNext} /> : null}
        {activeStep === 4 ? <VoiceAnalysis onBack={goBack} onNext={goNext} /> : null}
        {activeStep === 5 ? <MultimodalResult /> : null}
      </main>
      <SiteFooter />
    </div>
  )
}
