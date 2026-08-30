'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { CheckCircle2, RotateCcw, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { api, ApiError } from '@/lib/api'
import { useScreening } from '@/lib/screening-store'
import { ResultSummary } from '@/components/shared/result-summary'
import { Disclaimer } from '@/components/shared/disclaimer'
import { StageChecklist, ErrorState, type Stage } from '@/components/shared/status-blocks'
import { FusionVisualization } from './fusion-visualization'
import { ModalityResultCard } from './modality-result-card'
import { PerformanceMetrics } from './performance-metrics'
import { SuggestionsSection } from './suggestions-section'
import { LimeView } from './lime-view'
import type { ModalityKind } from '@/lib/types'

const FUSION_STAGES: Stage[] = [
  { label: 'Text analysis', state: 'done' },
  { label: 'Photo analysis', state: 'done' },
  { label: 'Live face analysis', state: 'done' },
  { label: 'Voice analysis', state: 'done' },
  { label: 'Multimodal fusion', state: 'active' },
  { label: 'LIME explanation', state: 'pending' },
]

export function MultimodalResult() {
  const {
    text,
    photo,
    face,
    voice,
    multimodal,
    textInput,
    sessionId,
    setMultimodal,
    getRawInputs,
    reset,
  } = useScreening()

  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>(multimodal ? 'idle' : 'loading')
  const [error, setError] = useState('')
  const [stages, setStages] = useState(FUSION_STAGES)

  async function runFusion() {
    setStatus('loading')
    setError('')
    setStages(FUSION_STAGES)

    try {
      const raw = getRawInputs()
      const result = await api.analyzeMultimodal({
        text: textInput,
        photo: raw.photoBlob,
        faceFrames: raw.faceFrames,
        audio: raw.voiceBlob,
        sessionId,
      })

      setStages((prev) =>
        prev.map((s) =>
          s.label === 'Multimodal fusion' || s.label === 'LIME explanation'
            ? { ...s, state: 'done' as const }
            : s,
        ),
      )

      const individualResults: Partial<
        Record<ModalityKind, { probability: number; confidence: number; riskLevel: typeof result.riskLevel }>
      > = {}

      if (text) individualResults.text = { probability: text.probability, confidence: text.confidence, riskLevel: text.riskLevel }
      if (photo) individualResults.photo = { probability: photo.probability, confidence: photo.confidence, riskLevel: photo.riskLevel }
      if (face) individualResults.face = { probability: face.probability, confidence: face.confidence, riskLevel: face.riskLevel }
      if (voice) individualResults.voice = { probability: voice.probability, confidence: voice.confidence, riskLevel: voice.riskLevel }

      setMultimodal({ ...result, individualResults })
      setStatus('idle')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Unable to complete multimodal analysis.')
      setStatus('error')
    }
  }

  useEffect(() => {
    if (!multimodal) runFusion()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const limeTabs = [
    text?.lime ? { key: 'text', label: 'Text', lime: text.lime } : null,
    photo?.lime ? { key: 'photo', label: 'Photo', lime: photo.lime } : null,
    face?.lime ? { key: 'face', label: 'Face', lime: face.lime } : null,
    voice?.lime ? { key: 'voice', label: 'Voice', lime: voice.lime } : null,
    multimodal?.lime ? { key: 'overall', label: 'Overall', lime: multimodal.lime } : null,
  ].filter(Boolean) as Array<{ key: string; label: string; lime: NonNullable<typeof text>['lime'] }>

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col items-center gap-2 text-center">
        <CheckCircle2 className="size-10 text-risk-low" />
        <h2 className="text-2xl font-semibold">Analysis complete</h2>
        <p className="text-muted-foreground">Multimodal screening result</p>
      </div>

      {status === 'loading' ? (
        <div className="rounded-2xl border border-border bg-card p-6">
          <StageChecklist title="Running multimodal fusion..." stages={stages} />
        </div>
      ) : null}

      {status === 'error' ? (
        <ErrorState message={error} onRetry={runFusion} />
      ) : null}

      {multimodal && status === 'idle' ? (
        <>
          <div className="rounded-2xl border border-border bg-card p-6">
            <ResultSummary
              level={multimodal.riskLevel}
              probability={multimodal.probability}
              confidence={multimodal.confidence}
              label="Overall screening indicator"
            />
          </div>

          <section className="flex flex-col gap-4">
            <h3 className="text-lg font-medium">Individual modalities</h3>
            <div className="grid gap-4 sm:grid-cols-2">
              {text ? (
                <ModalityResultCard modality="text" {...text} />
              ) : null}
              {photo ? (
                <ModalityResultCard modality="photo" {...photo} />
              ) : null}
              {face ? (
                <ModalityResultCard modality="face" {...face} />
              ) : null}
              {voice ? (
                <ModalityResultCard modality="voice" {...voice} />
              ) : null}
            </div>
          </section>

          <section className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6">
            <h3 className="text-lg font-medium">How the model reached this result</h3>
            <FusionVisualization weights={multimodal.weights} />
          </section>

          <section className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6">
            <h3 className="text-lg font-medium">Why did the model reach this result?</h3>
            <p className="text-sm text-muted-foreground">
              LIME shows which model features influenced the prediction. These features represent
              model behavior and should not be interpreted as medical causes or proof of depression.
            </p>
            {limeTabs.length ? (
              <Tabs defaultValue={limeTabs[0].key}>
                <TabsList>
                  {limeTabs.map((tab) => (
                    <TabsTrigger key={tab.key} value={tab.key}>
                      {tab.label}
                    </TabsTrigger>
                  ))}
                </TabsList>
                {limeTabs.map((tab) => (
                  <TabsContent key={tab.key} value={tab.key} className="pt-4">
                    {tab.lime ? <LimeView explanation={tab.lime} /> : null}
                  </TabsContent>
                ))}
              </Tabs>
            ) : (
              <p className="text-sm text-muted-foreground">
                Explanation data is not available for this analysis.
              </p>
            )}
          </section>

          <section className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6">
            <h3 className="text-lg font-medium">Model performance</h3>
            <PerformanceMetrics />
          </section>

          <section className="flex flex-col gap-4">
            <h3 className="text-lg font-medium">What you can do next</h3>
            <SuggestionsSection recommendations={multimodal.recommendations} />
          </section>

          {multimodal.safetyConcern ? (
            <div className="rounded-2xl border border-risk-high/30 bg-risk-high/10 p-5 text-sm">
              <p className="font-medium text-risk-high">Safety notice</p>
              <p className="mt-1 text-muted-foreground">
                Your responses may indicate immediate concern. If you are in crisis, contact local
                emergency services or a crisis helpline immediately.
              </p>
            </div>
          ) : null}

          <Disclaimer />

          <div className="flex flex-wrap justify-center gap-3 pt-2">
            <Button variant="outline" onClick={runFusion} className="gap-1.5">
              <Sparkles className="size-4" />
              Re-run fusion
            </Button>
            <Button variant="ghost" onClick={reset} className="gap-1.5">
              <RotateCcw className="size-4" />
              Start new screening
            </Button>
            <Link href="/" className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-border bg-background px-2.5 text-sm font-medium hover:bg-muted">
              Back to home
            </Link>
          </div>
        </>
      ) : null}
    </div>
  )
}
