'use client'

import { useState } from 'react'
import { Camera, ScanFace, CheckCircle2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { api, ApiError } from '@/lib/api'
import { useCamera } from '@/hooks/use-camera'
import { useScreening } from '@/lib/screening-store'
import { FRAME_TARGET } from '@/lib/constants'
import { ResultSummary } from '@/components/shared/result-summary'
import { ExpressionDistribution } from './expression-distribution'
import { ErrorState } from '@/components/shared/status-blocks'
import { LimeDialog } from './lime-dialog'
import { StepFooter } from './step-footer'

type Phase = 'idle' | 'capturing' | 'analyzing' | 'done' | 'error'

export function LiveFaceAnalysis({ onBack, onNext }: { onBack: () => void; onNext: () => void }) {
  const { face, setFace } = useScreening()
  const { videoRef, status, error: camError, start, stop, capture } = useCamera()
  const [phase, setPhase] = useState<Phase>(face ? 'done' : 'idle')
  const [captured, setCaptured] = useState(0)
  const [error, setError] = useState('')
  const [showLime, setShowLime] = useState(false)

  async function runCapture() {
    setPhase('capturing')
    setCaptured(0)
    const frames: Blob[] = []
    for (let i = 0; i < FRAME_TARGET; i++) {
      const frame = await capture()
      if (frame) frames.push(frame)
      setCaptured(i + 1)
      await new Promise((r) => setTimeout(r, 350))
    }
    stop()

    if (frames.length === 0) {
      setError('No frames could be captured. Please try again.')
      setPhase('error')
      return
    }

    setPhase('analyzing')
    try {
      const result = await api.analyzeFace(frames)
      setFace(result, frames)
      setPhase('done')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Unable to analyze the frames.')
      setPhase('error')
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="relative overflow-hidden rounded-2xl border border-border bg-slate-950">
        <div className="relative aspect-video w-full">
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video
            ref={videoRef}
            playsInline
            muted
            className="size-full object-cover"
            style={{ transform: 'scaleX(-1)' }}
          />

          {status !== 'active' && phase !== 'done' ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-slate-950 px-6 text-center">
              <span className="flex size-14 items-center justify-center rounded-2xl bg-white/5 text-white/80">
                <Camera className="size-6" />
              </span>
              {status === 'requesting' ? (
                <p className="text-sm text-white/70">Requesting camera access...</p>
              ) : (
                <div className="flex flex-col items-center gap-1">
                  <p className="font-medium text-white">Camera access required</p>
                  <p className="max-w-xs text-sm text-white/60 text-pretty">
                    Your browser will request permission to use your camera.
                  </p>
                </div>
              )}
            </div>
          ) : null}

          {/* face guide */}
          {status === 'active' ? (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="size-48 rounded-full border-2 border-dashed border-white/40" />
            </div>
          ) : null}

          {/* live indicator */}
          {status === 'active' ? (
            <div className="absolute right-3 top-3 flex items-center gap-1.5 rounded-full bg-black/50 px-2.5 py-1 text-xs font-medium text-white backdrop-blur">
              <span className="size-1.5 animate-pulse rounded-full bg-red-500" />
              LIVE
            </div>
          ) : null}

          {/* capture progress */}
          {phase === 'capturing' ? (
            <div className="absolute inset-x-0 bottom-0 flex flex-col gap-1.5 bg-gradient-to-t from-black/70 to-transparent p-4">
              <span className="text-xs font-medium text-white">
                Capturing frame {captured} / {FRAME_TARGET}
              </span>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/20">
                <div
                  className="h-full rounded-full bg-primary transition-[width] duration-300"
                  style={{ width: `${(captured / FRAME_TARGET) * 100}%` }}
                />
              </div>
            </div>
          ) : null}

          {phase === 'done' ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/95 text-center">
              <CheckCircle2 className="size-10 text-risk-low" />
              <p className="font-medium text-white">Analysis complete</p>
            </div>
          ) : null}
        </div>
      </div>

      {/* controls */}
      {phase === 'idle' && status !== 'active' ? (
        <Button onClick={start} className="rounded-xl">
          <Camera className="size-4" />
          Start camera
        </Button>
      ) : null}

      {status === 'active' && phase === 'idle' ? (
        <Button onClick={runCapture} className="rounded-xl">
          <ScanFace className="size-4" />
          Capture & analyze
        </Button>
      ) : null}

      {phase === 'analyzing' ? (
        <div className="flex items-center justify-center gap-2 rounded-2xl border border-border bg-card p-4 text-sm text-muted-foreground">
          <span className="size-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          Analyzing facial expressions...
        </div>
      ) : null}

      {(phase === 'error' || status === 'error') && phase !== 'done' ? (
        <ErrorState
          title="Camera error"
          message={error || camError || 'Camera access was unavailable.'}
          onRetry={() => {
            setPhase('idle')
            setError('')
            start()
          }}
        />
      ) : null}

      {face && phase === 'done' ? (
        <div className="animate-in fade-in slide-in-from-bottom-3 flex flex-col gap-6 rounded-2xl border border-border bg-card p-5 duration-500 sm:p-6">
          <div className="flex flex-col gap-4">
            <h2 className="text-lg font-medium">Facial expression signals</h2>
            <ExpressionDistribution expressions={face.expressions} />
          </div>
          <div className="border-t border-border pt-5">
            <ResultSummary
              level={face.riskLevel}
              probability={face.probability}
              confidence={face.confidence}
              label="Facial screening indicator"
            />
          </div>
          <div className="flex flex-wrap gap-2">
            {face.lime ? (
              <Button variant="ghost" size="sm" onClick={() => setShowLime(true)}>
                View explanation
              </Button>
            ) : null}
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setPhase('idle')
                setCaptured(0)
                start()
              }}
            >
              Recapture
            </Button>
          </div>
          {face.lime ? (
            <LimeDialog
              open={showLime}
              onOpenChange={setShowLime}
              explanation={face.lime}
              title="Live Face — explanation"
            />
          ) : null}
        </div>
      ) : null}

      <StepFooter onBack={onBack} onNext={onNext} nextLabel="Continue to voice" />
    </div>
  )
}
