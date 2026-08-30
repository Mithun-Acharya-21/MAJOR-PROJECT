'use client'

import { useEffect, useState } from 'react'
import { Mic, Square, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { api, ApiError } from '@/lib/api'
import { useRecorder } from '@/hooks/use-recorder'
import { useScreening } from '@/lib/screening-store'
import { ResultSummary } from '@/components/shared/result-summary'
import { ContributionBars } from '@/components/shared/contribution-bar'
import { StageChecklist, ErrorState, type Stage } from '@/components/shared/status-blocks'
import { LimeDialog } from './lime-dialog'
import { StepFooter } from './step-footer'

const STAGES: Stage[] = [
  { label: 'Audio received', state: 'done' },
  { label: 'Transcribing speech', state: 'active' },
  { label: 'Extracting voice features', state: 'pending' },
  { label: 'Running model', state: 'pending' },
]

function formatTime(seconds: number) {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export function VoiceAnalysis({ onBack, onNext }: { onBack: () => void; onNext: () => void }) {
  const { voice, setVoice, sessionId, setSessionId } = useScreening()
  const { status, error: micError, seconds, amplitude, blob, start, stop, reset } = useRecorder()
  const [phase, setPhase] = useState<'idle' | 'loading' | 'error'>('idle')
  const [error, setError] = useState('')
  const [showLime, setShowLime] = useState(false)
  const [transcript, setTranscript] = useState(voice?.transcript ?? '')

  useEffect(() => {
    if (!sessionId) {
      api.startSession().then((s) => setSessionId(s.session_id)).catch(() => {})
    }
  }, [sessionId, setSessionId])

  async function analyze() {
    if (!blob) return
    setPhase('loading')
    setError('')
    try {
      const result = await api.analyzeVoice(blob, sessionId)
      setVoice(result, blob)
      setTranscript(result.transcript)
      setPhase('idle')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Unable to analyze your recording.')
      setPhase('error')
    }
  }

  const waveformBars = Array.from({ length: 24 }, (_, i) => {
    const offset = Math.sin((i + seconds) * 0.5) * 0.3
    const h = status === 'recording' ? Math.max(0.15, amplitude + offset) : 0.12
    return h
  })

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col items-center gap-6 rounded-2xl border border-border bg-card p-8">
        <span className="flex size-16 items-center justify-center rounded-full bg-primary/10 text-primary">
          <Mic className="size-7" />
        </span>

        <div className="text-center">
          <p className="font-medium">
            {status === 'recording'
              ? 'Recording...'
              : status === 'stopped'
                ? 'Recording complete'
                : 'Ready to record'}
          </p>
          {status === 'recording' || status === 'stopped' ? (
            <p className="mt-1 font-mono text-sm text-muted-foreground tabular-nums">
              {formatTime(seconds)}
            </p>
          ) : (
            <p className="mt-1 text-sm text-muted-foreground">
              Speak naturally about how you have been feeling.
            </p>
          )}
        </div>

        {/* Waveform */}
        <div
          className="flex h-12 w-full max-w-md items-end justify-center gap-1"
          aria-hidden
        >
          {waveformBars.map((h, i) => (
            <div
              key={i}
              className="w-1.5 rounded-full bg-primary/60 transition-all duration-150"
              style={{ height: `${h * 100}%` }}
            />
          ))}
        </div>

        <div className="flex flex-wrap justify-center gap-3">
          {status === 'idle' || status === 'error' ? (
            <Button onClick={start} className="rounded-xl">
              <Mic className="size-4" />
              Start recording
            </Button>
          ) : null}
          {status === 'recording' ? (
            <Button variant="destructive" onClick={stop} className="rounded-xl">
              <Square className="size-4" />
              Stop recording
            </Button>
          ) : null}
          {status === 'stopped' && !voice ? (
            <>
              <Button onClick={analyze} disabled={phase === 'loading'} className="rounded-xl">
                <Sparkles className="size-4" />
                {phase === 'loading' ? 'Analyzing...' : 'Analyze voice'}
              </Button>
              <Button variant="outline" onClick={reset}>
                Record again
              </Button>
            </>
          ) : null}
        </div>
      </div>

      {phase === 'loading' ? (
        <div className="rounded-2xl border border-border bg-card p-6">
          <StageChecklist title="Analyzing your voice..." stages={STAGES} />
        </div>
      ) : null}

      {(phase === 'error' || (status === 'error' && !voice)) ? (
        <ErrorState
          title="Recording error"
          message={error || micError || 'Microphone access was unavailable.'}
          onRetry={() => {
            setPhase('idle')
            setError('')
            reset()
          }}
        />
      ) : null}

      {transcript ? (
        <div className="rounded-2xl border border-border bg-card p-5">
          <h3 className="text-sm font-medium text-muted-foreground">Complete transcript</h3>
          <p className="mt-2 text-sm leading-relaxed text-foreground/90">{transcript}</p>
        </div>
      ) : null}

      {voice && phase === 'idle' ? (
        <div className="animate-in fade-in slide-in-from-bottom-3 flex flex-col gap-6 rounded-2xl border border-border bg-card p-5 sm:p-6">
          <h2 className="text-lg font-medium">Voice analysis</h2>
          <ResultSummary
            level={voice.riskLevel}
            probability={voice.probability}
            confidence={voice.confidence}
            label="Voice screening indicator"
          />
          {voice.features?.speakingRate > 0 ? (
            <div className="grid gap-3 border-t border-border pt-5 sm:grid-cols-3">
              <VoiceMetric label="Speaking rate" value={`${voice.features.speakingRate.toFixed(1)}/s`} />
              <VoiceMetric label="Pause ratio" value={`${Math.round(voice.features.pauseRatio * 100)}%`} />
              <VoiceMetric label="Energy" value={voice.features.energy.toFixed(3)} />
            </div>
          ) : null}
          {voice.lime?.features?.length ? (
            <div className="border-t border-border pt-5">
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-sm font-medium">Contributing features</h3>
                <Button variant="ghost" size="sm" onClick={() => setShowLime(true)}>
                  View LIME explanation
                </Button>
              </div>
              <ContributionBars features={voice.lime.features} />
            </div>
          ) : null}
          {voice.lime ? (
            <LimeDialog
              open={showLime}
              onOpenChange={setShowLime}
              explanation={voice.lime}
              title="Voice — LIME explanation"
            />
          ) : null}
        </div>
      ) : null}

      <StepFooter
        onBack={onBack}
        onNext={onNext}
        nextDisabled={!voice}
        nextLabel="View multimodal result"
      />
    </div>
  )
}

function VoiceMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border bg-background/50 p-3">
      <span className="text-xs text-muted-foreground">{label}</span>
      <p className="mt-1 font-mono text-sm font-medium">{value}</p>
    </div>
  )
}
