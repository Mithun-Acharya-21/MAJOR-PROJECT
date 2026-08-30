'use client'

import { useState } from 'react'
import { Sparkles, Wand2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { api, ApiError } from '@/lib/api'
import { SAMPLE_TEXT, EXAMPLE_PROMPTS } from '@/lib/constants'
import { useScreening } from '@/lib/screening-store'
import { ResultSummary } from '@/components/shared/result-summary'
import { ContributionBars } from '@/components/shared/contribution-bar'
import { StageChecklist, ErrorState, type Stage } from '@/components/shared/status-blocks'
import { LimeDialog } from './lime-dialog'
import { StepFooter } from './step-footer'

const STAGES: Stage[] = [
  { label: 'Input received', state: 'done' },
  { label: 'Extracting linguistic features', state: 'active' },
  { label: 'Running model', state: 'pending' },
  { label: 'Generating explanation', state: 'pending' },
]

export function TextAnalysis({ onNext }: { onNext: () => void }) {
  const { text, textInput, setText } = useScreening()
  const [value, setValue] = useState(textInput ?? '')
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle')
  const [error, setError] = useState('')
  const [showLime, setShowLime] = useState(false)

  const changed = text != null && value.trim() !== (textInput ?? '').trim()

  async function analyze() {
    if (value.trim().length < 10) return
    setStatus('loading')
    setError('')
    try {
      const result = await api.analyzeText(value.trim())
      setText(result, value.trim())
      setStatus('idle')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Unable to analyze your response.')
      setStatus('error')
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-2xl border border-border bg-card p-5 sm:p-6">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-3">
            <label htmlFor="text-input" className="text-sm font-medium">
              Describe how you have been feeling recently.
            </label>
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                <Wand2 className="size-3.5" />
                Use sample text
              </span>
              <Button variant="outline" size="sm" onClick={() => setValue(SAMPLE_TEXT.higher)}>
                Higher-risk example
              </Button>
              <Button variant="outline" size="sm" onClick={() => setValue(SAMPLE_TEXT.lower)}>
                Lower-risk example
              </Button>
            </div>
          </div>

          <Textarea
            id="text-input"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="Write a few sentences about your mood, energy, motivation, sleep, concentration, or daily activities..."
            className="min-h-40 resize-y rounded-xl text-[0.95rem] leading-relaxed"
            aria-describedby="char-count"
          />

          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap gap-1.5">
              {EXAMPLE_PROMPTS.map((p) => (
                <span
                  key={p}
                  className="rounded-full border border-border bg-muted/40 px-2.5 py-1 text-xs text-muted-foreground"
                >
                  {p}
                </span>
              ))}
            </div>
            <span id="char-count" className="text-xs text-muted-foreground tabular-nums">
              {value.length} characters
            </span>
          </div>

          {changed ? (
            <p className="rounded-xl bg-amber-500/10 px-3 py-2 text-xs text-risk-moderate">
              Changing this input may update your final multimodal result.
            </p>
          ) : null}

          <Button
            onClick={analyze}
            disabled={value.trim().length < 10 || status === 'loading'}
            className="rounded-xl"
          >
            <Sparkles className="size-4" />
            {status === 'loading' ? 'Analyzing your response...' : 'Analyze text'}
          </Button>
        </div>
      </div>

      {status === 'loading' ? (
        <div className="rounded-2xl border border-border bg-card p-6">
          <StageChecklist title="Analyzing your response..." stages={STAGES} />
        </div>
      ) : null}

      {status === 'error' ? (
        <ErrorState message={error} onRetry={analyze} />
      ) : null}

      {text && status === 'idle' ? (
        <div className="animate-in fade-in slide-in-from-bottom-3 duration-500">
          <div className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-5 sm:p-6">
            <h2 className="text-lg font-medium">Text analysis</h2>
            <ResultSummary
              level={text.riskLevel}
              probability={text.probability}
              confidence={text.confidence}
            />
            {text.features?.length ? (
              <div className="flex flex-col gap-4 border-t border-border pt-5">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium">Contributing features</h3>
                  {text.lime ? (
                    <Button variant="ghost" size="sm" onClick={() => setShowLime(true)}>
                      View LIME explanation
                    </Button>
                  ) : null}
                </div>
                <ContributionBars features={text.features} />
              </div>
            ) : null}
          </div>
          {text.lime ? (
            <LimeDialog
              open={showLime}
              onOpenChange={setShowLime}
              explanation={text.lime}
              title="Text — LIME explanation"
            />
          ) : null}
        </div>
      ) : null}

      <StepFooter onNext={onNext} nextDisabled={!text} nextLabel="Continue to photo" showBack={false} />
    </div>
  )
}
