'use client'

import { useCallback, useEffect, useState } from 'react'
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Info,
  RotateCcw,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { api, ApiError } from '@/lib/api'
import { useScreening } from '@/lib/screening-store'
import { Disclaimer } from '@/components/shared/disclaimer'
import { ResultSummary } from '@/components/shared/result-summary'
import { CONTRACT_MODALITY_LABELS, MODALITY_META } from '@/lib/constants'
import type {
  CanonicalScreeningResult,
  ModalityContractEntry,
  ModalityKind,
  PersonalizedRecommendation,
} from '@/lib/types'

/**
 * Final result page (spec §30). Sections in order:
 *   1. Final screening result (category, score, confidence, uncertainty)
 *   2. Modality breakdown — TEXT / FACE / LIVE FACE / VOICE
 *      (score, confidence, availability, signal type)
 *   3. Why this result? — LIME explanation + modality contributions
 *   4. Key screening areas (dimension profile)
 *   5. Personalized suggestions (with reason + matched area + priority)
 */

const DIMENSION_LABELS: Record<string, string> = {
  mood: 'Mood',
  loss_of_interest: 'Loss of interest',
  sleep: 'Sleep',
  energy: 'Energy',
  appetite: 'Appetite',
  concentration: 'Concentration',
  self_worth: 'Self-worth',
  guilt: 'Guilt',
  psychomotor_changes: 'Psychomotor changes',
  social_withdrawal: 'Social withdrawal',
  daily_functioning: 'Daily functioning',
}

function severityBand(severity: number): string {
  if (severity >= 2.5) return 'High'
  if (severity >= 1.5) return 'Moderate'
  if (severity >= 0.5) return 'Mild'
  return 'Low'
}

function severityToken(severity: number): string {
  if (severity >= 2.5) return 'var(--risk-high)'
  if (severity >= 1.5) return 'var(--risk-moderate)'
  return 'var(--risk-low)'
}

function severityToWidth(severity: number): number {
  return Math.max(0, Math.min(100, (severity / 3) * 100))
}

function scoreBandLabel(entry: ModalityContractEntry): string {
  if (!entry.available || entry.score == null) return 'Not available'
  const s = entry.score * Math.max(0.2, Math.min(1, entry.confidence ?? 0.5))
  if (s >= 0.62) return 'Elevated signal'
  if (s >= 0.45) return 'Moderate auxiliary signal'
  return 'Low signal'
}

type Phase = 'idle' | 'loading' | 'ready' | 'error'

/** One assessment module and its completion state (informational only). */
interface AssessmentStatusEntry {
  key: ModalityKind
  label: string
  done: boolean
  /** Stepper step id so Results can link back to any module directly. */
  stepId: number
}

export function ScreeningResultView({
  onRestart,
  onGoToStep,
}: {
  onRestart: () => void
  /** Jump to any assessment step directly from Results (free navigation). */
  onGoToStep?: (id: number) => void
}) {
  const { sessionId, reset, getRawInputs, face, voice, photo, interviewDone } =
    useScreening()
  const [phase, setPhase] = useState<Phase>('idle')
  const [result, setResult] = useState<CanonicalScreeningResult | null>(null)
  const [error, setError] = useState('')
  const [showTrace, setShowTrace] = useState(false)
  const [trace, setTrace] = useState<Record<string, unknown> | null>(null)

  const runPipeline = useCallback(async () => {
    if (!sessionId) {
      setError('No screening session found — please complete the interview first.')
      setPhase('error')
      return
    }
    setPhase('loading')
    setError('')
    try {
      // Complete the screening WITH the captured modality inputs so photo,
      // live face, and voice actually participate in the fusion.
      const raw = getRawInputs()
      let res = await api.screeningComplete({
        sessionId,
        photo: raw.photoBlob,
        faceFrames: raw.faceFrames,
        audio: raw.voiceBlob,
      })
      // Safety override responses carry no fusion payload; render as-is.
      const canonical = res as CanonicalScreeningResult
      if (canonical.screening?.safety_override || canonical.fusion == null) {
        setResult(canonical)
        setPhase('ready')
        return
      }
      setResult(canonical)
      setPhase('ready')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Unable to load your screening result.')
      setPhase('error')
    }
  }, [sessionId, getRawInputs])

  useEffect(() => {
    // Only the fusion pipeline needs a session; it requires a completed
    // interview session. Otherwise the partial/empty views render instead.
    if (phase === 'idle' && interviewDone && sessionId) void runPipeline()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const loadTrace = useCallback(async () => {
    if (!sessionId) return
    try {
      const t = await api.screeningResearchTrace(sessionId)
      setTrace(t)
    } catch {
      setTrace(null)
    }
  }, [sessionId])

  function restart() {
    reset()
    onRestart()
  }

  // Assessment completion — tracked independently of navigation. These are
  // real states from the store; nothing is fabricated for missing modules.
  const assessments: AssessmentStatusEntry[] = [
    { key: 'text', label: MODALITY_META.text.label, done: Boolean(interviewDone), stepId: 1 },
    { key: 'photo', label: MODALITY_META.photo.label, done: Boolean(photo), stepId: 2 },
    { key: 'face', label: MODALITY_META.face.label, done: Boolean(face), stepId: 3 },
    { key: 'voice', label: MODALITY_META.voice.label, done: Boolean(voice), stepId: 4 },
  ]
  const anyCompleted = assessments.some((a) => a.done)
  // The fusion pipeline runs on a completed-interview screening session.
  const canRunFusion = Boolean(interviewDone && sessionId)

  const statusStrip = (
    <ul className="flex flex-wrap gap-2" aria-label="Assessment completion status">
      {assessments.map((a) => {
        const Icon = MODALITY_META[a.key].icon
        return (
          <li
            key={a.key}
            className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-sm"
          >
            <Icon className="size-4 text-muted-foreground" aria-hidden />
            <span>{a.label}</span>
            <span className={a.done ? 'font-medium text-risk-low' : 'text-muted-foreground'}>
              {a.done ? 'Completed' : 'Not completed'}
            </span>
            {onGoToStep ? (
              <button
                type="button"
                onClick={() => onGoToStep(a.stepId)}
                className="rounded-full text-xs text-primary underline underline-offset-2 hover:text-primary/80"
              >
                Open
              </button>
            ) : null}
          </li>
        )
      })}
    </ul>
  )

  // ---- Empty state: nothing completed yet (Results stays accessible) ----
  if (!anyCompleted) {
    return (
      <div className="flex flex-col items-center gap-5 rounded-2xl border border-dashed border-border bg-card p-8 text-center sm:p-10">
        <div className="flex flex-col gap-2">
          <h2 className="text-lg font-medium">No results yet</h2>
          <p className="max-w-md text-sm text-muted-foreground text-pretty">
            No assessment results available yet. Complete any assessment to view its results
            here.
          </p>
        </div>
        {onGoToStep ? (
          <div className="flex flex-wrap justify-center gap-2">
            {assessments.map((a) => {
              const Icon = MODALITY_META[a.key].icon
              return (
                <Button
                  key={a.key}
                  variant="outline"
                  size="sm"
                  className="gap-1.5 rounded-xl"
                  onClick={() => onGoToStep(a.stepId)}
                >
                  <Icon className="size-4" />
                  Start {a.label}
                </Button>
              )
            })}
          </div>
        ) : null}
      </div>
    )
  }

  // ---- Partial state without an interview session ----
  // Sensor modalities (photo / live face / voice) can be completed in any
  // order — each one was already analyzed on its own step. Show those real
  // results and mark the rest "Not completed"; the combined multimodal
  // result requires the text interview session, so no fusion is invented.
  if (!canRunFusion) {
    return (
      <div className="flex flex-col gap-8">
        <section className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6">
          <div>
            <h2 className="text-lg font-medium">Assessment status</h2>
            <p className="text-sm text-muted-foreground">
              Every module is available at any time — results appear here as you complete them.
            </p>
          </div>
          {statusStrip}
        </section>

        {photo ? (
          <section className="flex flex-col gap-5 rounded-2xl border border-border bg-card p-5 sm:p-6">
            <h3 className="text-lg font-medium">{MODALITY_META.photo.label} analysis</h3>
            <ResultSummary
              level={photo.riskLevel}
              probability={photo.probability}
              confidence={photo.confidence}
              label="Photo screening indicator"
            />
          </section>
        ) : null}

        {face ? (
          <section className="flex flex-col gap-5 rounded-2xl border border-border bg-card p-5 sm:p-6">
            <h3 className="text-lg font-medium">{MODALITY_META.face.label} analysis</h3>
            <ResultSummary
              level={face.riskLevel}
              probability={face.probability}
              confidence={face.confidence}
              label="Facial screening indicator"
            />
          </section>
        ) : null}

        {voice ? (
          <section className="flex flex-col gap-5 rounded-2xl border border-border bg-card p-5 sm:p-6">
            <h3 className="text-lg font-medium">{MODALITY_META.voice.label} analysis</h3>
            <ResultSummary
              level={voice.riskLevel}
              probability={voice.probability}
              confidence={voice.confidence}
              label="Voice screening indicator"
            />
          </section>
        ) : null}

        <div className="flex flex-col items-start gap-3 rounded-xl border border-border bg-muted/30 p-4 text-sm sm:flex-row sm:items-center sm:justify-between">
          <p className="text-muted-foreground">
            The combined multimodal result becomes available after the text interview. No
            scores are estimated for assessments you have not completed.
          </p>
          {onGoToStep ? (
            <Button size="sm" className="shrink-0 rounded-xl" onClick={() => onGoToStep(1)}>
              Go to text interview
            </Button>
          ) : null}
        </div>

        <Disclaimer />
      </div>
    )
  }

  // ---- Loading / error states ----
  if (phase === 'loading') {
    return (
      <div className="flex items-center gap-2 rounded-2xl border border-border bg-card p-6 text-sm text-muted-foreground">
        <span className="size-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
        Running the multimodal screening pipeline (fusion → calibration → uncertainty → suggestions)…
      </div>
    )
  }

  if (phase === 'error') {
    return (
      <div className="flex flex-col gap-4 rounded-2xl border border-risk-high/30 bg-risk-high/10 p-6">
        <div className="flex items-center gap-2">
          <AlertTriangle className="size-5 text-risk-high" />
          <p className="font-medium text-risk-high">Could not compute your result</p>
        </div>
        <p className="text-sm text-muted-foreground">{error}</p>
        <div className="flex gap-3">
          <Button size="sm" onClick={() => void runPipeline()}>
            Try again
          </Button>
          <Button size="sm" variant="ghost" onClick={restart}>
            Start new screening
          </Button>
        </div>
      </div>
    )
  }

  if (!result) return null

  // ---- Safety override view ----
  const isSafety = Boolean(result.screening?.safety_override)
  if (isSafety) {
    return (
      <div className="flex flex-col gap-6">
        <div className="rounded-2xl border border-risk-high/30 bg-risk-high/10 p-6">
          <h2 className="text-xl font-semibold text-risk-high">{result.screening?.category}</h2>
          {result.message ? <p className="mt-2 text-sm">{result.message}</p> : null}
          {result.crisis_resources?.length ? (
            <ul className="mt-4 flex flex-col gap-2 text-sm">
              {result.crisis_resources.map((r) => (
                <li key={r.label}>
                  <span className="font-medium">{r.label}:</span>{' '}
                  <span className="text-muted-foreground">{r.value}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
        <Disclaimer />
        <div className="flex justify-center">
          <Button variant="ghost" onClick={restart} className="gap-1.5">
            <RotateCcw className="size-4" />
            Start new screening
          </Button>
        </div>
      </div>
    )
  }

  const fusion = result.fusion!
  const screening = result.screening!
  const profile = result.profile!
  const suggestions: PersonalizedRecommendation[] =
    result.suggestions ?? result.recommendations ?? []
  const modalities: ModalityContractEntry[] = [
    result.text,
    result.face,
    result.live_face,
    result.voice,
  ].filter(Boolean) as ModalityContractEntry[]
  const availableCount = modalities.filter((m) => m.available).length

  const dimensionEntries = Object.entries(profile.dimensions)
    .map(([key, info]) => ({
      key,
      label: DIMENSION_LABELS[key] ?? key,
      severity: Number(info?.severity ?? 0),
      confidence: Number(info?.confidence ?? 0),
    }))
    .filter((d) => d.severity > 0)
    .sort((a, b) => b.severity - a.severity)
    .slice(0, 8)

  const limeFeatures = result.explainability?.lime?.features ?? []
  const contributions = result.explainability?.modality_contributions ?? {}

  return (
    <div className="flex flex-col gap-8">
      {/* Assessment completion status (free navigation — Results is always
          reachable; incomplete modules are honestly marked). */}
      {statusStrip}

      {/* ================= 1. Final screening result ================= */}
      <section className="flex flex-col gap-5 rounded-2xl border border-border bg-card p-6">
        <div className="flex flex-col items-center gap-2 text-center">
          <div className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/50 px-3 py-1 text-xs text-muted-foreground">
            <span>Experimental Heuristic Fusion</span>
            <span>·</span>
            <span>Research Prototype</span>
          </div>
          <h2 className="text-2xl font-semibold">Multimodal screening result</h2>
          <p className="text-lg text-muted-foreground">{screening.category}</p>
        </div>
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="flex flex-col gap-1 rounded-xl border border-border bg-background/50 p-4">
            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Fused heuristic score
            </span>
            <span className="text-3xl font-semibold tabular-nums">
              {Math.round(fusion.score * 100)}%
            </span>
          </div>
          <div className="flex flex-col gap-1 rounded-xl border border-border bg-background/50 p-4">
            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Confidence
            </span>
            <span className="text-3xl font-semibold tabular-nums">
              {Math.round(fusion.confidence * 100)}%
            </span>
          </div>
          <div className="flex flex-col gap-1 rounded-xl border border-border bg-background/50 p-4">
            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Uncertainty
            </span>
            <span className="text-3xl font-semibold tabular-nums">
              {fusion.uncertainty_label ?? fusion.uncertainty.toFixed(2)}
            </span>
          </div>
        </div>
        <p className="text-xs text-muted-foreground leading-relaxed">
          {fusion.note} This score is an experimental weighted combination of multiple heterogeneous signals (symptom questionnaire, text processing, acoustics, and facial expression). It is <strong>not a clinically validated single depression diagnosis</strong>.
        </p>
        {availableCount < 4 ? (
          <p className="text-xs text-muted-foreground">
            {availableCount} of 4 modalities were available — fusion weights were
            renormalized over the available signals only.
          </p>
        ) : null}
      </section>

      {/* ================= 2. Modality breakdown ================= */}
      <section className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6">
        <div>
          <h3 className="text-lg font-medium">Independent Modality Breakdown</h3>
          <p className="text-sm text-muted-foreground">
            Signals are displayed separately by their actual model capabilities to prevent misleading diagnosis claims.
          </p>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {modalities.map((m) => {
            const meta = CONTRACT_MODALITY_LABELS[m.modality]
            const Icon = meta.icon
            const isFace = m.modality === 'face' || m.modality === 'live_face'
            const isVoice = m.modality === 'voice'
            const isText = m.modality === 'text'

            return (
              <div
                key={m.modality}
                className={`flex flex-col gap-3 rounded-xl border p-4 ${
                  m.available ? 'border-border bg-background/50' : 'border-dashed border-border bg-muted/20 opacity-70'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="inline-flex items-center gap-2 font-medium">
                    <Icon className="size-4 text-muted-foreground" />
                    {meta.label}
                  </span>
                  <div className="flex items-center gap-1.5">
                    <span className="rounded-full border border-border/80 bg-muted/40 px-2 py-0.5 text-[11px] text-muted-foreground">
                      {isFace ? 'Auxiliary Expression' : (isVoice ? 'Auxiliary Acoustic' : 'Screening Text')}
                    </span>
                    <span
                      className={`rounded-full border px-2 py-0.5 text-xs ${
                        m.available
                          ? 'border-border text-muted-foreground'
                          : 'border-dashed border-border text-muted-foreground/70'
                      }`}
                    >
                      {m.available ? 'Available' : 'Not provided'}
                    </span>
                  </div>
                </div>
                {m.available ? (
                  <>
                    {isFace ? (
                      <div className="flex flex-col gap-2">
                        <div className="flex items-baseline justify-between">
                          <span className="text-xs text-muted-foreground">Detected expression</span>
                          <span className="font-semibold text-sm rounded bg-primary/10 px-2 py-0.5 text-primary">
                            {(m.evidence as any)?.dominant_expression ?? (m.evidence as any)?.dominant_emotion ?? 'Neutral'}
                          </span>
                        </div>
                        <div className="flex items-end gap-2">
                          <span className="text-xl font-semibold tabular-nums">
                            {Math.round((m.confidence ?? 0.5) * 100)}%
                          </span>
                          <span className="mb-0.5 text-xs text-muted-foreground">
                            expression prominence (MediaPipe blendshapes)
                          </span>
                        </div>
                        <p className="text-[11px] text-muted-foreground leading-tight">
                          Auxiliary facial expression observation only — <strong>not a depression diagnosis or probability</strong>.
                        </p>
                      </div>
                    ) : (
                      <>
                        <div className="flex items-end gap-2">
                          <span className="text-2xl font-semibold tabular-nums">
                            {Math.round((m.score ?? 0) * 100)}%
                          </span>
                          <span className="mb-1 text-xs text-muted-foreground">
                            confidence {Math.round((m.confidence ?? 0) * 100)}%
                          </span>
                        </div>
                        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full rounded-full bg-primary transition-[width] duration-700"
                            style={{ width: `${Math.round((m.score ?? 0) * 100)}%` }}
                          />
                        </div>
                        <p className="text-xs text-muted-foreground">
                          <span className="font-medium">{scoreBandLabel(m)}</span>
                          {' · '}
                          {m.signal_type_label ?? m.signal_type}
                        </p>
                        {isVoice ? (
                          <p className="text-[11px] text-muted-foreground">
                            Auxiliary acoustic speech biomarkers (pitch, energy, pauses) — not a standalone clinical diagnostic.
                          </p>
                        ) : null}
                      </>
                    )}
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    This branch was not captured, so it did not contribute to the
                    fusion.
                  </p>
                )}
              </div>
            )
          })}
        </div>
      </section>

      {/* ================= 3. Why this result? ================= */}
      <section className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6">
        <div>
          <h3 className="text-lg font-medium">Why this result?</h3>
          <p className="text-sm text-muted-foreground">
            LIME explains the actual text prediction function; the contribution
            chart shows how much each available modality weighted into the
            fused score.
          </p>
        </div>

        {/* Modality contributions */}
        <div className="flex flex-col gap-3">
          {Object.entries(contributions).length ? (
            Object.entries(contributions)
              .sort(([, a], [, b]) => (b ?? 0) - (a ?? 0))
              .map(([modality, contribution]) => {
                const maxC = Math.max(...Object.values(contributions).map((c) => c ?? 0), 0.0001)
                return (
                  <div key={modality} className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="capitalize">
                        {CONTRACT_MODALITY_LABELS[modality as 'text' | 'face' | 'live_face' | 'voice']?.label ?? modality}
                      </span>
                      <span className="font-mono text-xs tabular-nums text-muted-foreground">
                        {contribution != null ? `${(contribution * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-primary/70 transition-[width] duration-700"
                        style={{ width: `${((contribution ?? 0) / maxC) * 100}%` }}
                      />
                    </div>
                  </div>
                )
              })
          ) : (
            <p className="text-sm text-muted-foreground">
              Contribution data is not available for this result.
            </p>
          )}
        </div>

        {/* LIME */}
        {limeFeatures.length ? (
          <div className="flex flex-col gap-3 border-t border-border pt-4">
            <div className="flex items-center justify-between gap-3">
              <h4 className="text-sm font-medium">
                Textual evidence (LIME — actual text prediction function)
              </h4>
              {sessionId ? (
                <Button variant="ghost" size="sm" onClick={() => void loadTrace()}>
                  View research trace
                </Button>
              ) : null}
            </div>
            <div className="flex flex-col gap-2">
              {limeFeatures.slice(0, 8).map((f, i) => {
                const increasing = f.direction === 'risk_increasing'
                const magnitude = Math.min(100, Math.abs(f.weight) * 100 * 2)
                return (
                  <div key={`${f.feature}-${i}`} className="flex items-center gap-3 text-sm">
                    <span className="w-44 shrink-0 truncate font-mono text-xs">“{f.feature}”</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${magnitude}%`,
                          backgroundColor: increasing ? 'var(--risk-high)' : 'var(--risk-low)',
                        }}
                      />
                    </div>
                    <span
                      className={`w-16 text-right font-mono text-xs tabular-nums ${
                        increasing ? 'text-risk-high' : 'text-risk-low'
                      }`}
                    >
                      {increasing ? '+' : '−'}
                      {Math.abs(f.weight).toFixed(2)}
                    </span>
                  </div>
                )
              })}
            </div>
            <p className="text-xs text-muted-foreground">{result.explainability?.lime?.label}</p>
          </div>
        ) : null}

        {showTrace && trace ? (
          <pre className="max-h-80 overflow-auto rounded-xl border border-border bg-muted/30 p-4 font-mono text-xs text-muted-foreground">
            {JSON.stringify(trace, null, 2)}
          </pre>
        ) : null}
        {showTrace && !trace && sessionId ? (
          <p className="text-xs text-muted-foreground">Loading trace…</p>
        ) : null}

        <p className="text-xs text-muted-foreground">{result.explainability?.agreement_note}</p>
      </section>

      {/* ================= 4. Key screening areas ================= */}
      <section className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6">
        <div>
          <h3 className="text-lg font-medium">Key screening areas</h3>
          <p className="text-sm text-muted-foreground">
            Structured dimensions from your responses (severity 0–3) — these
            drove the suggestions below.
          </p>
        </div>
        {dimensionEntries.length ? (
          <div className="flex flex-col gap-3">
            {dimensionEntries.map((d) => (
              <div key={d.key} className="flex items-center gap-3">
                <span className="w-40 shrink-0 text-sm">{d.label}</span>
                <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full transition-[width] duration-700 ease-out"
                    style={{
                      width: `${severityToWidth(d.severity)}%`,
                      backgroundColor: severityToken(d.severity),
                    }}
                  />
                </div>
                <span className="w-20 text-right text-xs tabular-nums text-muted-foreground">
                  {severityBand(d.severity)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            No elevated areas were recorded from your responses.
          </p>
        )}
      </section>

      {/* ================= 5. Personalized suggestions ================= */}
      <section className="flex flex-col gap-4">
        <div>
          <h3 className="text-lg font-medium">Personalized suggestions</h3>
          <p className="text-sm text-muted-foreground">
            Selected deterministically from YOUR screening profile — the reason
            under each item names the area that triggered it.
          </p>
        </div>
        {suggestions.length ? (
          <ol className="flex flex-col gap-4">
            {suggestions.map((rec, i) => (
              <li key={rec.id} className="flex flex-col gap-2 rounded-2xl border border-border bg-card p-5">
                <div className="flex items-start justify-between gap-3">
                  <h4 className="font-medium">
                    {i + 1}. {rec.title}
                  </h4>
                  <div className="flex shrink-0 gap-1.5">
                    {rec.matched_area ? (
                      <span className="rounded-full border border-border px-2 py-0.5 text-xs text-muted-foreground">
                        {DIMENSION_LABELS[rec.matched_area] ?? rec.matched_area.replace(/_/g, ' ')}
                      </span>
                    ) : null}
                    {rec.priority ? (
                      <span className="rounded-full border border-border px-2 py-0.5 text-xs text-muted-foreground">
                        priority: {rec.priority >= 0.85 ? 'high' : rec.priority >= 0.7 ? 'medium' : 'low'}
                      </span>
                    ) : null}
                  </div>
                </div>
                <p className="text-sm">
                  <span className="font-medium">Why this was suggested: </span>
                  <span className="text-muted-foreground">{rec.reason}</span>
                </p>
                <p className="text-sm">
                  <span className="font-medium">What you can try: </span>
                  <span className="text-muted-foreground">{rec.action}</span>
                </p>
              </li>
            ))}
          </ol>
        ) : (
          <p className="text-sm text-muted-foreground">
            No specific next steps were generated for this assessment.
          </p>
        )}
      </section>

      {/* ---- Footer info ---- */}
      <div className="flex items-start gap-2 rounded-xl border border-border bg-muted/30 p-4 text-xs text-muted-foreground">
        <Info className="mt-0.5 size-3.5 shrink-0" />
        <div>
          <p>
            Model version: {result.model_version}
            {screening.functional_impact ? ` · Functional impact: ${screening.functional_impact}` : ''}
            {screening.duration && screening.duration !== 'unknown' ? ` · Duration: ${screening.duration}` : ''}
          </p>
          {face || voice || photo ? (
            <p className="mt-1">
              Captured branches: {photo ? 'face photo ' : ''}
              {face ? 'live frames ' : ''}
              {voice ? 'voice' : ''}
            </p>
          ) : null}
        </div>
      </div>

      <Disclaimer />

      <div className="flex flex-wrap justify-center gap-3 pt-2">
        <Button variant="ghost" onClick={restart} className="gap-1.5">
          <RotateCcw className="size-4" />
          Start new screening
        </Button>
      </div>
    </div>
  )
}
