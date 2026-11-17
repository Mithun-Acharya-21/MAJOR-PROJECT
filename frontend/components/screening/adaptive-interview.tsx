'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { CheckCircle2, Loader2, Send, ShieldAlert } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Progress } from '@/components/ui/progress'
import { api, ApiError } from '@/lib/api'
import { useScreening } from '@/lib/screening-store'
import type {
  AdaptiveProgress,
  AdaptiveQuestion,
  CrisisResource,
} from '@/lib/types'

const SAMPLE_RESPONSES = [
  "I've been sleeping very badly and I feel tired most days.",
  'Work has been overwhelming, I can barely focus on anything.',
  "I've stopped seeing friends, I don't enjoy much anymore.",
  'Things have been mostly okay, a few normal ups and downs.',
]

interface InterviewState {
  sessionId: string
  question: AdaptiveQuestion | null
  progress: AdaptiveProgress
  done: boolean
  safety: boolean
  message?: string
  crisisResources?: CrisisResource[]
}

export function AdaptiveInterview({
  onComplete,
  onSafety,
  onViewResults,
}: {
  onComplete: (sessionId: string) => void
  onSafety: (sessionId: string) => void
  /** Jump to the Results step from the completed-interview panel. */
  onViewResults?: () => void
}) {
  const { interview, interviewDone, sessionId, setInterview, clearInterview } = useScreening()
  const [state, setState] = useState<InterviewState | null>(null)
  const [answer, setAnswer] = useState('')
  const [selectedOption, setSelectedOption] = useState<string | null>(null)
  const [history, setHistory] = useState<{ q: string; a: string }[]>([])
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  const startInterview = useCallback(() => {
    let cancelled = false
    api
      .screeningStart()
      .then((res) => {
        if (!cancelled) {
          const next: InterviewState = {
            sessionId: res.session_id,
            question: res.question,
            progress: res.progress,
            done: false,
            safety: false,
          }
          setState(next)
          setInterview({
            sessionId: next.sessionId,
            question: next.question,
            progress: next.progress,
            history: [],
            done: false,
            safety: false,
            draftAnswer: '',
            draftOptionId: null,
          })
        }
      })
      .catch((err) =>
        setError(err instanceof ApiError ? err.message : 'Unable to start the screening interview.'),
      )
    return () => {
      cancelled = true
    }
  }, [setInterview])

  useEffect(() => {
    // Returning to the Text step restores any in-progress interview instead
    // of starting a new one — free navigation must not lose prior answers.
    if (interview) {
      setState({
        sessionId: interview.sessionId,
        question: interview.question,
        progress: interview.progress as AdaptiveProgress,
        done: interview.done,
        safety: interview.safety,
        message: interview.message,
        crisisResources: interview.crisisResources,
      })
      setHistory(interview.history)
      setAnswer(interview.draftAnswer ?? '')
      setSelectedOption(interview.draftOptionId ?? null)
      return
    }
    // Completed interview without a live snapshot (storage written before
    // snapshots existed): show the completed panel — never restart it.
    if (interviewDone && sessionId) {
      setState({
        sessionId,
        question: null,
        progress: {
          questions_answered: 0,
          min_questions: 0,
          max_questions: 0,
          progress_percent: 100,
          required_dimensions_covered: true,
          dimension_coverage: {},
          missing_information: [],
        },
        done: true,
        safety: false,
      })
      return
    }
    return startInterview()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /** Persist each transition so navigating away + back restores this exact
   * state (question, answered history, and the in-progress draft). */
  const persist = useCallback(
    (
      s: InterviewState,
      hist: { q: string; a: string }[],
      draftAnswer: string,
      draftOptionId: string | null,
    ) =>
      setInterview({
        sessionId: s.sessionId,
        question: s.question,
        progress: s.progress,
        history: hist,
        done: s.done,
        safety: s.safety,
        message: s.message,
        crisisResources: s.crisisResources,
        draftAnswer,
        draftOptionId,
      }),
    [setInterview],
  )

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [history.length, state?.question?.id])

  const submit = useCallback(
    async (text: string, optionId?: string) => {
      if (!state || !state.question || submitting) return
      const trimmed = text.trim()
      if (!trimmed && !optionId) return

      setSubmitting(true)
      setError('')
      try {
        const res = await api.screeningResponseOption(state.sessionId, trimmed, optionId)
        const answeredLabel = optionId
          ? (state.question.options.find((o) => o.id === optionId)?.label ?? trimmed)
          : trimmed
        const nextHistory = [...history, { q: state.question!.text, a: answeredLabel }]
        setHistory(nextHistory)
        setAnswer('')
        setSelectedOption(null)

        if (res.safety_concern) {
          const safetyState: InterviewState = {
            ...state,
            done: true,
            safety: true,
            message: res.message,
            crisisResources: res.crisis_resources,
          }
          setState(safetyState)
          persist(safetyState, nextHistory, '', null)
          onSafety(state.sessionId)
          return
        }

        const next: InterviewState = {
          ...state,
          question: res.question,
          progress: res.progress,
          done: res.done,
        }
        setState(next)
        persist(next, nextHistory, '', null)
        if (res.done) {
          onComplete(state.sessionId)
        }
      } catch (err) {
        setError(err instanceof ApiError ? err.message : 'Failed to submit your response.')
      } finally {
        setSubmitting(false)
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [state, submitting, history, onComplete, onSafety, persist],
  )

  if (error && !state) {
    return (
      <div className="rounded-2xl border border-risk-high/30 bg-risk-high/10 p-6 text-sm">
        <p className="font-medium text-risk-high">Unable to start</p>
        <p className="mt-1 text-muted-foreground">{error}</p>
      </div>
    )
  }

  if (!state) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Preparing your interview…
      </div>
    )
  }

  // ---- Safety override view ----
  if (state.safety) {
    return (
      <div className="flex flex-col gap-4 rounded-2xl border border-risk-high/30 bg-risk-high/10 p-6">
        <div className="flex items-center gap-2">
          <ShieldAlert className="size-5 text-risk-high" />
          <h3 className="font-medium text-risk-high">Immediate support recommended</h3>
        </div>
        {state.message ? <p className="text-sm">{state.message}</p> : null}
        {state.crisisResources?.length ? (
          <ul className="flex flex-col gap-2 text-sm">
            {state.crisisResources.map((r) => (
              <li key={r.label}>
                <span className="font-medium">{r.label}:</span>{' '}
                <span className="text-muted-foreground">{r.value}</span>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    )
  }

  // ---- Completed: hand off to result view ----
  // When the interview was already finished (e.g. the user navigated back
  // from Results), show a stable completed panel — never silently restart.
  if (state.done) {
    return (
      <div className="flex flex-col items-start gap-4 rounded-2xl border border-border bg-card p-6">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="size-5 text-risk-low" />
          <p className="font-medium">Interview complete</p>
        </div>
        <p className="text-sm text-muted-foreground">
          Your responses were recorded. You can review your combined result or
          jump to any other assessment — everything stays in place.
        </p>
        <div className="flex flex-wrap gap-2">
          {onViewResults ? (
            <Button size="sm" className="rounded-xl" onClick={onViewResults}>
              View your result
            </Button>
          ) : null}
          <Button
            size="sm"
            variant="outline"
            className="rounded-xl"
            onClick={() => {
              clearInterview()
              setState(null)
              setHistory([])
              setAnswer('')
              setSelectedOption(null)
              setError('')
              startInterview()
            }}
          >
            Start a new interview
          </Button>
        </div>
      </div>
    )
  }

  const progress = state.progress
  const question = state.question
  const hasOptions = Boolean(question?.options?.length)

  return (
    <div className="flex flex-col gap-6">
      {/* Adaptive progress — never a fake "Question N of M" */}
      <div className="flex flex-col gap-2 rounded-2xl border border-border bg-card p-5">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">Questions answered: {progress.questions_answered}</span>
          <span className="text-muted-foreground">Assessment progress: {progress.progress_percent}%</span>
        </div>
        <Progress value={progress.progress_percent} className="w-full" />
        <p className="text-xs text-muted-foreground">
          The interview adapts to your responses — follow-up questions target
          the areas where more information is needed. Typical interviews take{' '}
          {progress.min_questions}–{progress.max_questions} questions.
        </p>
      </div>

      {/* Previous exchanges */}
      {history.length ? (
        <div className="flex flex-col gap-3">
          {history.slice(-4).map((h, i) => (
            <div key={i} className="rounded-xl border border-border bg-muted/40 p-4 text-sm">
              <p className="font-medium">{h.q}</p>
              <p className="mt-1 text-muted-foreground">You: {h.a}</p>
            </div>
          ))}
        </div>
      ) : null}

      {/* Current adaptive question with its OWN option set */}
      {question ? (
        <div className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6">
          <div>
            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Current question
            </span>
            <p className="mt-1 text-lg font-medium">{question.text}</p>
          </div>

          {hasOptions ? (
            <div className="flex flex-col gap-2" role="radiogroup" aria-label="Answer options">
              {question.options.map((opt) => {
                const isSelected = selectedOption === opt.id
                return (
                <button
                  key={opt.id}
                  type="button"
                  role="radio"
                  aria-checked={isSelected}
                  disabled={submitting}
                  onClick={() => {
                    setSelectedOption(opt.id)
                    if (state) persist(state, history, answer, opt.id)
                  }}
                    className={`flex items-start gap-3 rounded-xl border px-4 py-3 text-left text-sm transition-colors disabled:opacity-50 ${
                      isSelected
                        ? 'border-primary bg-primary/10 text-foreground'
                        : 'border-border bg-background hover:border-primary/40 hover:bg-muted/50'
                    }`}
                  >
                    <span
                      className={`mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-full border ${
                        isSelected ? 'border-primary bg-primary' : 'border-muted-foreground/40'
                      }`}
                    >
                      {isSelected ? <span className="size-1.5 rounded-full bg-white" /> : null}
                    </span>
                    <span className="text-pretty">{opt.label}</span>
                  </button>
                )
              })}
            </div>
          ) : null}

          {hasOptions ? (
            <p className="text-xs text-muted-foreground">
              Not exactly right? You can also{' '}
              <button
                type="button"
                className="underline underline-offset-2 hover:text-foreground"
                onClick={() => setSelectedOption(null)}
                disabled={submitting}
              >
                answer in your own words
              </button>
              .
            </p>
          ) : null}

          {!selectedOption || !hasOptions ? (
            <Textarea
              value={answer}
              onChange={(e) => {
                setAnswer(e.target.value)
                // Persist drafts so an in-progress answer survives navigation.
                if (state) persist(state, history, e.target.value, selectedOption)
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                  void submit(answer)
                }
              }}
              placeholder="Answer in your own words — a sentence or two is enough."
              rows={3}
              disabled={submitting}
            />
          ) : null}

          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap gap-1.5">
              {!hasOptions
                ? SAMPLE_RESPONSES.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setAnswer(s)}
                      disabled={submitting}
                      className="rounded-full border border-border px-2.5 py-1 text-xs text-muted-foreground hover:bg-muted disabled:opacity-50"
                    >
                      {s.length > 42 ? `${s.slice(0, 42)}…` : s}
                    </button>
                  ))
                : null}
            </div>
            <Button
              onClick={() => void submit(answer, selectedOption ?? undefined)}
              disabled={submitting || (hasOptions ? !selectedOption && !answer.trim() : !answer.trim())}
              className="gap-1.5 rounded-xl"
            >
              {submitting ? <Loader2 className="size-4 animate-spin" /> : <Send className="size-4" />}
              Send answer
            </Button>
          </div>
        </div>
      ) : null}

      {error ? <p className="text-sm text-risk-high">{error}</p> : null}
      <div ref={bottomRef} />
    </div>
  )
}
