'use client'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { AlertTriangle, Check, Loader2, Circle } from 'lucide-react'

export type StageState = 'pending' | 'active' | 'done'

export interface Stage {
  label: string
  state: StageState
}

/** A vertical checklist that communicates multi-stage processing without freezing the UI. */
export function StageChecklist({
  title,
  stages,
  className,
}: {
  title?: string
  stages: Stage[]
  className?: string
}) {
  return (
    <div className={cn('flex flex-col gap-3', className)} role="status" aria-live="polite">
      {title ? <p className="text-sm font-medium text-foreground">{title}</p> : null}
      <ul className="flex flex-col gap-2.5">
        {stages.map((s) => (
          <li key={s.label} className="flex items-center gap-2.5 text-sm">
            {s.state === 'done' ? (
              <span className="flex size-5 items-center justify-center rounded-full bg-risk-low/15 text-risk-low">
                <Check className="size-3" />
              </span>
            ) : s.state === 'active' ? (
              <span className="flex size-5 items-center justify-center rounded-full bg-primary/15 text-primary">
                <Loader2 className="size-3 animate-spin" />
              </span>
            ) : (
              <span className="flex size-5 items-center justify-center text-muted-foreground/50">
                <Circle className="size-3" />
              </span>
            )}
            <span
              className={cn(
                s.state === 'pending' ? 'text-muted-foreground' : 'text-foreground/90',
              )}
            >
              {s.label}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** Polished inline error with retry, never exposing raw stack traces. */
export function ErrorState({
  title = 'Something went wrong',
  message,
  onRetry,
  className,
}: {
  title?: string
  message: string
  onRetry?: () => void
  className?: string
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center gap-3 rounded-2xl border border-destructive/20 bg-destructive/5 px-6 py-8 text-center',
        className,
      )}
      role="alert"
    >
      <span className="flex size-10 items-center justify-center rounded-full bg-destructive/10 text-destructive">
        <AlertTriangle className="size-5" />
      </span>
      <div className="flex flex-col gap-1">
        <p className="font-medium text-foreground">{title}</p>
        <p className="max-w-sm text-sm text-muted-foreground text-pretty">{message}</p>
      </div>
      {onRetry ? (
        <Button variant="outline" size="sm" onClick={onRetry} className="mt-1">
          Try again
        </Button>
      ) : null}
    </div>
  )
}

/** Neutral empty state for modalities that were skipped or produced no result. */
export function EmptyState({
  title,
  message,
  className,
}: {
  title: string
  message: string
  className?: string
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center gap-2 rounded-2xl border border-dashed border-border bg-muted/30 px-6 py-8 text-center',
        className,
      )}
    >
      <p className="font-medium text-foreground">{title}</p>
      <p className="max-w-sm text-sm text-muted-foreground text-pretty">{message}</p>
    </div>
  )
}
