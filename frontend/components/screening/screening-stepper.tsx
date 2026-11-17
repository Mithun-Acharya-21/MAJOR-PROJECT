'use client'

import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'
import { STEPS } from '@/lib/constants'

interface StepperProps {
  /** 1-based active step id. */
  active: number
  /** Set of completed step ids — an indicator only, never a navigation lock. */
  completed: number[]
  /** Called when a step is clicked. Every step is always selectable. */
  onSelect: (id: number) => void
}

export function ScreeningStepper({ active, completed, onSelect }: StepperProps) {
  const activeStep = STEPS.find((s) => s.id === active) ?? STEPS[0]
  const progress = ((active - 1) / (STEPS.length - 1)) * 100

  return (
    <div>
      {/* Mobile compact progress — same free navigation as desktop: every
          step is always selectable, completion only affects indicators. */}
      <div className="flex flex-col gap-2 md:hidden">
        <div className="flex items-baseline justify-between">
          <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Step {active} of {STEPS.length}
          </span>
          <span className="text-sm font-medium">{activeStep.label}</span>
        </div>
        <ol aria-label="Assessment steps" className="flex gap-1.5 overflow-x-auto pb-1">
          {STEPS.map((step) => {
            const isDone = completed.includes(step.id)
            const isActive = step.id === active
            const Icon = step.icon
            return (
              <li key={step.id} className="shrink-0">
                <button
                  type="button"
                  onClick={() => onSelect(step.id)}
                  aria-current={isActive ? 'step' : undefined}
                  className={cn(
                    'flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-1.5 text-xs outline-none transition-colors focus-visible:ring-3 focus-visible:ring-ring/50',
                    isActive && 'border-primary bg-primary font-medium text-primary-foreground',
                    isDone && !isActive && 'border-primary/30 bg-primary/10 text-primary',
                    !isActive && !isDone && 'border-border bg-card text-muted-foreground',
                  )}
                >
                  {isDone && !isActive ? (
                    <Check className="size-3.5" />
                  ) : (
                    <Icon className="size-3.5" />
                  )}
                  {step.short}
                </button>
              </li>
            )
          })}
        </ol>
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary transition-[width] duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Desktop horizontal stepper */}
      <ol aria-label="Assessment steps" className="hidden items-center md:flex">
        {STEPS.map((step, i) => {
          const isDone = completed.includes(step.id)
          const isActive = step.id === active
          const Icon = step.icon
          return (
            <li key={step.id} className="flex flex-1 items-center last:flex-none">
              <button
                type="button"
                onClick={() => onSelect(step.id)}
                aria-current={isActive ? 'step' : undefined}
                className="group flex cursor-pointer items-center gap-3 rounded-xl px-2 py-1.5 text-left outline-none transition-colors focus-visible:ring-3 focus-visible:ring-ring/50"
              >
                <span
                  className={cn(
                    'flex size-9 shrink-0 items-center justify-center rounded-xl border text-sm font-medium transition-colors',
                    isActive && 'border-primary bg-primary text-primary-foreground',
                    isDone && !isActive && 'border-primary/30 bg-primary/10 text-primary',
                    !isActive &&
                      !isDone &&
                      'border-border bg-card text-muted-foreground',
                  )}
                >
                  {isDone && !isActive ? (
                    <Check className="size-4" />
                  ) : (
                    <Icon className="size-4" />
                  )}
                </span>
                <span className="flex flex-col">
                  <span className="font-mono text-[0.7rem] text-muted-foreground/60">
                    {String(step.id).padStart(2, '0')}
                  </span>
                  <span
                    className={cn(
                      'text-sm font-medium leading-tight',
                      isActive ? 'text-foreground' : 'text-muted-foreground',
                    )}
                  >
                    {step.short}
                  </span>
                </span>
              </button>
              {i < STEPS.length - 1 ? (
                <span
                  className={cn(
                    'mx-2 h-px flex-1 transition-colors',
                    completed.includes(step.id) ? 'bg-primary/40' : 'bg-border',
                  )}
                  aria-hidden
                />
              ) : null}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
