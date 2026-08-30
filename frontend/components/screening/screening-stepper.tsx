'use client'

import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'
import { STEPS } from '@/lib/constants'

interface StepperProps {
  /** 1-based active step id. */
  active: number
  /** Set of completed step ids. */
  completed: number[]
  /** Called when a reachable step is clicked. */
  onSelect: (id: number) => void
  /** A step is reachable if it is completed or immediately follows the last completed. */
  isReachable: (id: number) => boolean
}

export function ScreeningStepper({ active, completed, onSelect, isReachable }: StepperProps) {
  const activeStep = STEPS.find((s) => s.id === active) ?? STEPS[0]
  const progress = ((active - 1) / (STEPS.length - 1)) * 100

  return (
    <div>
      {/* Mobile compact progress */}
      <div className="flex flex-col gap-2 md:hidden">
        <div className="flex items-baseline justify-between">
          <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Step {active} of {STEPS.length}
          </span>
          <span className="text-sm font-medium">{activeStep.label}</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary transition-[width] duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Desktop horizontal stepper */}
      <ol className="hidden items-center md:flex">
        {STEPS.map((step, i) => {
          const isDone = completed.includes(step.id)
          const isActive = step.id === active
          const reachable = isReachable(step.id)
          const Icon = step.icon
          return (
            <li key={step.id} className="flex flex-1 items-center last:flex-none">
              <button
                type="button"
                disabled={!reachable}
                onClick={() => reachable && onSelect(step.id)}
                aria-current={isActive ? 'step' : undefined}
                className={cn(
                  'group flex items-center gap-3 rounded-xl px-2 py-1.5 text-left outline-none transition-colors focus-visible:ring-3 focus-visible:ring-ring/50',
                  reachable ? 'cursor-pointer' : 'cursor-not-allowed',
                )}
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
