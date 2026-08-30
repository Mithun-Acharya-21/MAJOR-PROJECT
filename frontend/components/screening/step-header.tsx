import type { StepConfig } from '@/lib/constants'

export function StepHeader({ step }: { step: StepConfig }) {
  const Icon = step.icon
  return (
    <div className="flex flex-col items-center gap-3 text-center">
      <span className="flex size-12 items-center justify-center rounded-2xl bg-primary/10 text-primary ring-1 ring-inset ring-primary/15">
        <Icon className="size-6" />
      </span>
      <div className="flex flex-col gap-1.5">
        <span className="font-mono text-xs text-muted-foreground/60">
          {String(step.id).padStart(2, '0')} — {step.label}
        </span>
        <h1 className="text-2xl font-semibold tracking-tight text-balance sm:text-3xl">
          {step.label}
        </h1>
        <p className="text-muted-foreground text-pretty">{step.description}</p>
      </div>
    </div>
  )
}
