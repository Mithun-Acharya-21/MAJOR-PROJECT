import { cn } from '@/lib/utils'
import { pct, pctNumber } from '@/lib/format'
import { RISK_META } from '@/lib/constants'
import { RiskBadge } from './risk-badge'
import type { RiskLevel } from '@/lib/types'

/** Compact indicator + probability + confidence block used on every modality result. */
export function ResultSummary({
  level,
  probability,
  confidence,
  label = 'Screening indicator',
  className,
}: {
  level: RiskLevel
  probability: number
  confidence: number
  label?: string
  className?: string
}) {
  const token = RISK_META[level].token
  return (
    <div className={cn('flex flex-col gap-5', className)}>
      <div className="flex flex-col gap-2">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
        <RiskBadge level={level} />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Metric label="Probability" value={pct(probability)} pctValue={pctNumber(probability)} token={token} />
        <Metric label="Confidence" value={pct(confidence)} pctValue={pctNumber(confidence)} token="primary" />
      </div>
    </div>
  )
}

function Metric({
  label,
  value,
  pctValue,
  token,
}: {
  label: string
  value: string
  pctValue: number
  token: string
}) {
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-border bg-background/50 p-4">
      <div className="flex items-baseline justify-between">
        <span className="text-sm text-muted-foreground">{label}</span>
        <span className="text-2xl font-semibold tabular-nums">{value}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full transition-[width] duration-700 ease-out"
          style={{ width: `${pctValue}%`, backgroundColor: `var(--${token})` }}
        />
      </div>
    </div>
  )
}
