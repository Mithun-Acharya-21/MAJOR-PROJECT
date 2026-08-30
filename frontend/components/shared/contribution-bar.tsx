import { cn } from '@/lib/utils'
import { ArrowUpRight, ArrowDownRight } from 'lucide-react'
import type { FeatureContribution } from '@/lib/types'
import { withMagnitudes } from '@/lib/format'

/**
 * Renders signed feature contributions as horizontal bars.
 * Positive weight = risk-increasing (high token), negative = risk-decreasing (low token).
 */
export function ContributionBars({
  features,
  className,
}: {
  features: FeatureContribution[]
  className?: string
}) {
  const rows = withMagnitudes(features)
  return (
    <ul className={cn('flex flex-col gap-3', className)}>
      {rows.map((f, i) => {
        const increasing = f.weight >= 0
        const token = increasing ? 'risk-high' : 'risk-low'
        return (
          <li key={`${f.label}-${i}`} className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="text-foreground/90 truncate">{f.label}</span>
              <span
                className="inline-flex shrink-0 items-center gap-1 text-xs font-medium"
                style={{ color: `var(--${token})` }}
              >
                {increasing ? (
                  <ArrowUpRight className="size-3" />
                ) : (
                  <ArrowDownRight className="size-3" />
                )}
                {increasing ? 'Risk increasing' : 'Risk decreasing'}
              </span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full transition-[width] duration-700 ease-out"
                style={{
                  width: `${Math.round(f.magnitude * 100)}%`,
                  backgroundColor: `var(--${token})`,
                }}
              />
            </div>
          </li>
        )
      })}
    </ul>
  )
}
