import { cn } from '@/lib/utils'
import { RISK_META } from '@/lib/constants'
import type { RiskLevel } from '@/lib/types'

export function RiskBadge({
  level,
  className,
}: {
  level: RiskLevel
  className?: string
}) {
  const meta = RISK_META[level]
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium',
        meta.badgeClass,
        className,
      )}
    >
      <span
        className="size-1.5 rounded-full"
        style={{ backgroundColor: `var(--${meta.token})` }}
        aria-hidden
      />
      {meta.label}
    </span>
  )
}
