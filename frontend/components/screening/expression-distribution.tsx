import { pctNumber } from '@/lib/format'
import type { FaceExpression } from '@/lib/types'

const PALETTE = ['chart-1', 'chart-2', 'chart-3', 'chart-4', 'chart-5']

export function ExpressionDistribution({ expressions }: { expressions: FaceExpression[] }) {
  const sorted = [...expressions].sort((a, b) => b.value - a.value)
  return (
    <ul className="flex flex-col gap-3">
      {sorted.map((e, i) => (
        <li key={e.label} className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-sm">
            <span className="capitalize text-foreground/90">{e.label}</span>
            <span className="tabular-nums text-muted-foreground">{pctNumber(e.value)}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full transition-[width] duration-700 ease-out"
              style={{
                width: `${pctNumber(e.value)}%`,
                backgroundColor: `var(--${PALETTE[i % PALETTE.length]})`,
              }}
            />
          </div>
        </li>
      ))}
    </ul>
  )
}
