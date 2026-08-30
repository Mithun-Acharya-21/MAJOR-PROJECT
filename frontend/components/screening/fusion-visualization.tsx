import { MODALITY_META } from '@/lib/constants'
import { pct } from '@/lib/format'
import type { ModalityWeight } from '@/lib/types'

export function FusionVisualization({ weights }: { weights: ModalityWeight[] }) {
  if (!weights.length) {
    return (
      <p className="text-sm text-muted-foreground">
        Individual modality contribution data is unavailable.
      </p>
    )
  }

  const sorted = [...weights].sort((a, b) => b.weight - a.weight)

  return (
    <div className="flex flex-col gap-6">
      {/* ASCII-style flow diagram */}
      <div className="hidden rounded-xl border border-border bg-muted/30 p-4 font-mono text-xs text-muted-foreground sm:block">
        <pre className="overflow-x-auto whitespace-pre">{`Text ─────────┐
              │
Photo ────────┤
              ├──→ Multimodal Fusion → Final Result
Face ─────────┤
              │
Voice ────────┘`}</pre>
      </div>

      <ul className="flex flex-col gap-4">
        {sorted.map((w) => {
          const meta = MODALITY_META[w.modality]
          const Icon = meta.icon
          const pctWeight = Math.round(w.weight * 100)
          return (
            <li key={w.modality} className="flex flex-col gap-2">
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="inline-flex items-center gap-2 font-medium">
                  <Icon className="size-4 text-muted-foreground" />
                  {meta.label}
                </span>
                <span className="font-mono tabular-nums text-muted-foreground">{pctWeight}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary transition-[width] duration-700"
                  style={{ width: `${pctWeight}%` }}
                />
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
