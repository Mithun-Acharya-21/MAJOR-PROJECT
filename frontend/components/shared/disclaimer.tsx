import { Info } from 'lucide-react'
import { cn } from '@/lib/utils'

export function Disclaimer({ className, compact = false }: { className?: string; compact?: boolean }) {
  return (
    <div
      className={cn(
        'flex gap-3 rounded-2xl border border-border bg-muted/40 p-4',
        className,
      )}
      role="note"
    >
      <Info className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
      <div className="flex flex-col gap-1 text-sm text-muted-foreground">
        <p className="text-pretty">
          <span className="font-medium text-foreground/90">
            EmotionDep is a research and educational screening tool and does not provide a
            medical diagnosis.
          </span>
        </p>
        {!compact ? (
          <p className="text-pretty">
            Results are model-generated screening indicators and should not be interpreted as
            clinical conclusions.
          </p>
        ) : null}
      </div>
    </div>
  )
}
