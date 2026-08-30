import { MODALITY_META } from '@/lib/constants'
import { pct } from '@/lib/format'
import { RiskBadge } from '@/components/shared/risk-badge'
import type { ModalityKind, RiskLevel } from '@/lib/types'

export function ModalityResultCard({
  modality,
  probability,
  confidence,
  riskLevel,
}: {
  modality: ModalityKind
  probability: number
  confidence: number
  riskLevel: RiskLevel
}) {
  const meta = MODALITY_META[modality]
  const Icon = meta.icon

  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-5">
      <div className="flex items-center gap-2">
        <Icon className="size-4 text-muted-foreground" />
        <span className="font-medium">{meta.label}</span>
      </div>
      <RiskBadge level={riskLevel} />
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <span className="text-muted-foreground">Probability</span>
          <p className="font-mono text-xl font-semibold tabular-nums">{pct(probability)}</p>
        </div>
        <div>
          <span className="text-muted-foreground">Confidence</span>
          <p className="font-mono text-xl font-semibold tabular-nums">{pct(confidence)}</p>
        </div>
      </div>
    </div>
  )
}
