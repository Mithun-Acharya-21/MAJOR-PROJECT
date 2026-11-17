import { MODALITY_META } from '@/lib/constants'
import { pct } from '@/lib/format'
import { RiskBadge } from '@/components/shared/risk-badge'
import type { ModalityKind, RiskLevel } from '@/lib/types'

export function ModalityResultCard({
  modality,
  probability,
  confidence,
  riskLevel,
  dominantExpression,
  modelName,
  modelType,
}: {
  modality: ModalityKind
  probability: number
  confidence: number
  riskLevel: RiskLevel
  dominantExpression?: string
  modelName?: string
  modelType?: string
}) {
  const meta = MODALITY_META[modality]
  const Icon = meta.icon

  const isAuxiliaryExpression = modality === 'photo' || modality === 'face'
  const isAcousticVoice = modality === 'voice'

  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-5">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Icon className="size-4 text-muted-foreground" />
          <span className="font-medium">{meta.label}</span>
        </div>
        <span className="rounded-full border border-border px-2 py-0.5 text-xs text-muted-foreground">
          {isAuxiliaryExpression ? 'Auxiliary Expression' : (isAcousticVoice ? 'Auxiliary Acoustic' : 'Screening Text')}
        </span>
      </div>

      {isAuxiliaryExpression ? (
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Observed Expression</span>
            <span className="rounded-md bg-primary/10 px-2 py-0.5 text-xs font-semibold text-primary">
              {dominantExpression ?? 'Neutral / Flat affect'}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span className="text-xs text-muted-foreground">Model Type</span>
              <p className="font-medium text-xs truncate mt-0.5">{modelType ?? 'MediaPipe Blendshapes'}</p>
            </div>
            <div>
              <span className="text-xs text-muted-foreground">Expression Strength</span>
              <p className="font-mono text-xl font-semibold tabular-nums">{pct(confidence > 0 ? confidence : 0.5)}</p>
            </div>
          </div>
          <p className="text-[11px] text-muted-foreground">
            Auxiliary behavioral signal — NOT a depression probability or clinical diagnosis.
          </p>
        </div>
      ) : (
        <>
          <RiskBadge level={riskLevel} />
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span className="text-muted-foreground">{isAcousticVoice ? 'Acoustic Signal' : 'Probability'}</span>
              <p className="font-mono text-xl font-semibold tabular-nums">{pct(probability)}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Confidence</span>
              <p className="font-mono text-xl font-semibold tabular-nums">{pct(confidence)}</p>
            </div>
          </div>
          {isAcousticVoice ? (
            <p className="text-[11px] text-muted-foreground">
              Acoustic speech markers (pitch, energy, speaking rate) + Whisper STT.
            </p>
          ) : null}
        </>
      )}
    </div>
  )
}
