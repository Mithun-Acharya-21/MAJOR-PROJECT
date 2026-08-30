import Image from 'next/image'
import { Info } from 'lucide-react'
import { ContributionBars } from '@/components/shared/contribution-bar'
import type { LimeExplanation } from '@/lib/types'

/** Shared renderer for a LIME explanation: feature bars plus an optional image overlay. */
export function LimeView({ explanation }: { explanation: LimeExplanation }) {
  const hasImage = explanation.imageOriginalUrl && explanation.imageOverlayUrl
  return (
    <div className="flex flex-col gap-5">
      {hasImage ? (
        <div className="grid grid-cols-2 gap-3">
          <figure className="flex flex-col gap-1.5">
            <div className="relative aspect-square overflow-hidden rounded-xl border border-border bg-muted">
              <Image
                src={explanation.imageOriginalUrl as string}
                alt="Original input image"
                fill
                className="object-cover"
                unoptimized
              />
            </div>
            <figcaption className="text-center text-xs text-muted-foreground">Original</figcaption>
          </figure>
          <figure className="flex flex-col gap-1.5">
            <div className="relative aspect-square overflow-hidden rounded-xl border border-border bg-muted">
              <Image
                src={explanation.imageOverlayUrl as string}
                alt="Explanation overlay highlighting influential regions"
                fill
                className="object-cover"
                unoptimized
              />
            </div>
            <figcaption className="text-center text-xs text-muted-foreground">Explanation</figcaption>
          </figure>
        </div>
      ) : null}

      {hasImage ? (
        <p className="text-sm text-muted-foreground text-pretty">
          Highlighted regions indicate areas that influenced the model prediction.
        </p>
      ) : null}

      {explanation.features?.length ? <ContributionBars features={explanation.features} /> : null}

      <div className="flex gap-2 rounded-xl bg-muted/50 p-3 text-xs text-muted-foreground">
        <Info className="mt-0.5 size-3.5 shrink-0" />
        <p className="text-pretty">
          Positive or negative influence indicates how a feature affected the model prediction. It
          does not indicate a clinical cause.
        </p>
      </div>
    </div>
  )
}
