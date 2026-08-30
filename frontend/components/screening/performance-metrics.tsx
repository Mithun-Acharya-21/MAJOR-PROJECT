'use client'

import { useEffect, useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { api } from '@/lib/api'
import { pct } from '@/lib/format'
import type { ModelMetrics } from '@/lib/types'
import { Collapsible } from '@/components/ui/collapsible'
import { Skeleton } from '@/components/ui/skeleton'

export function PerformanceMetrics() {
  const [metrics, setMetrics] = useState<(ModelMetrics & { source?: string; note?: string }) | null>(
    null,
  )
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showDetails, setShowDetails] = useState(false)

  useEffect(() => {
    api
      .metrics()
      .then(setMetrics)
      .catch(() => setError('Model performance metrics are unavailable.'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex flex-col gap-3">
        <Skeleton className="h-6 w-48" />
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-16 w-full" />
      </div>
    )
  }

  if (error || !metrics) {
    return <p className="text-sm text-muted-foreground">{error || 'Metrics unavailable.'}</p>
  }

  const rows = [
    { label: 'Accuracy', value: metrics.accuracy },
    { label: 'Precision', value: metrics.precision },
    { label: 'Recall', value: metrics.recall },
    { label: 'F1 Score', value: metrics.f1 },
    { label: 'ROC-AUC', value: metrics.rocAuc },
  ]

  return (
    <div className="flex flex-col gap-5">
      <ul className="flex flex-col gap-4">
        {rows.map((row) => (
          <li key={row.label} className="flex flex-col gap-2">
            <div className="flex items-baseline justify-between text-sm">
              <span className="text-muted-foreground">{row.label}</span>
              <span className="font-mono text-lg font-semibold tabular-nums">
                {row.label === 'ROC-AUC' ? row.value.toFixed(2) : pct(row.value)}
              </span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-[width] duration-700"
                style={{
                  width: `${Math.round((row.label === 'ROC-AUC' ? row.value : row.value) * 100)}%`,
                }}
              />
            </div>
          </li>
        ))}
      </ul>

      <Collapsible open={showDetails} onOpenChange={setShowDetails}>
        <button
          type="button"
          onClick={() => setShowDetails(!showDetails)}
          className="inline-flex items-center gap-1.5 text-sm font-medium text-primary hover:underline"
        >
          Technical details
          {showDetails ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
        </button>
        {showDetails ? (
          <div className="mt-3 rounded-xl border border-border bg-muted/30 p-4 text-xs text-muted-foreground">
            <p>
              Metrics source: {metrics.source === 'baselines' ? 'evaluation baselines API' : 'model metrics endpoint'}.
            </p>
            {metrics.note ? <p className="mt-2">{metrics.note}</p> : null}
            <p className="mt-2">
              These values reflect model performance on validation/test datasets and are not
              patient-specific predictions.
            </p>
          </div>
        ) : null}
      </Collapsible>
    </div>
  )
}
