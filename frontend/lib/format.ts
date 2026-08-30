import type { FeatureContribution } from './types'

/** Format a 0..1 probability/confidence as a whole percent. */
export function pct(value: number | undefined): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${Math.round(value * 100)}%`
}

/** Clamp a 0..1 value to a percentage number for bar widths. */
export function pctNumber(value: number | undefined): number {
  if (value == null || Number.isNaN(value)) return 0
  return Math.max(0, Math.min(100, Math.round(value * 100)))
}

/**
 * Normalize signed feature weights to 0..1 magnitudes for bar rendering,
 * relative to the strongest absolute weight in the set.
 */
export function withMagnitudes(features: FeatureContribution[]): Required<FeatureContribution>[] {
  const max = features.reduce((m, f) => Math.max(m, Math.abs(f.weight)), 0) || 1
  return features.map((f) => ({
    ...f,
    magnitude: f.magnitude ?? Math.abs(f.weight) / max,
  }))
}
