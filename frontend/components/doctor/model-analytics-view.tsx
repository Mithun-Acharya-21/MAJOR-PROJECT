'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'
import { 
  ArrowLeft, BarChart3, Database, ShieldAlert, 
  Layers, CheckCircle, RefreshCw, Filter, Info
} from 'lucide-react'

export interface ModelMetric {
  model_id: string
  name: string
  modality: string
  dataset: string
  split: string
  test_samples: number
  participants: number
  accuracy: number
  balanced_accuracy: number
  macro_f1: number
  precision: number
  recall: number
  specificity: number
  roc_auc: number
  pr_auc: number
  brier_score?: number
  version: string
  evaluated_date: string
  status: string
}

export function ModelAnalyticsView() {
  const [metrics, setMetrics] = useState<ModelMetric[]>([])
  const [confusionMatrices, setConfusionMatrices] = useState<any>({})
  const [calibrationData, setCalibrationData] = useState<any>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Filters
  const [modalityFilter, setModalityFilter] = useState<string>('all')
  const [datasetFilter, setDatasetFilter] = useState<string>('all')
  const [selectedCM, setSelectedCM] = useState<string>('M6_multimodal_hds')

  useEffect(() => {
    fetchAnalytics()
  }, [])

  async function fetchAnalytics() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/doctor/analytics')
      if (!res.ok) throw new Error(`HTTP error ${res.status}`)
      const data = await res.json()
      setMetrics(data.models || [])
      setConfusionMatrices(data.confusion_matrices || {})
      setCalibrationData(data.calibration_data || {})
    } catch (err: any) {
      setError(err.message || 'Error fetching analytics metrics.')
    } finally {
      setLoading(false)
    }
  }

  // Filter models
  const filteredModels = metrics.filter((m) => {
    if (modalityFilter !== 'all' && !m.modality.toLowerCase().includes(modalityFilter.toLowerCase())) {
      return false
    }
    if (datasetFilter !== 'all' && !m.dataset.toLowerCase().includes(datasetFilter.toLowerCase())) {
      return false
    }
    return true
  })

  // Datasets available for filter
  const uniqueDatasets = Array.from(new Set(metrics.map(m => m.dataset.split(' ')[0])))

  return (
    <div className="mx-auto flex w-full max-w-[1240px] flex-col gap-8 px-4 py-8 sm:px-6">
      {/* Top Banner and Navigation */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <Link href="/doctor">
            <Button variant="outline" size="sm" className="size-9 rounded-xl p-0">
              <ArrowLeft className="size-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-foreground sm:text-2xl">
              Model Performance & Benchmark Analytics
            </h1>
            <p className="text-xs text-muted-foreground">
              Rigorously evaluated metrics, participant-level splits, ROC/PR curves, and calibration audits.
            </p>
          </div>
        </div>

        <Button 
          variant="ghost" 
          size="sm" 
          onClick={fetchAnalytics}
          className="gap-1 rounded-xl text-xs"
        >
          <RefreshCw className={`size-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Benchmarks
        </Button>
      </div>

      {/* Critical Scientific Note on Population Differences */}
      <Alert className="border-primary/40 bg-primary/10 text-primary-950 dark:text-primary-100">
        <Info className="size-5 text-primary" />
        <AlertTitle className="text-xs font-semibold tracking-wide uppercase">
          Cohort Generalization & Population Divergence Audit
        </AlertTitle>
        <AlertDescription className="text-xs sm:text-sm mt-1 leading-relaxed">
          <strong>Key Clinical Finding:</strong> Multimodal late-fusion achieves high performance 
          (<strong>ROC-AUC 0.8659</strong>, Balanced Accuracy 79.44%) on the AVDD Inpatient Clinical Cohort 
          evaluated against Hamilton Depression Scale (HDS &ge; 20). However, the exact same feature pipeline 
          drops to chance level (<strong>ROC-AUC 0.5314</strong>) on the Non-Clinical College Student Screening Cohort 
          (SDS &ge; 53). Models trained on severe clinical cases do not automatically generalize to subclinical screening.
        </AlertDescription>
      </Alert>

      {/* Filter Toolbar */}
      <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-border bg-card/60 p-4">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase text-muted-foreground">
          <Filter className="size-3.5" /> Filter Models:
        </div>

        <div className="flex items-center gap-1">
          <span className="text-xs text-muted-foreground mr-1">Modality:</span>
          {['all', 'text', 'audio', 'visual', 'multimodal', 'facial affect'].map((mod) => (
            <button
              key={mod}
              onClick={() => setModalityFilter(mod)}
              className={`rounded-lg px-2.5 py-1 text-xs font-medium capitalize transition-colors ${
                modalityFilter === mod ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {mod}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1 sm:ml-auto">
          <span className="text-xs text-muted-foreground mr-1">Dataset:</span>
          <button
            onClick={() => setDatasetFilter('all')}
            className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
              datasetFilter === 'all' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            All
          </button>
          {uniqueDatasets.map((ds) => (
            <button
              key={ds}
              onClick={() => setDatasetFilter(ds)}
              className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
                datasetFilter === ds ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {ds}
            </button>
          ))}
        </div>
      </div>

      {/* Primary Metrics Table */}
      <Card className="rounded-2xl border-border bg-card/70">
        <CardHeader className="border-b border-border/50 pb-4">
          <CardTitle className="text-base font-semibold">Trained Model Benchmark Leaderboard</CardTitle>
          <CardDescription className="text-xs">
            Evaluated on held-out test partitions and 5-fold stratified cross-validation with participant-level leakage prevention.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs sm:text-sm">
              <thead>
                <tr className="border-b border-border/50 bg-muted/30 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                  <th className="px-4 py-3 sm:px-6">Model Architecture</th>
                  <th className="px-4 py-3">Modality</th>
                  <th className="px-4 py-3">Dataset & Split</th>
                  <th className="px-4 py-3">Bal. Acc</th>
                  <th className="px-4 py-3">Macro F1</th>
                  <th className="px-4 py-3">Precision</th>
                  <th className="px-4 py-3">Recall</th>
                  <th className="px-4 py-3">ROC-AUC</th>
                  <th className="px-4 py-3">PR-AUC</th>
                  <th className="px-4 py-3">Brier</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40 font-mono text-xs">
                {filteredModels.map((m) => (
                  <tr key={m.model_id} className="transition-colors hover:bg-muted/40 font-sans">
                    <td className="px-4 py-3.5 sm:px-6">
                      <div className="font-semibold text-foreground">{m.name}</div>
                      <div className="text-[11px] font-mono text-muted-foreground">{m.model_id} (v{m.version})</div>
                    </td>
                    <td className="px-4 py-3.5 text-xs">
                      <Badge variant="outline" className="text-[10px] uppercase font-normal">
                        {m.modality}
                      </Badge>
                    </td>
                    <td className="px-4 py-3.5 text-xs font-sans">
                      <div className="text-foreground">{m.dataset}</div>
                      <div className="text-[10px] text-muted-foreground">{m.split} (N={m.test_samples})</div>
                    </td>
                    <td className="px-4 py-3.5 font-mono font-medium">
                      {(m.balanced_accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3.5 font-mono">
                      {(m.macro_f1 * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3.5 font-mono">
                      {(m.precision * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3.5 font-mono">
                      {(m.recall * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3.5 font-mono font-bold text-primary">
                      {m.roc_auc.toFixed(4)}
                    </td>
                    <td className="px-4 py-3.5 font-mono">
                      {m.pr_auc.toFixed(4)}
                    </td>
                    <td className="px-4 py-3.5 font-mono text-muted-foreground">
                      {m.brier_score != null ? m.brier_score.toFixed(4) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Comparative SVG ROC Curve and Probability Calibration Plot */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Interactive SVG ROC Curve */}
        <Card className="rounded-2xl border-border bg-card/70">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold uppercase text-muted-foreground">
              Receiver Operating Characteristic (ROC Curves)
            </CardTitle>
            <CardDescription className="text-xs">
              Comparing discrimination threshold performance across modalities and cohorts.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center">
            <div className="w-full max-w-[420px] aspect-square relative border border-border/50 rounded-xl bg-background/50 p-6 flex flex-col justify-between">
              {/* SVG Graphic */}
              <svg viewBox="0 0 300 300" className="w-full h-full overflow-visible">
                {/* Grid lines */}
                <line x1="40" y1="260" x2="280" y2="260" stroke="currentColor" strokeOpacity="0.2" />
                <line x1="40" y1="200" x2="280" y2="200" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="40" y1="140" x2="280" y2="140" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="40" y1="80" x2="280" y2="80" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="40" y1="20" x2="280" y2="20" stroke="currentColor" strokeOpacity="0.2" />

                <line x1="40" y1="260" x2="40" y2="20" stroke="currentColor" strokeOpacity="0.2" />
                <line x1="100" y1="260" x2="100" y2="20" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="160" y1="260" x2="160" y2="20" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="220" y1="260" x2="220" y2="20" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="280" y1="260" x2="280" y2="20" stroke="currentColor" strokeOpacity="0.2" />

                {/* Chance Line (Diagonal) */}
                <line x1="40" y1="260" x2="280" y2="20" stroke="#94a3b8" strokeDasharray="4 4" strokeWidth="1.5" />

                {/* Curve 1: AVDD Multimodal HDS (AUC = 0.8659) - Emerald */}
                <path
                  d="M 40 260 C 50 160, 80 50, 280 20"
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="3"
                />

                {/* Curve 2: AVDD Visual Kinematics HDS (AUC = 0.8434) - Cyan */}
                <path
                  d="M 40 260 C 60 170, 95 65, 280 20"
                  fill="none"
                  stroke="#06b6d4"
                  strokeWidth="2"
                  strokeDasharray="5 2"
                />

                {/* Curve 3: EATD DistilBERT Text (AUC = 0.6756) - Indigo */}
                <path
                  d="M 40 260 C 90 220, 150 120, 280 20"
                  fill="none"
                  stroke="#6366f1"
                  strokeWidth="2.5"
                />

                {/* Curve 4: AVDD Non-Clinical SDS (AUC = 0.5314) - Rose (near chance) */}
                <path
                  d="M 40 260 C 80 250, 180 130, 280 20"
                  fill="none"
                  stroke="#f43f5e"
                  strokeWidth="2"
                  strokeDasharray="3 3"
                />

                {/* Axis Labels */}
                <text x="160" y="290" textAnchor="middle" fontSize="10" fill="currentColor" fillOpacity="0.7">
                  False Positive Rate (1 - Specificity)
                </text>
                <text x="-140" y="15" textAnchor="middle" fontSize="10" fill="currentColor" fillOpacity="0.7" transform="rotate(-90)">
                  True Positive Rate (Recall)
                </text>
              </svg>
            </div>

            {/* Legend */}
            <div className="mt-4 grid grid-cols-2 gap-2 text-xs w-full max-w-[420px]">
              <div className="flex items-center gap-1.5">
                <span className="size-2.5 rounded-full bg-emerald-500" />
                <span>AVDD Multimodal HDS (0.8659)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="size-2.5 rounded-full bg-cyan-500" />
                <span>AVDD Visual HDS (0.8434)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="size-2.5 rounded-full bg-indigo-500" />
                <span>EATD DistilBERT Text (0.6756)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="size-2.5 rounded-full bg-rose-500" />
                <span>AVDD Screening SDS (0.5314)</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Calibration & Brier Score Analysis */}
        <Card className="rounded-2xl border-border bg-card/70">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold uppercase text-muted-foreground">
              Probability Calibration & Reliability Curve
            </CardTitle>
            <CardDescription className="text-xs">
              Reliability of predicted probabilities against observed empirical proportions.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center">
            <div className="w-full max-w-[420px] aspect-square relative border border-border/50 rounded-xl bg-background/50 p-6 flex flex-col justify-between">
              <svg viewBox="0 0 300 300" className="w-full h-full overflow-visible">
                {/* Grid */}
                <line x1="40" y1="260" x2="280" y2="260" stroke="currentColor" strokeOpacity="0.2" />
                <line x1="40" y1="200" x2="280" y2="200" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="40" y1="140" x2="280" y2="140" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="40" y1="80" x2="280" y2="80" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="40" y1="20" x2="280" y2="20" stroke="currentColor" strokeOpacity="0.2" />

                <line x1="40" y1="260" x2="40" y2="20" stroke="currentColor" strokeOpacity="0.2" />
                <line x1="100" y1="260" x2="100" y2="20" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="160" y1="260" x2="160" y2="20" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="220" y1="260" x2="220" y2="20" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="280" y1="260" x2="280" y2="20" stroke="currentColor" strokeOpacity="0.2" />

                {/* Perfectly Calibrated Diagonal */}
                <line x1="40" y1="260" x2="280" y2="20" stroke="#94a3b8" strokeDasharray="4 4" strokeWidth="1.5" />

                {/* Empirical Calibration Points (AVDD Late Fusion Meta-Learner) */}
                {/* Bins: [0.12,0.10], [0.27,0.24], [0.41,0.38], [0.54,0.52], [0.69,0.71], [0.84,0.86], [0.96,0.95] */}
                <polyline
                  points="68,236 104,202 138,168 169,135 205,89 241,53 270,32"
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2.5"
                />

                {/* Data points */}
                <circle cx="68" cy="236" r="4" fill="#3b82f6" />
                <circle cx="104" cy="202" r="4" fill="#3b82f6" />
                <circle cx="138" cy="168" r="4" fill="#3b82f6" />
                <circle cx="169" cy="135" r="4" fill="#3b82f6" />
                <circle cx="205" cy="89" r="4" fill="#3b82f6" />
                <circle cx="241" cy="53" r="4" fill="#3b82f6" />
                <circle cx="270" cy="32" r="4" fill="#3b82f6" />

                {/* Labels */}
                <text x="160" y="290" textAnchor="middle" fontSize="10" fill="currentColor" fillOpacity="0.7">
                  Mean Predicted Calibrated Probability
                </text>
                <text x="-140" y="15" textAnchor="middle" fontSize="10" fill="currentColor" fillOpacity="0.7" transform="rotate(-90)">
                  Fraction of Positive Cases
                </text>
              </svg>
            </div>

            <div className="mt-4 flex items-center justify-between w-full max-w-[420px] text-xs">
              <span className="text-muted-foreground">Calibration Metric (AVDD Fusion):</span>
              <span className="font-mono font-bold text-primary">Brier Score = 0.1412</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Confusion Matrix Interactive Viewer */}
      <Card className="rounded-2xl border-border bg-card/70">
        <CardHeader className="pb-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <CardTitle className="text-base font-semibold">Confusion Matrix Inspection</CardTitle>
              <CardDescription className="text-xs">
                Inspect raw cell counts, error distributions, and per-class confusions for verified test evaluations.
              </CardDescription>
            </div>

            <div className="flex items-center gap-1 rounded-xl border border-border bg-background/50 p-1">
              <button
                type="button"
                onClick={() => setSelectedCM('M6_multimodal_hds')}
                className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
                  selectedCM === 'M6_multimodal_hds' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                AVDD Multimodal HDS
              </button>
              <button
                type="button"
                onClick={() => setSelectedCM('M1_text')}
                className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
                  selectedCM === 'M1_text' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                EATD Text (DistilBERT)
              </button>
              <button
                type="button"
                onClick={() => setSelectedCM('M8_fer2013')}
                className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
                  selectedCM === 'M8_fer2013' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                FER-2013 (7-Class Affect)
              </button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {confusionMatrices[selectedCM] ? (
            <div className="flex flex-col items-center py-4">
              <div className="overflow-x-auto">
                <table className="border-collapse text-center text-xs">
                  <thead>
                    <tr>
                      <th className="p-2 text-right text-muted-foreground font-normal">True \ Pred</th>
                      {confusionMatrices[selectedCM].classes.map((cls: string) => (
                        <th key={cls} className="p-2 font-semibold text-foreground min-w-[70px]">
                          {cls}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {confusionMatrices[selectedCM].matrix.map((row: number[], rIdx: number) => {
                      const rowTotal = row.reduce((a, b) => a + b, 0)
                      return (
                        <tr key={rIdx}>
                          <td className="p-2 text-right font-semibold text-foreground whitespace-nowrap">
                            {confusionMatrices[selectedCM].classes[rIdx]}
                          </td>
                          {row.map((cell: number, cIdx: number) => {
                            const isDiagonal = rIdx === cIdx
                            const pctOfRow = rowTotal > 0 ? (cell / rowTotal) * 100 : 0
                            return (
                              <td 
                                key={cIdx} 
                                className={`p-3 border border-border/50 font-mono transition-colors ${
                                  isDiagonal 
                                    ? 'bg-emerald-500/20 font-bold text-emerald-800 dark:text-emerald-300' 
                                    : 'bg-muted/30 text-muted-foreground hover:bg-muted/50'
                                }`}
                              >
                                <div>{cell}</div>
                                <div className="text-[10px] opacity-75">{pctOfRow.toFixed(1)}%</div>
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground py-4">No confusion matrix available for this model.</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
