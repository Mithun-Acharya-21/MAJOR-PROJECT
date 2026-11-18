'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'
import { 
  ArrowLeft, Database, ShieldAlert, CheckCircle2, 
  AlertTriangle, Lock, FileSpreadsheet, Cpu, RefreshCw, Info
} from 'lucide-react'

export interface DatasetInfo {
  name: string
  abbreviation: string
  source: string
  license: string
  download_status: string
  format: string
  original_participants: number
  sessions: number
  samples: number
  train_count: number
  validation_count: number
  test_count: number
  class_distribution: any
  modalities: string[]
  label_definition: string
  data_quality_exclusions: string
  clinical_boundary: string
}

export function DatasetInfoView() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchDatasets()
  }, [])

  async function fetchDatasets() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/doctor/datasets')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setDatasets(data.datasets || [])
    } catch (err: any) {
      setError(err.message || 'Error fetching datasets info.')
    } finally {
      setLoading(false)
    }
  }

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
              Dataset Inventory & Model Provenance
            </h1>
            <p className="text-xs text-muted-foreground">
              Clinical research provenance, data governance, participant splits, and algorithmic readiness classification.
            </p>
          </div>
        </div>

        <Button 
          variant="ghost" 
          size="sm" 
          onClick={fetchDatasets}
          className="gap-1 rounded-xl text-xs"
        >
          <RefreshCw className={`size-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Datasets
        </Button>
      </div>

      {/* Model Readiness Taxonomy Overview */}
      <Card className="rounded-2xl border-border bg-card/60">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold uppercase text-muted-foreground">
            NeuroScreen Algorithmic System Architecture
          </CardTitle>
          <CardDescription className="text-xs">
            Disciplined separation between trained statistical classifiers, deterministic feature extractors, and experimental heuristics.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-primary/30 bg-primary/5 p-4 text-xs space-y-2">
              <div className="flex items-center gap-2 font-semibold text-primary">
                <Cpu className="size-4" /> Trained ML / DL Models
              </div>
              <p className="text-muted-foreground text-[11px] leading-relaxed">
                Supervised models with optimized weights saved in <code className="font-mono">models/</code>:
              </p>
              <ul className="list-disc list-inside space-y-1 text-foreground">
                <li>DistilBERT + BiLSTM (EATD Text)</li>
                <li>Wav2Vec2 + LogReg (EATD Voice)</li>
                <li>AVDD Acoustic eGeMAPS LogReg</li>
                <li>AVDD OpenFace Kinematics LogReg</li>
                <li>AVDD Multimodal Late-Fusion Meta-Learner</li>
                <li>ResNet-18 (FER-2013 Affect Classifier)</li>
              </ul>
            </div>

            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-4 text-xs space-y-2">
              <div className="flex items-center gap-2 font-semibold text-emerald-600 dark:text-emerald-400">
                <Database className="size-4" /> Deterministic Feature Extractors
              </div>
              <p className="text-muted-foreground text-[11px] leading-relaxed">
                Standard acoustic and visual feature representations:
              </p>
              <ul className="list-disc list-inside space-y-1 text-foreground">
                <li>openSMILE 3.0 (eGeMAPSv02 88-dim acoustic)</li>
                <li>Librosa YIN Fundamental Frequency ($F_0$)</li>
                <li>OpenFace 2.0 (Action Unit kinematics)</li>
                <li>MediaPipe Face Landmarker (52 blendshapes)</li>
                <li>SciPy acoustic spectral autocorrelation</li>
              </ul>
            </div>

            <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-xs space-y-2">
              <div className="flex items-center gap-2 font-semibold text-amber-600 dark:text-amber-400">
                <ShieldAlert className="size-4" /> Experimental Heuristics
              </div>
              <p className="text-muted-foreground text-[11px] leading-relaxed">
                Interpretable screening rules and decision logic:
              </p>
              <ul className="list-disc list-inside space-y-1 text-foreground">
                <li>Adaptive interview dimension routing</li>
                <li>FACS-inspired prominence & blendshape ratios</li>
                <li>Clinical crisis keyword safety override</li>
                <li>Probability calibration mapping (Brier)</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Dataset Cards */}
      <div className="space-y-6">
        <h2 className="text-base font-semibold text-foreground">Dataset Profiles & Governance</h2>

        {datasets.map((ds) => {
          const isVerified = ds.download_status === 'DOWNLOADED_AND_VERIFIED'
          const isPending = ds.download_status === 'ACCESS_PENDING'
          const isBlocked = ds.clinical_boundary.includes('BLOCKED')

          return (
            <Card key={ds.abbreviation} className="rounded-2xl border-border bg-card/70 overflow-hidden">
              <CardHeader className="border-b border-border/50 pb-4">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <CardTitle className="text-base font-bold text-foreground">
                        {ds.name}
                      </CardTitle>
                      <Badge variant="outline" className="font-mono text-xs">
                        {ds.abbreviation}
                      </Badge>
                    </div>
                    <CardDescription className="text-xs mt-1">
                      Source: {ds.source} • License: {ds.license}
                    </CardDescription>
                  </div>

                  <div>
                    {isVerified ? (
                      <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/30 gap-1 text-xs">
                        <CheckCircle2 className="size-3.5" /> Downloaded & Verified
                      </Badge>
                    ) : isPending ? (
                      <Badge className="bg-blue-500/15 text-blue-600 border-blue-500/30 gap-1 text-xs">
                        <Lock className="size-3.5" /> Official Access Application Required
                      </Badge>
                    ) : (
                      <Badge className="bg-amber-500/15 text-amber-600 border-amber-500/30 gap-1 text-xs">
                        <AlertTriangle className="size-3.5" /> Quarantined
                      </Badge>
                    )}
                  </div>
                </div>
              </CardHeader>

              <CardContent className="pt-4 space-y-4 text-xs">
                {/* Cohort Counts and Format */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="rounded-xl border border-border/50 bg-muted/20 p-3">
                    <span className="text-[10px] uppercase font-semibold text-muted-foreground block">
                      Total Samples / Participants
                    </span>
                    <span className="text-sm font-bold font-mono text-foreground">
                      {ds.samples.toLocaleString()} / {ds.original_participants.toLocaleString()}
                    </span>
                  </div>

                  <div className="rounded-xl border border-border/50 bg-muted/20 p-3">
                    <span className="text-[10px] uppercase font-semibold text-muted-foreground block">
                      Train / Val / Test Split
                    </span>
                    <span className="text-xs font-mono font-medium text-foreground">
                      {ds.train_count} / {ds.validation_count} / {ds.test_count || 'CV-5'}
                    </span>
                  </div>

                  <div className="rounded-xl border border-border/50 bg-muted/20 p-3">
                    <span className="text-[10px] uppercase font-semibold text-muted-foreground block">
                      Ground Truth Scale
                    </span>
                    <span className="text-xs font-medium text-foreground">
                      {ds.label_definition.split(':')[0]}
                    </span>
                  </div>

                  <div className="rounded-xl border border-border/50 bg-muted/20 p-3">
                    <span className="text-[10px] uppercase font-semibold text-muted-foreground block">
                      Available Modalities
                    </span>
                    <div className="flex flex-wrap gap-1 mt-0.5">
                      {ds.modalities.map((m) => (
                        <Badge key={m} variant="outline" className="px-1 py-0 text-[9px]">
                          {m.split(' ')[0]}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Ground Truth and Exclusions Details */}
                <div className="rounded-xl border border-border/60 bg-muted/30 p-3.5 space-y-2">
                  <div>
                    <span className="font-semibold text-foreground">Label Definition: </span>
                    <span className="text-muted-foreground">{ds.label_definition}</span>
                  </div>
                  <div>
                    <span className="font-semibold text-foreground">Data Format & Exclusions: </span>
                    <span className="text-muted-foreground">{ds.format}. {ds.data_quality_exclusions}</span>
                  </div>
                  <div>
                    <span className="font-semibold text-foreground">Clinical Generalization Boundary: </span>
                    <span className={isBlocked ? 'text-rose-600 dark:text-rose-400 font-medium' : 'text-muted-foreground'}>
                      {ds.clinical_boundary}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
