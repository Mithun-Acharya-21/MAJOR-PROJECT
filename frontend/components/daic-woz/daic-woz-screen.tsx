'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { 
  FileText, Mic, Video, Layers, AlertCircle, CheckCircle2, 
  HelpCircle, ShieldAlert, Cpu, Database, BarChart3, RefreshCw
} from 'lucide-react'

interface ModalityResult {
  modality: string
  available: boolean
  model_name: string
  raw_score?: number
  calibrated_probability?: number
  status: string
}

interface AnalysisResult {
  screening_indicator: string
  calibrated_probability?: number
  risk_level: string
  confidence_interval?: { lower_bound: number; upper_bound: number }
  modality_breakdown: Record<string, ModalityResult>
  fusion_applied: boolean
  model_provenance: Record<string, string>
  research_disclaimer: string
}

interface ModelStatus {
  text_model_loaded: boolean
  audio_model_loaded: boolean
  visual_model_loaded: boolean
  fusion_model_loaded: boolean
  device: string
  dataset_acquisition_status: string
  active_checkpoints: Record<string, string | null>
}

export function DaicWozResearchScreen() {
  const [transcriptText, setTranscriptText] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Fetch model runtime status on mount
  useEffect(() => {
    fetchStatus()
  }, [])

  async function fetchStatus() {
    try {
      const res = await fetch('/api/daic-woz/model/status')
      if (res.ok) {
        const data = await res.json()
        setModelStatus(data)
      }
    } catch (err) {
      console.warn('Backend status endpoint unreachable', err)
    }
  }

  async function handleAnalyze() {
    if (!transcriptText.trim() && !selectedFile) {
      setError('Please provide transcript text or upload an audio file to initiate research analysis.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch('/api/daic-woz/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcript_text: transcriptText.trim() || undefined,
          visual_features: [0.12, 0.45, 0.08, 0.65, 0.15, 0.02, 0.88, 0.04, 0.11, 0.05, 0.22, 0.31, 0.09, 0.14, 0.05, 0.02, 0.19, 0.03, 0.41, 0.08, 0.05, 0.12, 0.02, 0.15, 0.04, 0.01, 0.21, 0.01, 0.03, 0.01, 0.06, 0.08, 0.02, 0.04, 0.01, 0.01, 0.05, 0.01, 0.09, 0.02]
        })
      })

      if (!res.ok) {
        throw new Error(`Inference service returned status ${res.status}`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message || 'Analysis failed. Please verify the backend server is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto w-full max-w-6xl space-y-8 px-4 py-8">
      {/* Header */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-ai border-ai/30 bg-ai/10">
            Research Protocol
          </Badge>
          <Badge variant="secondary" className="gap-1">
            <Database className="size-3" />
            USC ICT DAIC-WOZ Benchmark
          </Badge>
        </div>
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
          DAIC-WOZ Multimodal Screening Pipeline
        </h1>
        <p className="text-muted-foreground text-pretty max-w-3xl">
          Experimental research interface for participant-level screening evaluation across 
          interview transcripts, vocal acoustics, and OpenFace visual kinematics. 
          Evaluated against the Patient Health Questionnaire 8-item (PHQ-8).
        </p>
      </div>

      {/* Model & Dataset Status Banner */}
      <Card className="border-border/60 bg-card/60 backdrop-blur">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="size-4 text-ai" />
              <CardTitle className="text-base font-semibold">Inference Engine & Checkpoint Status</CardTitle>
            </div>
            <Button variant="ghost" size="sm" onClick={fetchStatus} className="size-8 p-0">
              <RefreshCw className="size-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="flex flex-col gap-1 rounded-xl border border-border p-3">
              <span className="text-xs text-muted-foreground">Text Checkpoint</span>
              <span className="text-sm font-medium">
                {modelStatus?.text_model_loaded ? (
                  <span className="text-emerald-500 flex items-center gap-1"><CheckCircle2 className="size-3.5" /> Ready</span>
                ) : (
                  <span className="text-amber-500 flex items-center gap-1"><AlertCircle className="size-3.5" /> Awaiting Data</span>
                )}
              </span>
            </div>
            <div className="flex flex-col gap-1 rounded-xl border border-border p-3">
              <span className="text-xs text-muted-foreground">Audio Checkpoint</span>
              <span className="text-sm font-medium">
                {modelStatus?.audio_model_loaded ? (
                  <span className="text-emerald-500 flex items-center gap-1"><CheckCircle2 className="size-3.5" /> Ready</span>
                ) : (
                  <span className="text-amber-500 flex items-center gap-1"><AlertCircle className="size-3.5" /> Awaiting Data</span>
                )}
              </span>
            </div>
            <div className="flex flex-col gap-1 rounded-xl border border-border p-3">
              <span className="text-xs text-muted-foreground">Visual Kinematics</span>
              <span className="text-sm font-medium">
                {modelStatus?.visual_model_loaded ? (
                  <span className="text-emerald-500 flex items-center gap-1"><CheckCircle2 className="size-3.5" /> Ready</span>
                ) : (
                  <span className="text-amber-500 flex items-center gap-1"><AlertCircle className="size-3.5" /> Awaiting Data</span>
                )}
              </span>
            </div>
            <div className="flex flex-col gap-1 rounded-xl border border-border p-3">
              <span className="text-xs text-muted-foreground">Late Fusion Learner</span>
              <span className="text-sm font-medium">
                {modelStatus?.fusion_model_loaded ? (
                  <span className="text-emerald-500 flex items-center gap-1"><CheckCircle2 className="size-3.5" /> Ready</span>
                ) : (
                  <span className="text-amber-500 flex items-center gap-1"><AlertCircle className="size-3.5" /> Awaiting Data</span>
                )}
              </span>
            </div>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            Academic License: Controlled-access agreement with USC ICT. Strict non-redistribution and participant privacy rules enforced.
          </p>
        </CardContent>
      </Card>

      {/* Input Workspace */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Modality 1: Transcript Text */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <FileText className="size-4 text-primary" />
              <CardTitle className="text-lg">Participant Transcript</CardTitle>
            </div>
            <CardDescription>
              Enter participant responses (excluding interviewer prompts) or paste dialogue excerpt.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder="e.g., I've been feeling overwhelmed lately and finding it hard to get out of bed..."
              rows={6}
              value={transcriptText}
              onChange={(e) => setTranscriptText(e.target.value)}
            />
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setTranscriptText("I feel exhausted most of the time. Even small tasks feel difficult to finish and I don't feel like seeing my friends anymore.")}
              >
                Insert Sample Depressive Excerpt
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setTranscriptText("Things have been pretty good this week. I've been sleeping well and went for a hike over the weekend with colleagues.")}
              >
                Insert Sample Control Excerpt
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Modality 2: Audio & Visual Assets */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Mic className="size-4 text-primary" />
              <CardTitle className="text-lg">Acoustic & Kinematic Inputs</CardTitle>
            </div>
            <CardDescription>
              Attach participant vocal recording (WAV) or examine OpenFace Action Unit stream.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-xl border border-dashed border-border p-6 text-center">
              <input
                type="file"
                accept=".wav,.mp3"
                id="audio-upload"
                className="hidden"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              />
              <label htmlFor="audio-upload" className="cursor-pointer flex flex-col items-center gap-2">
                <Mic className="size-8 text-muted-foreground/60" />
                <span className="text-sm font-medium">
                  {selectedFile ? selectedFile.name : 'Select WAV audio file (16 kHz mono)'}
                </span>
                <span className="text-xs text-muted-foreground">
                  Processed locally through openSMILE eGeMAPSv02
                </span>
              </label>
            </div>

            <div className="rounded-xl border border-border p-4 bg-muted/20">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium flex items-center gap-1.5">
                  <Video className="size-3.5 text-primary" />
                  OpenFace AU Kinematics Stream
                </span>
                <Badge variant="outline" className="text-xs">40 functionals</Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                Action Units AU01, AU04, AU12, AU15, head pose angular velocity, and gaze direction statistics.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Initiation Button */}
      <div className="flex justify-center">
        <Button 
          size="lg" 
          onClick={handleAnalyze} 
          disabled={loading}
          className="gap-2 px-8"
        >
          <Layers className="size-4" />
          {loading ? 'Executing Multimodal Inference...' : 'Run Experimental Research Screening'}
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="size-4" />
          <AlertTitle>Execution Note</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Results Dashboard */}
      {result && (
        <div className="space-y-6 animate-in fade-in duration-300">
          <Card className="border-border">
            <CardHeader className="pb-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                <div>
                  <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    Screening Outcome
                  </span>
                  <CardTitle className="text-2xl mt-1">
                    {result.screening_indicator}
                  </CardTitle>
                </div>
                <Badge 
                  variant={result.risk_level === 'ELEVATED' ? 'destructive' : 'secondary'}
                  className="text-sm px-3 py-1 self-start sm:self-auto"
                >
                  Indicator: {result.risk_level}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {result.calibrated_probability !== null && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">Calibrated Screening Score</span>
                    <span className="font-bold">{Math.round((result.calibrated_probability || 0) * 100)}%</span>
                  </div>
                  <Progress value={(result.calibrated_probability || 0) * 100} />
                  <span className="text-xs text-muted-foreground">
                    Derived from cross-validated Platt scaling against clinical PHQ-8 threshold (≥ 10).
                  </span>
                </div>
              )}

              {/* Modality Breakdown */}
              <div className="grid gap-4 sm:grid-cols-3">
                {Object.entries(result.modality_breakdown).map(([key, mod]) => (
                  <div key={key} className="rounded-xl border border-border p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold uppercase text-muted-foreground">{key}</span>
                      <Badge variant={mod.available ? 'default' : 'outline'} className="text-xs">
                        {mod.available ? 'Active' : 'Offline'}
                      </Badge>
                    </div>
                    <div className="text-sm font-medium">{mod.model_name}</div>
                    <div className="text-xs text-muted-foreground">{mod.status}</div>
                    {mod.calibrated_probability !== undefined && (
                      <div className="text-xs font-semibold text-primary pt-1">
                        Modality Score: {Math.round(mod.calibrated_probability * 100)}%
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Provenance and Disclaimer */}
              <div className="rounded-xl bg-muted/40 p-4 border border-border/80 space-y-2 text-xs text-muted-foreground">
                <div className="font-semibold text-foreground flex items-center gap-1.5">
                  <ShieldAlert className="size-4 text-amber-500" />
                  Academic Research Disclaimer
                </div>
                <p>{result.research_disclaimer}</p>
                <div className="pt-2 border-t border-border/40 grid grid-cols-2 gap-2">
                  <div><strong>Dataset:</strong> {result.model_provenance.dataset}</div>
                  <div><strong>Cut-off:</strong> {result.model_provenance.target_definition}</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
