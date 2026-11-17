'use client'

import { useRef, useState } from 'react'
import Image from 'next/image'
import { Upload, ImageIcon, CheckCircle2, XCircle, Sparkles, Info, AlertTriangle, Activity } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { api, ApiError } from '@/lib/api'
import { useScreening } from '@/lib/screening-store'
import { ExpressionDistribution } from './expression-distribution'
import { StageChecklist, ErrorState, type Stage } from '@/components/shared/status-blocks'
import { LimeView } from './lime-view'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { StepFooter } from './step-footer'

const STAGES: Stage[] = [
  { label: 'Image received', state: 'done' },
  { label: 'Detecting face', state: 'active' },
  { label: 'Running model', state: 'pending' },
  { label: 'Generating explanation', state: 'pending' },
]

export function PhotoAnalysis({ onBack, onNext }: { onBack: () => void; onNext: () => void }) {
  const { photo, setPhoto } = useScreening()
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle')
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)

  function selectFile(f: File | undefined) {
    if (!f) return
    if (!f.type.startsWith('image/')) {
      setError('Please choose a JPG or PNG image.')
      setStatus('error')
      return
    }
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setStatus('idle')
    setError('')
  }

  async function analyze() {
    if (!file) return
    setStatus('loading')
    setError('')
    try {
      const result = await api.analyzePhoto(file)
      setPhoto(result, file)
      setStatus('idle')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Unable to analyze the photo.')
      setStatus('error')
    }
  }

  const noFace = photo != null && !photo.faceDetected

  return (
    <div className="flex flex-col gap-6">
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg"
        className="sr-only"
        onChange={(e) => selectFile(e.target.files?.[0])}
      />

      {!preview ? (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            selectFile(e.dataTransfer.files?.[0])
          }}
          className={`flex flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed bg-card px-6 py-16 text-center transition-colors ${
            dragging ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/40'
          }`}
        >
          <span className="flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <Upload className="size-6" />
          </span>
          <div className="flex flex-col gap-1">
            <span className="font-medium">Upload a facial photo</span>
            <span className="text-sm text-muted-foreground">JPG, PNG up to 10MB</span>
          </div>
          <span className="rounded-xl bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground">
            Choose image
          </span>
        </button>
      ) : (
        <div className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-5 sm:flex-row sm:items-center">
          <div className="relative size-32 shrink-0 overflow-hidden rounded-xl border border-border bg-muted">
            <Image src={preview} alt="Uploaded photo preview" fill className="object-cover" unoptimized />
          </div>
          <div className="flex flex-1 flex-col gap-2">
            <div className="flex items-center gap-2 text-sm">
              <ImageIcon className="size-4 text-muted-foreground" />
              <span className="truncate font-medium">{file?.name ?? 'photo.jpg'}</span>
            </div>
            {photo?.faceDetected ? (
              <span className="inline-flex items-center gap-1.5 text-sm text-risk-low">
                <CheckCircle2 className="size-4" />
                Face detected
              </span>
            ) : noFace ? (
              <span className="inline-flex items-center gap-1.5 text-sm text-risk-high">
                <XCircle className="size-4" />
                Face not detected
              </span>
            ) : null}
            <div className="mt-1 flex flex-wrap gap-2">
              <Button variant="outline" size="sm" onClick={() => inputRef.current?.click()}>
                Change photo
              </Button>
              {!photo || noFace ? (
                <Button size="sm" onClick={analyze} disabled={status === 'loading'} className="gap-1.5">
                  <Sparkles className="size-3.5" />
                  {status === 'loading' ? 'Analyzing...' : 'Analyze photo'}
                </Button>
              ) : null}
            </div>
          </div>
        </div>
      )}

      {status === 'loading' ? (
        <div className="rounded-2xl border border-border bg-card p-6">
          <StageChecklist title="Analyzing your photo..." stages={STAGES} />
        </div>
      ) : null}

      {status === 'error' ? <ErrorState message={error} onRetry={file ? analyze : undefined} /> : null}

      {noFace && status === 'idle' ? (
        <div className="flex flex-col items-center gap-3 rounded-2xl border border-risk-high/20 bg-risk-high/5 px-6 py-8 text-center">
          <XCircle className="size-8 text-risk-high" />
          <div className="flex flex-col gap-1">
            <p className="font-medium">Face not detected</p>
            <p className="max-w-sm text-sm text-muted-foreground text-pretty">
              Please upload a clear image containing one visible face.
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={() => inputRef.current?.click()}>
            Choose another image
          </Button>
        </div>
      ) : null}

      {photo?.faceDetected && status === 'idle' ? (
        <div className="animate-in fade-in slide-in-from-bottom-3 flex flex-col gap-6 rounded-2xl border border-border bg-card p-5 duration-500 sm:p-6">
          {/* Header & Status */}
          <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-lg font-medium">Photo Facial-Expression Analysis</h2>
              <p className="text-xs text-muted-foreground">Auxiliary behavioral signal — not a depression diagnosis</p>
            </div>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-risk-low/30 bg-risk-low/10 px-3 py-1 text-xs font-medium text-risk-low">
              <CheckCircle2 className="size-3.5" />
              Face Detected
            </span>
          </div>

          {/* Model Information Card */}
          <div className="grid gap-3 rounded-xl border border-border bg-background/50 p-4 text-xs sm:grid-cols-3">
            <div>
              <span className="text-muted-foreground">Model Name</span>
              <p className="mt-0.5 font-medium">{photo.modelName ?? 'MediaPipe FaceLandmarker'}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Model Type</span>
              <p className="mt-0.5 font-medium">{photo.modelType ?? 'Facial Landmark & Expression Detection'}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Processing Status</span>
              <p className="mt-0.5 font-medium">{photo.processingStatus ?? 'Face analyzed successfully'}</p>
            </div>
          </div>

          {/* Dominant Expression & Uncertainty Classification */}
          <div className="flex flex-col gap-4 rounded-xl border border-border bg-background/50 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-col">
                <span className="text-xs text-muted-foreground">Observed Expression Pattern</span>
                <span className="text-base font-semibold">{photo.dominantExpression ?? 'Neutral'}</span>
              </div>
              <div className="flex items-center gap-2">
                {photo.uncertaintyLabel ? (
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium ${
                      photo.uncertaintyLabel.includes('Mixed') ||
                      photo.uncertaintyLabel.includes('Conflicting') ||
                      photo.uncertaintyLabel.includes('Grimace')
                        ? 'border-amber-500/30 bg-amber-500/10 text-amber-500'
                        : photo.uncertaintyLabel.includes('Resting')
                        ? 'border-muted-foreground/30 bg-muted/40 text-muted-foreground'
                        : 'border-primary/30 bg-primary/10 text-primary'
                    }`}
                  >
                    {photo.uncertaintyLabel.includes('Conflicting') || photo.uncertaintyLabel.includes('Mixed') ? (
                      <AlertTriangle className="size-3" />
                    ) : (
                      <Activity className="size-3" />
                    )}
                    {photo.uncertaintyLabel}
                  </span>
                ) : null}
              </div>
            </div>

            {photo.confidence > 0 ? (
              <div className="flex flex-col gap-1.5 pt-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Raw Muscle Activation Prominence</span>
                  <span className="font-mono font-medium tabular-nums">{Math.round(photo.confidence * 100)}%</span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-primary/70 transition-[width] duration-700"
                    style={{ width: `${Math.round(photo.confidence * 100)}%` }}
                  />
                </div>
                <span className="text-[11px] text-muted-foreground">
                  Continuous Action Unit intensity for the dominant pattern (uninflated physical activation level).
                </span>
              </div>
            ) : null}
          </div>

          {/* Observed Facial Movement Indicators (Action Units) */}
          {photo.observedMovement && Object.keys(photo.observedMovement).length > 0 ? (
            <div className="flex flex-col gap-3 rounded-xl border border-border/80 bg-background/30 p-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  FACS Movement Indicators (Key Action Units)
                </span>
                <span className="text-[11px] text-muted-foreground">MediaPipe 52 Blendshapes</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-3 md:grid-cols-4">
                {Object.entries(photo.observedMovement).map(([key, value]) => {
                  const displayLabel = key
                    .split('_')
                    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
                    .join(' ')
                  const pct = Math.round(Number(value) * 100)
                  return (
                    <div
                      key={key}
                      className="flex flex-col gap-1 rounded-lg border border-border/50 bg-background/60 p-2"
                    >
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="truncate text-muted-foreground">{displayLabel}</span>
                        <span className="font-mono font-medium tabular-nums">{pct}%</span>
                      </div>
                      <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-primary/60 transition-[width] duration-500"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ) : null}

          {/* Full Expression Distribution */}
          {photo.expressions && photo.expressions.length > 0 ? (
            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Relative Expression Share</span>
                <span className="text-xs text-muted-foreground">Normalized across 7 affect categories (Sums to 100%)</span>
              </div>
              <ExpressionDistribution expressions={photo.expressions} />
            </div>
          ) : null}

          {/* Scientific Honesty Disclaimer Banner */}
          <div className="flex gap-3 rounded-xl border border-border/70 bg-muted/30 p-4 text-xs text-muted-foreground">
            <Info className="size-4 shrink-0 text-primary mt-0.5" />
            <div className="flex flex-col gap-1">
              <span className="font-medium text-foreground">Scientific Integrity Notice</span>
              <p className="leading-relaxed">
                Facial expression analysis measures momentary facial muscle activations and affect.
                {' '}<strong>It is not a clinical depression diagnosis or screening probability.</strong>{' '}
                Emotional expressions fluctuate naturally based on social context, lighting, and camera angle.
                In accordance with research protocol, this project does not calculate depression probability from facial expressions.
              </p>
            </div>
          </div>

          {photo.lime ? (
            <div className="border-t border-border pt-5">
              {photo.lime.imageOverlayUrl ? (
                <Tabs defaultValue="explanation">
                  <TabsList>
                    <TabsTrigger value="original">Original</TabsTrigger>
                    <TabsTrigger value="explanation">Explanation</TabsTrigger>
                  </TabsList>
                  <TabsContent value="original" className="pt-4">
                    {preview ? (
                      <div className="relative mx-auto aspect-square max-w-xs overflow-hidden rounded-xl border border-border">
                        <Image src={preview} alt="Original photo" fill className="object-cover" unoptimized />
                      </div>
                    ) : null}
                  </TabsContent>
                  <TabsContent value="explanation" className="pt-4">
                    <LimeView explanation={photo.lime} />
                  </TabsContent>
                </Tabs>
              ) : (
                <LimeView explanation={photo.lime} />
              )}
            </div>
          ) : null}
        </div>
      ) : null}

      <StepFooter onBack={onBack} onNext={onNext} nextLabel="Continue to live face" />
    </div>
  )
}
