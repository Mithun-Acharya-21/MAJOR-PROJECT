'use client'

import { useRef, useState } from 'react'
import Image from 'next/image'
import { Upload, ImageIcon, CheckCircle2, XCircle, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { api, ApiError } from '@/lib/api'
import { useScreening } from '@/lib/screening-store'
import { ResultSummary } from '@/components/shared/result-summary'
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
          <h2 className="text-lg font-medium">Photo analysis</h2>
          <ResultSummary
            level={photo.riskLevel}
            probability={photo.probability}
            confidence={photo.confidence}
          />
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

      <StepFooter
        onBack={onBack}
        onNext={onNext}
        nextDisabled={!photo}
        nextLabel="Continue to live face"
      />
    </div>
  )
}
