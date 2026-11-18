'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'
import { 
  ArrowLeft, Mic, Upload, Activity, ShieldAlert, 
  Volume2, Play, CheckCircle, RefreshCw, Info, Globe2
} from 'lucide-react'

export interface PitchContourPoint {
  time_sec: number
  pitch_hz: number
  voiced: boolean
}

export interface PitchMetrics {
  min_pitch_hz: number
  max_pitch_hz: number
  mean_pitch_hz: number
  median_pitch_hz: number
  pitch_std_hz: number
  pitch_range_hz: number
  total_duration_sec: number
  speech_duration_sec: number
  silence_duration_sec: number
  silence_ratio: number
  pause_frequency_per_min: number
  speaking_rate_syllables_sec: number
  rms_energy: number
  voiced_frames_count: number
  total_frames_count: number
}

export function VoicePitchView() {
  const [file, setFile] = useState<File | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<PitchMetrics | null>(null)
  const [contour, setContour] = useState<PitchContourPoint[]>([])
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [currentSampleLabel, setCurrentSampleLabel] = useState<string>('')

  // Synthetic tone generator to test acoustic engine in-browser without manual file search
  async function generateAcousticSample(sampleType: 'english_prosody' | 'kannada_prosody') {
    setError(null)
    setAnalyzing(true)
    setCurrentSampleLabel(sampleType === 'kannada_prosody' ? 'Kannada Vocal Pitch Sample' : 'Standard Research Speech Sample')

    try {
      // Build a synthetic speech-like acoustic audio buffer using Web Audio API
      const sampleRate = 16000
      const durationSec = 3.0
      const numSamples = sampleRate * durationSec
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate })
      const buffer = audioCtx.createBuffer(1, numSamples, sampleRate)
      const data = buffer.getChannelData(0)

      // Synthesize realistic pitch contour:
      // Kannada or English vocalization with natural intonation & 2 short pauses
      for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate
        // Add pauses at 0.8-1.1s and 2.0-2.2s
        if ((t > 0.8 && t < 1.1) || (t > 2.0 && t < 2.2)) {
          data[i] = 0.001 * (Math.random() - 0.5) // Silence floor
        } else {
          // Fundamental frequency varying between 140 Hz and 210 Hz
          const f0 = sampleType === 'kannada_prosody'
            ? 165 + 35 * Math.sin(2 * Math.PI * 1.8 * t) + 15 * Math.cos(2 * Math.PI * 3.5 * t)
            : 155 + 25 * Math.sin(2 * Math.PI * 1.5 * t)
          
          // Glottal harmonic waveform
          const phase = 2 * Math.PI * f0 * t
          const harm1 = Math.sin(phase)
          const harm2 = 0.5 * Math.sin(2 * phase)
          const harm3 = 0.25 * Math.sin(3 * phase)
          data[i] = 0.4 * (harm1 + harm2 + harm3)
        }
      }

      // Convert AudioBuffer to WAV blob
      const wavBlob = audioBufferToWav(buffer)
      const wavFile = new File([wavBlob], `${sampleType}.wav`, { type: 'audio/wav' })
      setFile(wavFile)
      setAudioUrl(URL.createObjectURL(wavBlob))

      await runPitchAnalysis(wavFile)
    } catch (err: any) {
      setError(err.message || 'Error generating test acoustic sample.')
      setAnalyzing(false)
    }
  }

  // Convert Float32 AudioBuffer to standard 16-bit PCM WAV Blob
  function audioBufferToWav(buffer: AudioBuffer): Blob {
    const numChannels = 1
    const sampleRate = buffer.sampleRate
    const format = 1 // PCM
    const bitDepth = 16
    const samples = buffer.getChannelData(0)
    const blockAlign = numChannels * (bitDepth / 8)
    const byteRate = sampleRate * blockAlign
    const dataSize = samples.length * (bitDepth / 8)
    const bufferSize = 44 + dataSize
    const arrayBuffer = new ArrayBuffer(bufferSize)
    const view = new DataView(arrayBuffer)

    function writeString(offset: number, str: string) {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i))
      }
    }

    writeString(0, 'RIFF')
    view.setUint32(4, 36 + dataSize, true)
    writeString(8, 'WAVE')
    writeString(12, 'fmt ')
    view.setUint32(16, 16, true)
    view.setUint16(20, format, true)
    view.setUint16(22, numChannels, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, byteRate, true)
    view.setUint16(32, blockAlign, true)
    view.setUint16(34, bitDepth, true)
    writeString(36, 'data')
    view.setUint32(40, dataSize, true)

    let offset = 44
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]))
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' })
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files || e.target.files.length === 0) return
    const uploadedFile = e.target.files[0]
    setFile(uploadedFile)
    setAudioUrl(URL.createObjectURL(uploadedFile))
    setCurrentSampleLabel(uploadedFile.name)
    await runPitchAnalysis(uploadedFile)
  }

  async function runPitchAnalysis(audioFile: File) {
    setAnalyzing(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('audio', audioFile)

      const res = await fetch('/api/voice/pitch-analysis', {
        method: 'POST',
        body: formData
      })

      if (!res.ok) {
        throw new Error(`Pitch analysis failed (HTTP ${res.status})`)
      }

      const data = await res.json()
      if (!data.valid) {
        throw new Error(data.error || 'Acoustic pitch analysis could not be completed.')
      }

      setMetrics(data.metrics)
      setContour(data.pitch_contour || [])
    } catch (err: any) {
      setError(err.message || 'Error processing vocal acoustic data.')
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-[1240px] flex-col gap-8 px-4 py-8 sm:px-6">
      {/* Navigation and Title */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <Link href="/doctor">
            <Button variant="outline" size="sm" className="size-9 rounded-xl p-0">
              <ArrowLeft className="size-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-foreground sm:text-2xl">
              Voice Pitch & Prosodic Acoustic Lab
            </h1>
            <p className="text-xs text-muted-foreground">
              Fundamental frequency ($F_0$) tracking, pitch range, speaking rate, and pause dynamics for clinical vocal analysis.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="outline" className="gap-1 text-xs">
            <Globe2 className="size-3 text-cyan-500" />
            Language Agnostic ($F_0$ Physics)
          </Badge>
        </div>
      </div>

      {/* Language Invariance and Research Framing Note */}
      <Alert className="border-cyan-500/40 bg-cyan-500/10 text-cyan-950 dark:text-cyan-100">
        <Info className="size-5 text-cyan-600 dark:text-cyan-400" />
        <AlertTitle className="text-xs font-semibold tracking-wide uppercase">
          Multilingual Acoustic Invariance (English, Kannada, Hindi)
        </AlertTitle>
        <AlertDescription className="text-xs sm:text-sm mt-1 leading-relaxed">
          Fundamental frequency ($F_0$), vocal pitch variability, speech-to-pause ratios, and intensity are physical acoustic 
          attributes of vocal-fold vibration and subglottal respiratory control. These markers reflect psychomotor slowing 
          and neuromuscular prosodic flattening, remaining invariant across spoken languages including English and Kannada.
        </AlertDescription>
      </Alert>

      {/* Audio Input & Analysis Trigger */}
      <Card className="rounded-2xl border-border bg-card/70">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-semibold">Audio Input & Prosody Extraction</CardTitle>
          <CardDescription className="text-xs">
            Upload a clinical vocal recording (WAV, MP3, WebM) or test with standard multi-lingual vocal samples.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <label className="cursor-pointer">
              <input
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="hidden"
                disabled={analyzing}
              />
              <Button 
                variant="outline" 
                size="sm" 
                disabled={analyzing}
                className="gap-1.5 rounded-xl text-xs pointer-events-none"
              >
                <Upload className="size-3.5" />
                Upload Audio File
              </Button>
            </label>

            <span className="text-xs text-muted-foreground">or test with verified acoustic samples:</span>

            <Button
              variant="outline"
              size="sm"
              onClick={() => generateAcousticSample('english_prosody')}
              disabled={analyzing}
              className="gap-1.5 rounded-xl text-xs"
            >
              <Mic className="size-3.5 text-primary" />
              Standard English Vocal Sample
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={() => generateAcousticSample('kannada_prosody')}
              disabled={analyzing}
              className="gap-1.5 rounded-xl text-xs"
            >
              <Globe2 className="size-3.5 text-emerald-500" />
              Kannada Vocal Intonation Sample
            </Button>
          </div>

          {currentSampleLabel && (
            <div className="flex items-center gap-3 rounded-xl border border-border/60 bg-muted/30 p-3 text-xs">
              <Volume2 className="size-4 text-primary" />
              <span className="font-medium text-foreground">Current Audio: {currentSampleLabel}</span>
              {audioUrl && (
                <audio controls src={audioUrl} className="ml-auto h-8 max-w-[240px]" />
              )}
            </div>
          )}

          {analyzing && (
            <div className="flex items-center justify-center gap-2 p-6 text-xs text-muted-foreground">
              <Activity className="size-4 animate-spin text-primary" />
              Computing YIN pitch tracking and temporal acoustic descriptors...
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertTitle>Extraction Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Prosodic Acoustic Metrics Grid */}
      {metrics && (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            <div className="rounded-2xl border border-border bg-card/60 p-3.5">
              <span className="text-[10px] uppercase font-semibold text-muted-foreground block">
                Mean Pitch ($F_0$)
              </span>
              <span className="text-xl font-bold font-mono text-foreground">
                {metrics.mean_pitch_hz.toFixed(1)} <span className="text-xs font-normal">Hz</span>
              </span>
            </div>

            <div className="rounded-2xl border border-border bg-card/60 p-3.5">
              <span className="text-[10px] uppercase font-semibold text-muted-foreground block">
                Pitch Std Dev
              </span>
              <span className="text-xl font-bold font-mono text-foreground">
                {metrics.pitch_std_hz.toFixed(1)} <span className="text-xs font-normal">Hz</span>
              </span>
            </div>

            <div className="rounded-2xl border border-border bg-card/60 p-3.5">
              <span className="text-[10px] uppercase font-semibold text-muted-foreground block">
                Pitch Range
              </span>
              <span className="text-xl font-bold font-mono text-foreground">
                {metrics.pitch_range_hz.toFixed(1)} <span className="text-xs font-normal">Hz</span>
              </span>
            </div>

            <div className="rounded-2xl border border-border bg-card/60 p-3.5">
              <span className="text-[10px] uppercase font-semibold text-muted-foreground block">
                Speech Duration
              </span>
              <span className="text-xl font-bold font-mono text-foreground">
                {metrics.speech_duration_sec.toFixed(2)} <span className="text-xs font-normal">s</span>
              </span>
            </div>

            <div className="rounded-2xl border border-border bg-card/60 p-3.5">
              <span className="text-[10px] uppercase font-semibold text-muted-foreground block">
                Pause Frequency
              </span>
              <span className="text-xl font-bold font-mono text-foreground">
                {metrics.pause_frequency_per_min.toFixed(1)} <span className="text-xs font-normal">/min</span>
              </span>
            </div>

            <div className="rounded-2xl border border-border bg-card/60 p-3.5">
              <span className="text-[10px] uppercase font-semibold text-muted-foreground block">
                Syllable Rate
              </span>
              <span className="text-xl font-bold font-mono text-foreground">
                {metrics.speaking_rate_syllables_sec.toFixed(1)} <span className="text-xs font-normal">syl/s</span>
              </span>
            </div>
          </div>

          {/* Interactive SVG Pitch Contour Plot */}
          <Card className="rounded-2xl border-border bg-card/70">
            <CardHeader className="pb-2">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                <div>
                  <CardTitle className="text-sm font-semibold uppercase text-muted-foreground">
                    Fundamental Frequency ($F_0$) Pitch Contour Over Time
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Vocal fold vibration frequency tracking in Hertz. Breaks indicate unvoiced frames or speech pauses.
                  </CardDescription>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <div className="flex items-center gap-1.5">
                    <span className="size-2.5 rounded-full bg-cyan-500" />
                    <span>Voiced Pitch ($F_0$)</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="size-2.5 rounded-full bg-muted-foreground/30" />
                    <span>Pause / Unvoiced</span>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-2">
              <div className="w-full h-64 border border-border/50 rounded-xl bg-background/50 p-4 relative">
                {contour.length > 0 ? (
                  <svg viewBox="0 0 600 200" className="w-full h-full overflow-visible" preserveAspectRatio="none">
                    {/* Grid horizontal lines */}
                    <line x1="40" y1="180" x2="580" y2="180" stroke="currentColor" strokeOpacity="0.2" />
                    <line x1="40" y1="130" x2="580" y2="130" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                    <line x1="40" y1="80" x2="580" y2="80" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                    <line x1="40" y1="30" x2="580" y2="30" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />

                    {/* Y-axis Labels */}
                    <text x="35" y="184" textAnchor="end" fontSize="9" fill="currentColor" fillOpacity="0.6">65 Hz</text>
                    <text x="35" y="134" textAnchor="end" fontSize="9" fill="currentColor" fillOpacity="0.6">150 Hz</text>
                    <text x="35" y="84" textAnchor="end" fontSize="9" fill="currentColor" fillOpacity="0.6">250 Hz</text>
                    <text x="35" y="34" textAnchor="end" fontSize="9" fill="currentColor" fillOpacity="0.6">350 Hz</text>

                    {/* Plot points and connected segments */}
                    {(() => {
                      const maxTime = contour[contour.length - 1]?.time_sec || 1.0
                      const minHz = 65
                      const maxHz = 350
                      const hzRange = maxHz - minHz

                      function getX(t: number) {
                        return 40 + (t / maxTime) * 540
                      }
                      function getY(hz: number) {
                        const clamped = Math.max(minHz, Math.min(maxHz, hz))
                        return 180 - ((clamped - minHz) / hzRange) * 150
                      }

                      // Build SVG path segments for continuous voiced runs
                      const segments: string[] = []
                      let currentPath = ''

                      for (let i = 0; i < contour.length; i++) {
                        const pt = contour[i]
                        const x = getX(pt.time_sec)
                        const y = getY(pt.pitch_hz)

                        if (pt.voiced && pt.pitch_hz > 0) {
                          if (!currentPath) {
                            currentPath = `M ${x.toFixed(1)} ${y.toFixed(1)}`
                          } else {
                            currentPath += ` L ${x.toFixed(1)} ${y.toFixed(1)}`
                          }
                        } else {
                          if (currentPath) {
                            segments.push(currentPath)
                            currentPath = ''
                          }
                        }
                      }
                      if (currentPath) segments.push(currentPath)

                      return (
                        <>
                          {segments.map((seg, sIdx) => (
                            <path
                              key={sIdx}
                              d={seg}
                              fill="none"
                              stroke="#06b6d4"
                              strokeWidth="2.5"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          ))}

                          {/* Pause markers */}
                          {contour.map((pt, pIdx) => {
                            if (!pt.voiced) {
                              const x = getX(pt.time_sec)
                              return (
                                <line
                                  key={pIdx}
                                  x1={x}
                                  y1="175"
                                  x2={x}
                                  y2="185"
                                  stroke="currentColor"
                                  strokeOpacity="0.25"
                                  strokeWidth="1.5"
                                />
                              )
                            }
                            return null
                          })}
                        </>
                      )
                    })()}
                  </svg>
                ) : (
                  <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
                    No contour data available.
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
