'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { Textarea } from '@/components/ui/textarea'
import { 
  ArrowLeft, ShieldAlert, FileText, Activity, AlertTriangle, 
  CheckCircle, Clock, Mic, Video, MessageSquare, Layers, 
  Send, User, TrendingUp, Info
} from 'lucide-react'

interface SessionDetailProps {
  sessionId: string
}

export function SessionDetailView({ sessionId }: SessionDetailProps) {
  const [data, setData] = useState<any | null>(null)
  const [timeline, setTimeline] = useState<any | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Clinician notes state
  const [notes, setNotes] = useState<any[]>([])
  const [newNote, setNewNote] = useState('')
  const [submittingNote, setSubmittingNote] = useState(false)
  const [noteSuccess, setNoteSuccess] = useState(false)

  useEffect(() => {
    fetchSessionData()
  }, [sessionId])

  async function fetchSessionData() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`/api/doctor/sessions/${sessionId}`)
      if (!res.ok) {
        throw new Error(`Failed to load session details (HTTP ${res.status})`)
      }
      const json = await res.json()
      setData(json)
      setNotes(json.clinician_notes || [])

      // Fetch longitudinal timeline if patient_id is present
      if (json.patient_id) {
        fetchTimeline(json.patient_id)
      }
    } catch (err: any) {
      setError(err.message || 'Error loading session data.')
    } finally {
      setLoading(false)
    }
  }

  async function fetchTimeline(patientId: string) {
    try {
      const res = await fetch(`/api/doctor/patient/${patientId}/timeline`)
      if (res.ok) {
        const tData = await res.json()
        setTimeline(tData)
      }
    } catch {
      // Timeline is optional for first-time patients
    }
  }

  async function handleAddNote(e: React.FormEvent) {
    e.preventDefault()
    if (!newNote.trim() || submittingNote) return

    setSubmittingNote(true)
    setNoteSuccess(false)
    try {
      const res = await fetch(`/api/doctor/sessions/${sessionId}/notes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          note: newNote.trim(),
          clinician: 'Attending Research Clinician'
        })
      })
      if (!res.ok) throw new Error('Failed to save clinician note.')
      const resData = await res.json()
      setNotes(resData.notes || [...notes, {
        note: newNote.trim(),
        clinician: 'Attending Research Clinician',
        created_at: new Date().toISOString()
      }])
      setNewNote('')
      setNoteSuccess(true)
      setTimeout(() => setNoteSuccess(false), 3000)
    } catch (err: any) {
      alert(err.message || 'Error recording note.')
    } finally {
      setSubmittingNote(false)
    }
  }

  if (loading) {
    return (
      <div className="mx-auto flex w-full max-w-[1240px] items-center justify-center py-24 text-sm text-muted-foreground">
        <Activity className="mr-2 size-5 animate-spin text-primary" />
        Loading clinical screening record...
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="mx-auto flex w-full max-w-[1240px] flex-col gap-4 px-4 py-12">
        <Link href="/doctor">
          <Button variant="ghost" size="sm" className="gap-1 rounded-xl">
            <ArrowLeft className="size-4" /> Back to Doctor Dashboard
          </Button>
        </Link>
        <Alert variant="destructive">
          <AlertTitle>Unable to retrieve session</AlertTitle>
          <AlertDescription>{error || 'Session not found.'}</AlertDescription>
        </Alert>
      </div>
    )
  }

  const res = data.screening_result || {}
  const qRes = data.questionnaire_result || {}
  const modalities = data.modality_breakdown || {}
  const lime = data.lime_attributions || []
  const isElevated = res.category?.toLowerCase().includes('elevated') || (res.score != null && res.score >= 0.5)

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
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-foreground sm:text-2xl">
                Session: {sessionId}
              </h1>
              <Badge variant="outline" className="font-mono text-xs">
                {data.patient_id || 'PATIENT'}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              Conducted: {data.started_at ? new Date(typeof data.started_at === 'number' && data.started_at < 1e11 ? data.started_at * 1000 : data.started_at).toLocaleString() : 'Recent'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isElevated ? (
            <Badge className="bg-amber-500/15 text-amber-600 border-amber-500/30 text-xs px-3 py-1">
              Elevated Screening Indicator
            </Badge>
          ) : (
            <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/30 text-xs px-3 py-1">
              Lower-Risk Screening Indicator
            </Badge>
          )}
        </div>
      </div>

      {/* Prominent Clinical Disclaimer */}
      <Alert className="border-amber-500/40 bg-amber-500/10 text-amber-900 dark:text-amber-200">
        <ShieldAlert className="size-5 text-amber-600 dark:text-amber-400" />
        <AlertTitle className="text-xs font-semibold tracking-wide uppercase">
          Experimental Screening Result — Requires Professional Interpretation
        </AlertTitle>
        <AlertDescription className="text-xs sm:text-sm mt-1">
          This system is an investigational multimodal decision-support research prototype. 
          Models provide statistical indicators based on training corpora (EATD-Corpus, AVDD-TCMI).
          Results must never be construed as an autonomous psychiatric diagnosis.
        </AlertDescription>
      </Alert>

      {/* Top Screening Result Overview Card */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="rounded-2xl border-border bg-card/70 lg:col-span-1">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold uppercase text-muted-foreground">
              Calibrated Screening Score
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex items-baseline gap-2">
              <span className="text-4xl font-extrabold text-foreground">
                {((res.confidence ?? res.score ?? 0.5) * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-muted-foreground">calibrated probability</span>
            </div>

            <Progress 
              value={(res.confidence ?? res.score ?? 0.5) * 100} 
              className={`h-2.5 w-full ${isElevated ? '[&>div]:bg-amber-500' : '[&>div]:bg-emerald-500'}`} 
            />

            <div className="rounded-xl border border-border bg-muted/30 p-3 text-xs space-y-1.5">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Screening Category:</span>
                <span className="font-semibold">{res.category || 'Clinical Research Trace'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Raw Composite Score:</span>
                <span className="font-mono">{res.score?.toFixed(3) ?? '—'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Prediction Uncertainty:</span>
                <span className="font-mono">&plusmn;{(res.uncertainty ?? 0.15).toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Safety Override:</span>
                <span className={res.safety_override ? 'font-semibold text-rose-500' : 'text-muted-foreground'}>
                  {res.safety_override ? 'Active (Crisis Triggered)' : 'None'}
                </span>
              </div>
            </div>

            <p className="text-xs text-muted-foreground leading-relaxed">
              {res.explanation || 'Multimodal indicators synthesized across interview transcripts and vocal acoustic analysis.'}
            </p>
          </CardContent>
        </Card>

        {/* Adaptive Questionnaire Dimension Profile */}
        <Card className="rounded-2xl border-border bg-card/70 lg:col-span-2">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-sm font-semibold uppercase text-muted-foreground">
                  Adaptive Symptom Dimension Profile
                </CardTitle>
                <CardDescription className="text-xs">
                  {qRes.instrument || 'Experimental Adaptive Research Questionnaire'} ({qRes.total_questions_answered || 9} questions answered)
                </CardDescription>
              </div>
              <Badge variant="outline" className="text-xs">Psychometric Trace</Badge>
            </div>
          </CardHeader>
          <CardContent>
            {qRes.symptom_dimensions ? (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {Object.entries(qRes.symptom_dimensions).map(([dim, val]: [string, any]) => {
                  const severity = typeof val === 'object' ? val.severity : val
                  const maxSeverity = 3.0
                  const pct = Math.min(100, (severity / maxSeverity) * 100)
                  const dimLabel = dim.replace(/_/g, ' ')

                  return (
                    <div key={dim} className="rounded-xl border border-border/60 bg-muted/20 p-3">
                      <div className="flex items-center justify-between text-xs mb-1.5">
                        <span className="font-medium capitalize">{dimLabel}</span>
                        <span className="font-mono text-muted-foreground">
                          {typeof severity === 'number' ? severity.toFixed(1) : severity} / 3.0
                        </span>
                      </div>
                      <Progress value={pct} className="h-1.5" />
                      {typeof val === 'object' && val.observations && (
                        <span className="mt-1 block text-[10px] text-muted-foreground">
                          {val.observations} observation{val.observations > 1 ? 's' : ''} • confidence: {(val.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">No questionnaire dimension breakdown recorded for this session.</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Modality-Wise Breakdown Cards */}
      <div>
        <h2 className="text-base font-semibold text-foreground mb-4">Modality-Wise Analysis</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {/* Text Modality */}
          <Card className="rounded-2xl border-border bg-card/60">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xs font-semibold uppercase text-muted-foreground">Text Analysis</CardTitle>
                <MessageSquare className="size-4 text-primary" />
              </div>
            </CardHeader>
            <CardContent className="space-y-2 text-xs">
              <div className="font-medium text-foreground">
                {modalities.text?.model || 'DistilBERT + BiLSTM (EATD)'}
              </div>
              <div className="flex justify-between border-b border-border/40 pb-1">
                <span className="text-muted-foreground">Model Score:</span>
                <span className="font-mono font-semibold">
                  {modalities.text?.score != null ? modalities.text.score.toFixed(2) : 'N/A'}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                {modalities.text?.summary || 'Transcript semantics analyzed with cross-attention.'}
              </p>
            </CardContent>
          </Card>

          {/* Voice Modality */}
          <Card className="rounded-2xl border-border bg-card/60">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xs font-semibold uppercase text-muted-foreground">Voice Acoustic</CardTitle>
                <Mic className="size-4 text-cyan-500" />
              </div>
            </CardHeader>
            <CardContent className="space-y-2 text-xs">
              <div className="font-medium text-foreground">
                {modalities.voice?.model || 'openSMILE eGeMAPSv02 / Wav2Vec2'}
              </div>
              <div className="flex justify-between border-b border-border/40 pb-1">
                <span className="text-muted-foreground">Model Score:</span>
                <span className="font-mono font-semibold">
                  {modalities.voice?.score != null ? modalities.voice.score.toFixed(2) : 'N/A'}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                {modalities.voice?.summary || 'Acoustic prosody, F0 pitch contours, and pause rates.'}
              </p>
            </CardContent>
          </Card>

          {/* Facial Modality */}
          <Card className="rounded-2xl border-border bg-card/60">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xs font-semibold uppercase text-muted-foreground">Facial Affect</CardTitle>
                <Video className="size-4 text-violet-500" />
              </div>
            </CardHeader>
            <CardContent className="space-y-2 text-xs">
              <div className="font-medium text-foreground">
                {modalities.face?.model || 'MediaPipe FACS / FER-2013'}
              </div>
              <div className="flex justify-between border-b border-border/40 pb-1">
                <span className="text-muted-foreground">Auxiliary Affect:</span>
                <span className="font-mono font-semibold">
                  {modalities.face?.score != null ? modalities.face.score.toFixed(2) : 'Auxiliary'}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                {modalities.face?.summary || 'Auxiliary facial affect: strictly separated from depression.'}
              </p>
            </CardContent>
          </Card>

          {/* Late Fusion Modality */}
          <Card className="rounded-2xl border-border bg-card/60">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xs font-semibold uppercase text-muted-foreground">Multimodal Fusion</CardTitle>
                <Layers className="size-4 text-emerald-500" />
              </div>
            </CardHeader>
            <CardContent className="space-y-2 text-xs">
              <div className="font-medium text-foreground">
                {modalities.fusion?.model || 'AVDD Logistic Meta-Learner'}
              </div>
              <div className="flex justify-between border-b border-border/40 pb-1">
                <span className="text-muted-foreground">Combined Score:</span>
                <span className="font-mono font-semibold">
                  {modalities.fusion?.score != null ? modalities.fusion.score.toFixed(2) : (res.score?.toFixed(2) ?? '—')}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                {modalities.fusion?.summary || 'Ensemble stacking of text and acoustic predictive outputs.'}
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* LIME Token Attributions View */}
      {lime.length > 0 && (
        <Card className="rounded-2xl border-border bg-card/70">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold uppercase text-muted-foreground">
              LIME Feature Attribution Trace (Text Interview Semantics)
            </CardTitle>
            <CardDescription className="text-xs">
              Local Interpretable Model-agnostic Explanations for interview responses. Tokens contributing towards or away from the screening score.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {lime.map((item: any, idx: number) => {
                const isDep = item.direction === 'depressive_indicator' || item.weight > 0
                return (
                  <div 
                    key={idx}
                    className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs ${
                      isDep 
                        ? 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300' 
                        : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                    }`}
                  >
                    <span className="font-mono font-medium">{item.token}</span>
                    <span className="text-[10px] opacity-80">
                      ({item.weight > 0 ? '+' : ''}{Number(item.weight).toFixed(2)})
                    </span>
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Longitudinal Patient Trajectory Timeline */}
      {timeline && timeline.sessions && timeline.sessions.length > 1 && (
        <Card className="rounded-2xl border-border bg-card/70">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-sm font-semibold uppercase text-muted-foreground">
                  Longitudinal Patient Screening History
                </CardTitle>
                <CardDescription className="text-xs">
                  Tracking score changes over time for Patient: {timeline.patient_id} ({timeline.total_sessions} recorded sessions)
                </CardDescription>
              </div>
              <TrendingUp className="size-4 text-primary" />
            </div>
          </CardHeader>
          <CardContent>
            {/* Trajectory Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-border/50 text-muted-foreground uppercase text-[10px]">
                    <th className="py-2 px-3">Session</th>
                    <th className="py-2 px-3">Date</th>
                    <th className="py-2 px-3">Score</th>
                    <th className="py-2 px-3">Category</th>
                    <th className="py-2 px-3">Pitch Mean (Hz)</th>
                    <th className="py-2 px-3">Speech Dur (s)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/30">
                  {timeline.sessions.map((ts: any) => (
                    <tr key={ts.session_id} className={ts.session_id === sessionId ? 'bg-primary/5 font-semibold' : ''}>
                      <td className="py-2.5 px-3 font-mono">{ts.session_id}</td>
                      <td className="py-2.5 px-3 text-muted-foreground whitespace-nowrap">
                        {new Date(typeof ts.timestamp === 'number' && ts.timestamp < 1e11 ? ts.timestamp * 1000 : ts.timestamp).toLocaleDateString()}
                      </td>
                      <td className="py-2.5 px-3 font-mono">{ts.score?.toFixed(2) ?? '—'}</td>
                      <td className="py-2.5 px-3">
                        <Badge variant="outline" className="text-[10px]">
                          {ts.category || 'Screened'}
                        </Badge>
                      </td>
                      <td className="py-2.5 px-3 font-mono text-muted-foreground">
                        {ts.pitch_mean_hz ? `${ts.pitch_mean_hz} Hz` : '—'}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-muted-foreground">
                        {ts.speech_duration_sec ? `${ts.speech_duration_sec}s` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Clinician Research Observation Notes Section */}
      <Card className="rounded-2xl border-border bg-card/70">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold uppercase text-muted-foreground">
            Clinician Observation Notes
          </CardTitle>
          <CardDescription className="text-xs">
            Record professional research impressions, qualitative observations, or protocol exceptions for this session.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Notes display */}
          {notes.length > 0 ? (
            <div className="space-y-2">
              {notes.map((n: any, idx: number) => (
                <div key={idx} className="rounded-xl border border-border/60 bg-muted/30 p-3 text-xs">
                  <div className="flex items-center justify-between text-muted-foreground text-[11px] mb-1">
                    <span className="font-semibold text-foreground flex items-center gap-1">
                      <User className="size-3 text-primary" /> {n.clinician || 'Clinician'}
                    </span>
                    <span>
                      {n.created_at ? new Date(n.created_at).toLocaleString() : 'Saved'}
                    </span>
                  </div>
                  <p className="text-foreground whitespace-pre-wrap">{n.note}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground italic">No clinician observation notes recorded yet.</p>
          )}

          {/* Form to add note */}
          <form onSubmit={handleAddNote} className="space-y-2 pt-2">
            <Textarea
              placeholder="Enter clinical observations, behavioral indicators, or patient interview notes..."
              value={newNote}
              onChange={(e) => setNewNote(e.target.value)}
              className="text-xs min-h-[80px] rounded-xl"
            />
            <div className="flex items-center justify-between">
              {noteSuccess && (
                <span className="text-xs text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                  <CheckCircle className="size-3.5" /> Note saved successfully
                </span>
              )}
              <div className="ml-auto">
                <Button 
                  type="submit" 
                  size="sm" 
                  disabled={submittingNote || !newNote.trim()}
                  className="rounded-xl gap-1 text-xs"
                >
                  <Send className="size-3.5" />
                  {submittingNote ? 'Saving...' : 'Record Observation'}
                </Button>
              </div>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
