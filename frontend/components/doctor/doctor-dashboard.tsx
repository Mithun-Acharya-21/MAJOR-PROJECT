'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'
import { 
  Users, Activity, BarChart3, Database, Mic, Search, Filter, 
  ExternalLink, Clock, AlertTriangle, CheckCircle, ShieldAlert,
  ChevronRight, RefreshCw, FileText
} from 'lucide-react'

export interface SessionSummary {
  session_id: string
  patient_id: string
  patient_alias?: string
  age_group?: string
  gender?: string
  started_at?: number | string
  created_at?: string
  status: string
  screening_score?: number
  confidence?: number
  risk_level?: string
  risk_indicator?: 'elevated' | 'low_risk' | 'pending' | 'inconclusive'
  modalities_used?: string[]
  modalities_available?: string[]
  notes_count?: number
  clinician_notes_count?: number
}

export function DoctorDashboard() {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'elevated' | 'low_risk' | 'in_progress'>('all')

  useEffect(() => {
    fetchSessions()
  }, [])

  async function fetchSessions() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/doctor/sessions')
      if (!res.ok) {
        throw new Error(`Failed to load sessions (HTTP ${res.status})`)
      }
      const data = await res.json()
      setSessions(data.sessions || [])
    } catch (err: any) {
      setError(err.message || 'Error fetching screening sessions.')
    } finally {
      setLoading(false)
    }
  }

  // Helper to determine risk level
  function getRiskIndicator(s: SessionSummary): 'elevated' | 'low_risk' | 'in_progress' {
    if (s.status === 'in_progress') return 'in_progress'
    const lvl = (s.risk_level || '').toLowerCase()
    if (lvl.includes('elevated') || (s.screening_score != null && s.screening_score >= 0.50)) {
      return 'elevated'
    }
    return 'low_risk'
  }

  // Filter sessions
  const filteredSessions = sessions.filter((s) => {
    const alias = s.patient_alias || ''
    const matchesSearch = 
      s.patient_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.session_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      alias.toLowerCase().includes(searchQuery.toLowerCase())

    if (!matchesSearch) return false

    const risk = getRiskIndicator(s)
    if (statusFilter === 'all') return true
    if (statusFilter === 'elevated') return risk === 'elevated'
    if (statusFilter === 'low_risk') return risk === 'low_risk'
    if (statusFilter === 'in_progress') return risk === 'in_progress'
    return true
  })

  // Aggregate statistics
  const totalCount = sessions.length
  const elevatedCount = sessions.filter(s => getRiskIndicator(s) === 'elevated').length
  const lowRiskCount = sessions.filter(s => getRiskIndicator(s) === 'low_risk').length
  const inProgressCount = sessions.filter(s => getRiskIndicator(s) === 'in_progress').length

  function formatSessionDate(ts?: number | string) {
    if (!ts) return 'Recent'
    const d = typeof ts === 'number' ? (ts < 1e11 ? new Date(ts * 1000) : new Date(ts)) : new Date(ts)
    if (isNaN(d.getTime())) return String(ts)
    return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
  }

  return (
    <div className="mx-auto flex w-full max-w-[1240px] flex-col gap-8 px-4 py-8 sm:px-6">
      {/* Top Banner: Clinical Research Disclaimer */}
      <Alert className="border-amber-500/40 bg-amber-500/10 text-amber-900 dark:text-amber-200">
        <ShieldAlert className="size-5 text-amber-600 dark:text-amber-400" />
        <AlertTitle className="text-sm font-semibold tracking-wide uppercase">
          Investigational Clinical Research Portal
        </AlertTitle>
        <AlertDescription className="text-xs sm:text-sm mt-1">
          Experimental screening result — requires professional interpretation. NeuroScreen is an auxiliary,
          multimodal decision-support research prototype. Outputs must not be used as an autonomous psychiatric diagnosis.
        </AlertDescription>
      </Alert>

      {/* Header and Quick Navigation */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground sm:text-3xl">
            Clinician & Researcher Portal
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Monitor multimodal patient screening cohorts, verify model calibration, inspect prosodic voice features, and review longitudinal trajectories.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Link href="/doctor/analytics">
            <Button variant="outline" size="sm" className="gap-1.5 rounded-xl">
              <BarChart3 className="size-4 text-primary" />
              Model Analytics
            </Button>
          </Link>
          <Link href="/doctor/datasets">
            <Button variant="outline" size="sm" className="gap-1.5 rounded-xl">
              <Database className="size-4 text-emerald-500" />
              Datasets & Models
            </Button>
          </Link>
          <Link href="/doctor/voice-analysis">
            <Button variant="outline" size="sm" className="gap-1.5 rounded-xl">
              <Mic className="size-4 text-cyan-500" />
              Voice Pitch Lab
            </Button>
          </Link>
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={fetchSessions}
            title="Refresh sessions"
            className="rounded-xl"
          >
            <RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>

      {/* Cohort Summary Metric Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="rounded-2xl border-border bg-card/60 backdrop-blur-sm">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-xs font-semibold uppercase text-muted-foreground">
              Total Screened Sessions
            </CardTitle>
            <Users className="size-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalCount}</div>
            <p className="mt-1 text-xs text-muted-foreground">Recorded research cohorts</p>
          </CardContent>
        </Card>

        <Card className="rounded-2xl border-border bg-card/60 backdrop-blur-sm">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-xs font-semibold uppercase text-muted-foreground">
              Elevated Indicators
            </CardTitle>
            <AlertTriangle className="size-4 text-amber-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">{elevatedCount}</div>
            <p className="mt-1 text-xs text-muted-foreground">Screening score &ge; 0.50 threshold</p>
          </CardContent>
        </Card>

        <Card className="rounded-2xl border-border bg-card/60 backdrop-blur-sm">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-xs font-semibold uppercase text-muted-foreground">
              Lower-Risk Indicators
            </CardTitle>
            <CheckCircle className="size-4 text-emerald-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">{lowRiskCount}</div>
            <p className="mt-1 text-xs text-muted-foreground">Screening score &lt; 0.50 threshold</p>
          </CardContent>
        </Card>

        <Card className="rounded-2xl border-border bg-card/60 backdrop-blur-sm">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-xs font-semibold uppercase text-muted-foreground">
              In Progress / Inconclusive
            </CardTitle>
            <Clock className="size-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{inProgressCount}</div>
            <p className="mt-1 text-xs text-muted-foreground">Awaiting additional modalities</p>
          </CardContent>
        </Card>
      </div>

      {/* Patient Screening Session List with Filters */}
      <Card className="rounded-2xl border-border bg-card/70 backdrop-blur-sm">
        <CardHeader className="border-b border-border/60 pb-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-lg font-semibold">Patient Screening Sessions</CardTitle>
              <CardDescription className="text-xs text-muted-foreground">
                Click any session to view individual multimodal breakdown, LIME explanations, longitudinal timeline, and record notes.
              </CardDescription>
            </div>

            {/* Filter controls */}
            <div className="flex flex-wrap items-center gap-2">
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Search Patient ID / Session..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="h-9 w-48 rounded-xl border border-input bg-background/80 pl-8 pr-3 text-xs placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary sm:w-64"
                />
              </div>

              <div className="flex items-center gap-1 rounded-xl border border-border bg-background/50 p-1">
                <button
                  type="button"
                  onClick={() => setStatusFilter('all')}
                  className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
                    statusFilter === 'all' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  All ({totalCount})
                </button>
                <button
                  type="button"
                  onClick={() => setStatusFilter('elevated')}
                  className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
                    statusFilter === 'elevated' ? 'bg-amber-600 text-white' : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  Elevated ({elevatedCount})
                </button>
                <button
                  type="button"
                  onClick={() => setStatusFilter('low_risk')}
                  className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
                    statusFilter === 'low_risk' ? 'bg-emerald-600 text-white' : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  Lower-Risk ({lowRiskCount})
                </button>
                <button
                  type="button"
                  onClick={() => setStatusFilter('in_progress')}
                  className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
                    statusFilter === 'in_progress' ? 'bg-blue-600 text-white' : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  Active ({inProgressCount})
                </button>
              </div>
            </div>
          </div>
        </CardHeader>

        <CardContent className="p-0">
          {error && (
            <div className="p-6">
              <Alert variant="destructive">
                <AlertTitle>Failed to load data</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            </div>
          )}

          {loading ? (
            <div className="flex flex-col items-center justify-center p-12 text-sm text-muted-foreground">
              <RefreshCw className="mb-2 size-6 animate-spin text-primary" />
              Loading research sessions...
            </div>
          ) : filteredSessions.length === 0 ? (
            <div className="p-12 text-center text-sm text-muted-foreground">
              No screening sessions match the current query or filter.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs sm:text-sm">
                <thead>
                  <tr className="border-b border-border/50 bg-muted/30 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    <th className="px-4 py-3 sm:px-6">Patient ID</th>
                    <th className="px-4 py-3">Timestamp</th>
                    <th className="px-4 py-3">Modalities</th>
                    <th className="px-4 py-3">Calibrated Score</th>
                    <th className="px-4 py-3">Category</th>
                    <th className="px-4 py-3">Notes</th>
                    <th className="px-4 py-3 text-right sm:px-6">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40">
                  {filteredSessions.map((s) => {
                    const risk = getRiskIndicator(s)
                    const isElevated = risk === 'elevated'
                    const isLowRisk = risk === 'low_risk'
                    const isPending = risk === 'in_progress'
                    const mods = s.modalities_used || s.modalities_available || ['text']
                    const notesCount = s.notes_count ?? s.clinician_notes_count ?? 0

                    return (
                      <tr 
                        key={s.session_id} 
                        className="transition-colors hover:bg-muted/40"
                      >
                        <td className="px-4 py-3.5 sm:px-6">
                          <div className="font-semibold text-foreground">{s.patient_id}</div>
                          <div className="text-xs text-muted-foreground">
                            {s.patient_alias ? `${s.patient_alias} • ` : ''}ID: {s.session_id.slice(0, 12)}
                          </div>
                        </td>
                        <td className="px-4 py-3.5 text-xs text-muted-foreground whitespace-nowrap">
                          {formatSessionDate(s.started_at || s.created_at)}
                        </td>
                        <td className="px-4 py-3.5">
                          <div className="flex flex-wrap gap-1">
                            {mods.map((mod) => (
                              <Badge 
                                key={mod} 
                                variant="outline" 
                                className="px-1.5 py-0 text-[10px] uppercase font-mono font-normal"
                              >
                                {mod}
                              </Badge>
                            ))}
                          </div>
                        </td>
                        <td className="px-4 py-3.5 font-mono text-xs">
                          {isPending ? (
                            <span className="text-muted-foreground">In progress</span>
                          ) : (
                            <div className="flex items-center gap-2">
                              <span className="font-semibold">
                                {(((s.confidence ?? s.screening_score ?? 0.5) * 100)).toFixed(1)}%
                              </span>
                              <span className="text-[11px] text-muted-foreground">
                                (Score: {s.screening_score?.toFixed(2) ?? '—'})
                              </span>
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-3.5">
                          {isElevated ? (
                            <Badge className="bg-amber-500/15 text-amber-600 border-amber-500/30 dark:text-amber-400">
                              Elevated Indicator
                            </Badge>
                          ) : isLowRisk ? (
                            <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/30 dark:text-emerald-400">
                              Lower-Risk Indicator
                            </Badge>
                          ) : (
                            <Badge variant="outline" className="text-blue-500">
                              Screening Active
                            </Badge>
                          )}
                        </td>
                        <td className="px-4 py-3.5 text-xs">
                          {notesCount > 0 ? (
                            <span className="inline-flex items-center gap-1 text-primary">
                              <FileText className="size-3" />
                              {notesCount} note{notesCount > 1 ? 's' : ''}
                            </span>
                          ) : (
                            <span className="text-muted-foreground/60">—</span>
                          )}
                        </td>
                        <td className="px-4 py-3.5 text-right sm:px-6">
                          <Link href={`/doctor/session/${s.session_id}`}>
                            <Button size="sm" variant="outline" className="h-8 gap-1 rounded-lg text-xs">
                              Inspect Result
                              <ChevronRight className="size-3.5" />
                            </Button>
                          </Link>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
