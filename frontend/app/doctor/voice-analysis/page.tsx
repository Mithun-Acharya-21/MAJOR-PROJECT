import { Metadata } from 'next'
import { SiteHeader } from '@/components/layout/site-header'
import { SiteFooter } from '@/components/layout/site-footer'
import { VoicePitchView } from '@/components/doctor/voice-pitch-view'

export const metadata: Metadata = {
  title: 'Voice Pitch & Prosody Lab — NeuroScreen Clinical Research',
  description: 'Interactive fundamental frequency pitch contour tracking, speaking rate, and multilingual acoustic prosodic analysis.',
}

export default function VoiceAnalysisPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <SiteHeader />
      <main className="flex-1 pb-16">
        <VoicePitchView />
      </main>
      <SiteFooter />
    </div>
  )
}
