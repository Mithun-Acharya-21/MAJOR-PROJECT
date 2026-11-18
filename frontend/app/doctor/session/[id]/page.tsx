import { Metadata } from 'next'
import { SiteHeader } from '@/components/layout/site-header'
import { SiteFooter } from '@/components/layout/site-footer'
import { SessionDetailView } from '@/components/doctor/session-detail-view'

interface PageProps {
  params: Promise<{ id: string }>
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { id } = await params
  return {
    title: `Session ${id} — NeuroScreen Clinical Research`,
    description: `Detailed multimodal screening result and clinical observation trace for session ${id}.`,
  }
}

export default async function SessionDetailPage({ params }: PageProps) {
  const { id } = await params

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <SiteHeader />
      <main className="flex-1 pb-16">
        <SessionDetailView sessionId={id} />
      </main>
      <SiteFooter />
    </div>
  )
}
