import { Metadata } from 'next'
import { SiteHeader } from '@/components/layout/site-header'
import { SiteFooter } from '@/components/layout/site-footer'
import { ModelAnalyticsView } from '@/components/doctor/model-analytics-view'

export const metadata: Metadata = {
  title: 'Model Performance Analytics — NeuroScreen Clinical Research',
  description: 'Rigorous benchmark metrics, ROC/PR curves, confusion matrices, and population divergence analysis.',
}

export default function AnalyticsPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <SiteHeader />
      <main className="flex-1 pb-16">
        <ModelAnalyticsView />
      </main>
      <SiteFooter />
    </div>
  )
}
