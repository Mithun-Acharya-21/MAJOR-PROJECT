import { Metadata } from 'next'
import { SiteHeader } from '@/components/layout/site-header'
import { SiteFooter } from '@/components/layout/site-footer'
import { DatasetInfoView } from '@/components/doctor/dataset-info-view'

export const metadata: Metadata = {
  title: 'Datasets & Model Information — NeuroScreen Clinical Research',
  description: 'Dataset provenance, participant cohorts, ground truth scales, and model readiness taxonomy.',
}

export default function DatasetsPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <SiteHeader />
      <main className="flex-1 pb-16">
        <DatasetInfoView />
      </main>
      <SiteFooter />
    </div>
  )
}
