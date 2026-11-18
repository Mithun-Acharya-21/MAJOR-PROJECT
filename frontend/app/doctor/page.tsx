import { Metadata } from 'next'
import { SiteHeader } from '@/components/layout/site-header'
import { SiteFooter } from '@/components/layout/site-footer'
import { DoctorDashboard } from '@/components/doctor/doctor-dashboard'

export const metadata: Metadata = {
  title: 'Doctor Portal — NeuroScreen Clinical Research',
  description: 'Doctor-facing clinical research screening portal, cohort monitoring, and multimodal model analysis.',
}

export default function DoctorPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <SiteHeader />
      <main className="flex-1 pb-16">
        <DoctorDashboard />
      </main>
      <SiteFooter />
    </div>
  )
}
