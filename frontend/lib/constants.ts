import { FileText, Camera, ScanFace, AudioLines, Sparkles, type LucideIcon } from 'lucide-react'
import type { ModalityKind, RiskLevel } from './types'

export interface StepConfig {
  id: number
  key: ModalityKind | 'result'
  slug: string
  label: string
  short: string
  description: string
  icon: LucideIcon
}

export const STEPS: StepConfig[] = [
  {
    id: 1,
    key: 'text',
    slug: 'text',
    label: 'Text Analysis',
    short: 'Text',
    description: 'Describe how you have been feeling recently.',
    icon: FileText,
  },
  {
    id: 2,
    key: 'photo',
    slug: 'photo',
    label: 'Photo Analysis',
    short: 'Photo',
    description: 'Analyze facial-expression information from a photo.',
    icon: Camera,
  },
  {
    id: 3,
    key: 'face',
    slug: 'face',
    label: 'Live Face',
    short: 'Face',
    description: 'Capture facial expressions over several webcam frames.',
    icon: ScanFace,
  },
  {
    id: 4,
    key: 'voice',
    slug: 'voice',
    label: 'Voice & Speech',
    short: 'Voice',
    description: 'Analyze speech characteristics and transcript.',
    icon: AudioLines,
  },
  {
    id: 5,
    key: 'result',
    slug: 'result',
    label: 'Result',
    short: 'Result',
    description: 'Your combined multimodal screening result.',
    icon: Sparkles,
  },
]

export const MODALITY_META: Record<ModalityKind, { label: string; icon: LucideIcon }> = {
  text: { label: 'Text', icon: FileText },
  photo: { label: 'Photo', icon: Camera },
  face: { label: 'Live Face', icon: ScanFace },
  voice: { label: 'Voice', icon: AudioLines },
}

export const RISK_META: Record<
  RiskLevel,
  { label: string; token: string; badgeClass: string }
> = {
  low: {
    label: 'Lower-risk indicators',
    token: 'risk-low',
    badgeClass: 'bg-risk-low/12 text-risk-low border-risk-low/25',
  },
  moderate: {
    label: 'Moderate-risk indicators',
    token: 'risk-moderate',
    badgeClass: 'bg-risk-moderate/12 text-risk-moderate border-risk-moderate/25',
  },
  high: {
    label: 'Higher-risk indicators',
    token: 'risk-high',
    badgeClass: 'bg-risk-high/12 text-risk-high border-risk-high/25',
  },
}

export const SAMPLE_TEXT = {
  higher:
    "Honestly I've been really drained lately. I can't concentrate on anything, I've lost interest in things I used to enjoy, and I'm tired all the time no matter how much I sleep. Most days it feels pointless to even get started.",
  lower:
    "Things have been pretty steady this week. Work was busy but manageable, I've been sleeping okay and got out for a couple of walks. I felt a bit stressed on Tuesday but caught up with friends over the weekend and that helped.",
}

export const EXAMPLE_PROMPTS = [
  'How have you been feeling lately?',
  'How has your daily routine been?',
  'Have you noticed changes in motivation or concentration?',
]

export const FRAME_TARGET = 10
