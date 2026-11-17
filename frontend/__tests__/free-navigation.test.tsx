/**
 * Free-navigation acceptance tests (spec: users can open Text, Photo,
 * Live Face, Voice and Results freely and independently, in any order,
 * regardless of assessment completion).
 */
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import React from 'react'

// The API layer is network-backed; navigation tests only care about UI state.
vi.mock('@/lib/api', () => ({
  ApiError: class ApiError extends Error {
    status?: number
    constructor(message: string, status?: number) {
      super(message)
      this.name = 'ApiError'
      this.status = status
    }
  },
  isApiConfigured: () => true,
  api: {
    screeningStart: vi.fn(() =>
      Promise.resolve({
        session_id: 'sess-1',
        status: 'in_progress',
        question: {
          id: 'q1',
          text: 'How have you been feeling lately?',
          target_dimensions: ['mood'],
          options: [],
        },
        progress: {
          questions_answered: 0,
          min_questions: 3,
          max_questions: 8,
          progress_percent: 0,
          required_dimensions_covered: false,
          dimension_coverage: {},
          missing_information: [],
        },
      }),
    ),
    screeningResponseOption: vi.fn(),
    screeningComplete: vi.fn(),
    screeningResearchTrace: vi.fn(),
    startSession: vi.fn(() => Promise.resolve({ session_id: 'legacy-1', status: 'started' })),
    analyzeText: vi.fn(),
    analyzePhoto: vi.fn(),
    analyzeFace: vi.fn(),
    analyzeVoice: vi.fn(),
    analyzeMultimodal: vi.fn(),
    health: vi.fn(),
    metrics: vi.fn(),
    recommendations: vi.fn(),
  },
}))

// next/link + next/image need the Next runtime; simple stand-ins suffice.
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: unknown; children?: React.ReactNode }) => (
    <a href={typeof href === 'string' ? href : '#'} {...rest}>
      {children}
    </a>
  ),
}))
vi.mock('next/image', () => ({
  default: (props: Record<string, unknown>) => {
    const { src, alt, ...rest } = props as { src: string; alt?: string }
    void rest
    // eslint-disable-next-line @next/next/no-img-element
    return <img src={src} alt={alt ?? ''} />
  },
}))

import ScreeningPage from '@/app/screening/page'
import { ScreeningProvider } from '@/lib/screening-store'
import { ThemeProvider } from '@/components/theme-provider'
import { api } from '@/lib/api'

const STORAGE_KEY = 'emotiondep-session'

function renderPage() {
  return render(
    <ThemeProvider>
      <ScreeningProvider>
        <ScreeningPage />
      </ScreeningProvider>
    </ThemeProvider>,
  )
}

/** All five stepper buttons (desktop stepper exposes 01–05 labels). */
function stepButton(label: RegExp) {
  const buttons = screen
    .getAllByRole('button')
    .filter((b) => label.test(b.textContent ?? ''))
  expect(buttons.length).toBeGreaterThan(0)
  return buttons[0]!
}

beforeEach(() => {
  localStorage.clear()
  vi.clearAllMocks()
})

describe('free navigation across assessment modules', () => {
  it('enables every navigation item when nothing is completed', async () => {
    renderPage()
    // Desktop + mobile steppers are both present in jsdom (CSS-hidden only).
    const navs = screen.getAllByRole('list', { name: /assessment steps/i })
    const buttons = navs.flatMap((nav) => within(nav).getAllByRole('button'))
    expect(buttons.length).toBeGreaterThanOrEqual(5)
    for (const b of buttons) {
      expect((b as HTMLButtonElement).disabled).toBe(false)
    }
  })

  it('opens Voice directly without completing Text', async () => {
    renderPage()
    fireEvent.click(stepButton(/Voice/))
    // Voice module UI (recorder) is present.
    expect(await screen.findByText(/Ready to record/i)).toBeTruthy()
    // Text interview was never started for this path? It mounts by default on
    // step 1 — but navigating away must not block anything.
    expect(screen.queryByText(/No screening session found/i)).toBeNull()
  })

  it('opens Live Face directly without Text or Photo', async () => {
    renderPage()
    fireEvent.click(stepButton(/Live Face/))
    expect(await screen.findByText(/Camera access required/i)).toBeTruthy()
  })

  it('opens Photo directly without completing Text', async () => {
    renderPage()
    fireEvent.click(stepButton(/Face(?!.*Live)/))
    expect(await screen.findByText(/Upload a facial photo/i)).toBeTruthy()
  })

  it('opens Results before any assessment and shows the empty state', async () => {
    renderPage()
    fireEvent.click(stepButton(/Result/))
    expect(
      await screen.findByText(
        /No assessment results available yet\. Complete any assessment to view its results here\./,
      ),
    ).toBeTruthy()
    // No fabrication, no error state, no forced redirect.
    expect(screen.queryByText(/No screening session found/i)).toBeNull()
    expect(screen.queryByText(/Could not compute your result/i)).toBeNull()
    // Jump buttons to every assessment are offered.
    expect(screen.getByRole('button', { name: /Start Text/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Start Face/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Start Live Face/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Start Voice/i })).toBeTruthy()
  })

  it('lets users switch between all five sections in any order', async () => {
    renderPage()
    const order: RegExp[] = [/Voice/, /Result/, /Text/, /Live Face/, /Face(?!.*Live)/, /Result/]
    for (const step of order) {
      fireEvent.click(stepButton(step))
    }
    // Landed on Results again — empty state renders (nothing completed).
    expect(
      await screen.findByText(/No assessment results available yet/i),
    ).toBeTruthy()
  })

  it('marks incomplete assessments as "Not completed" in partial results', async () => {
    // Simulate a completed photo assessment only (store shape = persisted state).
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        completedSteps: ['photo'],
        photo: {
          modality: 'photo',
          riskLevel: 'low',
          probability: 0.31,
          confidence: 0.72,
          faceDetected: true,
        },
      }),
    )
    renderPage()
    fireEvent.click(stepButton(/Result/))
    expect(await screen.findByText(/Assessment status/i)).toBeTruthy()
    expect(screen.getByText('Completed')).toBeTruthy()
    expect(screen.getAllByText('Not completed').length).toBeGreaterThanOrEqual(3)
    // Honest note: no invented fusion without the interview session.
    expect(
      screen.getByText(/No scores are estimated for assessments you have not completed/i),
    ).toBeTruthy()
    // Return directly to any assessment from Results.
    expect(screen.getByRole('button', { name: /Go to text interview/i })).toBeTruthy()
  })

  it('preserves interview answers when leaving and returning to Text', async () => {
    renderPage()
    // Wait for the interview to start.
    const textarea = (await screen.findByPlaceholderText(/Answer in your own words/i)) as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: 'I have felt tired all week.' } })

    // Leave to Voice and come back.
    fireEvent.click(stepButton(/Voice/))
    await screen.findByText(/Ready to record/i)
    fireEvent.click(stepButton(/Text/))

    // Same interview restored — no restart, draft preserved.
    const restored = (await screen.findByPlaceholderText(
      /Answer in your own words/i,
    )) as HTMLTextAreaElement
    expect(restored.value).toBe('I have felt tired all week.')
    expect(screen.getByText(/How have you been feeling lately/i)).toBeTruthy()
    // screeningStart must not have been called a second time.
    expect(api.screeningStart).toHaveBeenCalledTimes(1)
  })

  it('shows the completed-interview panel (not a restart) after finishing', async () => {
    vi.mocked(api.screeningResponseOption).mockResolvedValueOnce({
      session_id: 'sess-1',
      safety_concern: false,
      question: null,
      done: true,
      progress: {
        questions_answered: 3,
        min_questions: 3,
        max_questions: 8,
        progress_percent: 100,
        required_dimensions_covered: true,
        dimension_coverage: {},
        missing_information: [],
      },
      status: 'completed',
    })

    renderPage()
    const textarea = (await screen.findByPlaceholderText(
      /Answer in your own words/i,
    )) as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: 'Mostly fine.' } })
    fireEvent.click(screen.getByRole('button', { name: /Send answer/i }))

    // Page auto-advances to Photo on completion; return to Text.
    await waitFor(() => expect(screen.getByText(/Upload a facial photo/i)).toBeTruthy())
    fireEvent.click(stepButton(/Text/))

    expect(await screen.findByText(/Interview complete/i)).toBeTruthy()
    expect(screen.getByRole('button', { name: /View your result/i })).toBeTruthy()
    // Never silently restarted.
    expect(api.screeningStart).toHaveBeenCalledTimes(1)
  })
})
