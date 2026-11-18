# DAIC-WOZ Frontend Integration Documentation

---

## 1. User Interface Overview

The Next.js frontend integrates a dedicated research portal accessible via:
`http://localhost:3000/daic-woz`

### 1.1 Primary Components
- **`DaicWozResearchScreen` (`frontend/components/daic-woz/daic-woz-screen.tsx`):**
  - Live model checkpoint status banner (`/api/daic-woz/model/status`).
  - Interactive transcript input with sample clinical and control dialogue excerpts.
  - Vocal audio upload dropzone with openSMILE eGeMAPS integration indicators.
  - OpenFace kinematic stream summary preview.
  - Multimodal results card with calibrated risk gauge, per-modality breakdown, and research disclaimers.

### 1.2 Clinical Communication & Language Standards
- **Approved Output Text:** *"Elevated screening indicators"* or *"Lower-risk indicators"*.
- **Forbidden Diagnostic Claims:** *"Clinical diagnosis"*, *"Confirmed major depressive disorder"*.
- Visible persistent disclaimer: *"This research prototype is not a diagnostic medical device. Results are experimental and should not replace professional assessment."*
