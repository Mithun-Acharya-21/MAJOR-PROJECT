# NeuroScreen — Doctor & Clinician Research Dashboard

**Document Version:** 1.0.0  
**Target Audience:** Clinical Researchers, Attending Psychiatrists, System Evaluators  
**Application URL:** `/doctor`  
**API Endpoints:** `/api/doctor/sessions`, `/api/doctor/sessions/{session_id}`, `/api/doctor/sessions/{session_id}/notes`, `/api/doctor/patient/{patient_id}/timeline`

---

## 1. System Role & Ethical Positioning

The NeuroScreen Doctor Dashboard is designed exclusively for **clinical research screening, longitudinal cohort monitoring, and multimodal model interpretability**.

> [!IMPORTANT]
> **Clinical Research Boundary:**
> NeuroScreen is an experimental decision-support prototype. It does **not** provide autonomous psychiatric diagnoses, nor does it replace the clinical judgment of qualified psychiatrists, psychiatric nurses, or licensed clinical psychologists.

### Responsible Language Mandate
All screens and API responses adhere strictly to responsible clinical nomenclature:
- **Never display:** "Depression Confirmed", "Depressed Patient", "Healthy / Non-Depressed".
- **Always display:** "Experimental screening result — requires professional interpretation", "Elevated screening indicator (score $\ge 0.50$)", "Lower-risk indicator (score $< 0.50$)".

---

## 2. Dashboard Architecture & Workflow

```mermaid
flowchart TD
    subgraph Client [Frontend - Next.js / TypeScript]
        DD["Doctor Dashboard (/doctor)"]
        SD["Session Detail View (/doctor/session/[id])"]
        MA["Model Analytics (/doctor/analytics)"]
        DI["Dataset Inventory (/doctor/datasets)"]
        VP["Voice Pitch Lab (/doctor/voice-analysis)"]
    end

    subgraph Backend [FastAPI Backend Service]
        SM["ScreeningSessionManager"]
        DR["Doctor Router (/api/doctor)"]
        PS["PitchService (Librosa YIN)"]
        ART["Verified Results Registry (results/)"]
    end

    DD -->|GET /api/doctor/sessions| DR
    SD -->|GET /api/doctor/sessions/{id}| DR
    SD -->|POST /api/doctor/sessions/{id}/notes| DR
    SD -->|GET /api/doctor/patient/{id}/timeline| DR
    MA -->|GET /api/doctor/analytics| DR
    DI -->|GET /api/doctor/datasets| DR
    VP -->|POST /api/voice/pitch-analysis| DR
    DR --> SM
    DR --> PS
    DR --> ART
```

---

## 3. Core Capabilities & UI Modules

### 3.1 Cohort Metric Overview
The top of the dashboard displays 4 summary cards updating in real time:
1. **Total Screened Sessions:** Total volume of completed or active participant screenings.
2. **Elevated Indicators:** Sessions where composite calibrated risk score meets or exceeds the empirical threshold of 0.50.
3. **Lower-Risk Indicators:** Sessions scoring below 0.50 without active crisis triggers.
4. **In Progress / Inconclusive:** Active sessions currently undergoing adaptive interview questions or awaiting complementary modality uploads.

### 3.2 Patient Screening Session Table
A filterable, searchable directory providing:
- **Patient Identifier:** Pseudonymized format (e.g. `PT-RESEARCH-101`) to protect participant privacy.
- **Timestamp:** Formatted local date and time of session initiation.
- **Modalities Used:** Real-time badges for Text, Voice, Facial affect, and Late Fusion.
- **Calibrated Score:** Displayed as a percentage alongside raw composite score.
- **Clinical Category:** Clearly color-coded badges for elevated vs lower-risk indicators.
- **Clinician Notes Count:** Indicator showing recorded qualitative notes from attending clinicians.
- **Actions:** Quick link to inspect comprehensive research breakdown at `/doctor/session/[id]`.

### 3.3 Dynamic Filters & Search
- **Search Query:** Matches against Patient ID, Session ID, or patient alias.
- **Status Filter:** Allows filtering by `All`, `Elevated`, `Lower-Risk`, and `Active / In Progress`.

---

## 4. Clinician Observation Notes & Persistence

Clinicians can record qualitative clinical observations, behavioral impressions, or research notes directly into the session record via:
```http
POST /api/doctor/sessions/{session_id}/notes
Content-Type: application/json

{
  "note": "Patient showed delayed vocal response latency and psychomotor slowing during sleep inquiry.",
  "clinician": "Attending Research Clinician"
}
```
Notes are timestamped, attributed, and displayed in chronological order on the session detail screen.

---

## 5. Security & Privacy Considerations

1. **De-identification:** All participant records in research fixtures use non-reversible research IDs (`PT-RESEARCH-101`, `PT-RESEARCH-204`, etc.).
2. **Session Storage:** Live screening sessions reside in memory via `screening_session_manager` with explicit expiry, preventing unauthorized disk harvesting.
3. **Auditability:** Clinician notes preserve immutable creation timestamps for clinical trial and IRB compliance.
