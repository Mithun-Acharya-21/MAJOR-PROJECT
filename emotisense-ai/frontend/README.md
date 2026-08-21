# EmotiSense AI — Frontend (Phase 19)

React 18 + TypeScript + Vite single-page application.

## Pages

| Route | Page | Description |
|---|---|---|
| `/` | Dashboard | Health status, navigation cards |
| `/text` | Text Analysis | Consent gate → textarea → risk badge + tokens |
| `/voice` | Voice Analysis | Record or upload → quality badges + transcript |
| `/facial` | Facial Analysis | Upload image → face detection + emotion chart + Grad-CAM |
| `/multimodal` | Multimodal | Text + audio + image → fusion result + gate contributions |
| `/explain` | Explainability | Explanation of LIME, SHAP, Grad-CAM, gate views |
| `/suggest` | Suggestions | Wellness suggestion library + crisis resources |
| `/model` | Model Info | Version, readiness, 20-phase roadmap |
| `/ethics` | Ethics & Privacy | Limitations, privacy controls, crisis policy |

## Components

| Component | Purpose |
|---|---|
| `Disclaimer` | Safety banner — required on every model-output screen |
| `RiskBadge` | Colour-coded risk tier with score and uncertainty |
| `ConsentGate` | Explicit per-modality consent — must be granted before any processing |
| `EmotionChart` | Horizontal bar chart of emotion probabilities |
| `SuggestionList` | Policy-engine suggestions with crisis-first styling |
| `QualityBadge` | Quality-score pill for audio/face validation results |
| `NavBar` | Accessible top navigation |

## Hooks

| Hook | Purpose |
|---|---|
| `useAnalysis` | Generic async API call with loading/error/data state |
| `useAudioRecorder` | Browser MediaRecorder wrapper — never auto-starts |

## Development

```powershell
cd frontend
npm install
npm run dev        # starts at http://localhost:3000 (proxies /api to :8000)
npm run typecheck  # tsc --noEmit
npm run build      # production build → dist/
```

## Docker

```powershell
# From project root
docker compose up --build
# Backend: http://localhost:8000
# Frontend dev: http://localhost:3000
```

## Design decisions

- All consent is **explicit and per-modality** — no assumed consent.
- Disclaimer appears on **every screen** that shows model output.
- Crisis resources are **always visible** on the Suggestions page.
- Risk tier labels say "Concern" not "Diagnosis" or "Risk of depression".
- Grad-CAM is labelled "emotion model visualisation" not "depression evidence".
- `model_available: false` is shown honestly — no fabricated scores.
