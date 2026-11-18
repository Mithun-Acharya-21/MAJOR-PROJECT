# DAIC-WOZ REST API Documentation

Base URL: `http://localhost:8000/api/daic-woz`

---

## 1. Endpoints Reference

### 1.1 Model Status
`GET /model/status`
- **Description:** Returns runtime checkpoint availability and hardware device.
- **Response:**
  ```json
  {
    "text_model_loaded": false,
    "audio_model_loaded": false,
    "visual_model_loaded": false,
    "fusion_model_loaded": false,
    "device": "cuda",
    "dataset_acquisition_status": "PENDING_OFFICIAL_DOWNLOAD",
    "active_checkpoints": { ... }
  }
  ```

### 1.2 Benchmark Metrics
`GET /model/metrics`
- **Description:** Returns validation summary and target benchmark performance.

### 1.3 Multimodal Analysis
`POST /analyze`
- **Request Body:**
  ```json
  {
    "transcript_text": "I feel tired and have trouble sleeping at night.",
    "visual_features": [0.1, 0.45, 0.08, ...]
  }
  ```
- **Response:**
  ```json
  {
    "screening_indicator": "Lower-risk indicators",
    "calibrated_probability": 0.32,
    "risk_level": "LOW",
    "modality_breakdown": {
      "text": { "modality": "text", "available": true, "calibrated_probability": 0.28 },
      "visual": { "modality": "visual", "available": true, "calibrated_probability": 0.36 }
    },
    "research_disclaimer": "Experimental research screening prototype..."
  }
  ```

### 1.4 Single Modality Analysis
- `POST /analyze/text`: Processes transcript string payload.
- `POST /analyze/visual`: Processes OpenFace 40-dimensional feature array.
- `POST /analyze/audio`: Multipart file upload for WAV/MP3 recordings.
