# NeuroScreen — Voice Pitch & Prosodic Acoustic Analysis

**Document Version:** 1.0.0  
**Service Implementation:** `backend/app/services/pitch_service.py`  
**Frontend Interactive Lab:** `frontend/components/doctor/voice-pitch-view.tsx` (`/doctor/voice-analysis`)  
**API Endpoint:** `POST /api/voice/pitch-analysis`

---

## 1. Acoustic Methodology: Fundamental Frequency ($F_0$) Extraction

Fundamental frequency ($F_0$) is the physical rate of vocal fold vibration during voiced speech phonation.

### Extraction Pipeline
1. **Multi-Format Audio Decoding:** Converts incoming WAV, MP3, WebM, OGG, or FLAC bytes into mono 32-bit floating point arrays sampled at 16 kHz.
2. **YIN Pitch Tracking (`librosa.yin`):**
   - Estimates $F_0$ within the human vocal pitch range ($f_{\min} = 65\text{ Hz}$, $f_{\max} = 500\text{ Hz}$) with hop length 512 frames (~32 ms resolution).
   - Computes cumulative mean normalized difference function to minimize pitch octave errors.
3. **Voiced/Unvoiced Speech Discrimination:**
   - Frame RMS energy calculation:
     $$\text{RMS}_i = \sqrt{\frac{1}{K}\sum_{k=1}^K x_{i,k}^2}$$
   - Energy threshold: $\theta = \max(0.003, 0.08 \times \max(\text{RMS}))$.
   - A frame is flagged as **voiced** if $\text{RMS}_i > \theta$ and $65 \le F_0 \le 500\text{ Hz}$.
4. **Spectral Autocorrelation Fallback:**
   - If Librosa YIN encounters non-standard audio headers, the engine executes direct autocorrelation across 30ms windowed frames.

---

## 2. Calculated Prosodic & Temporal Metrics

| Metric | Units | Mathematical Definition | Clinical Relevance |
| :--- | :---: | :--- | :--- |
| **Minimum Pitch ($F_0\min$)** | Hz | $\min(\{F_{0,i} \mid \text{voiced}_i\})$ | Lower bound of vocal register |
| **Maximum Pitch ($F_0\max$)** | Hz | $\max(\{F_{0,i} \mid \text{voiced}_i\})$ | Upper pitch inflection limit |
| **Mean Pitch ($F_0\text{mean}$)** | Hz | $\frac{1}{V}\sum F_{0,i}$ | Habitual speaking pitch |
| **Median Pitch ($F_0\text{med}$)** | Hz | $\text{Median}(\{F_{0,i}\})$ | Central tendency robust to pitch outliers |
| **Pitch Std Dev ($F_0\text{std}$)** | Hz | $\sqrt{\frac{1}{V}\sum (F_{0,i} - F_{0\text{mean}})^2}$ | **Vocal variability / prosodic flatlining indicator** |
| **Pitch Range** | Hz | $F_0\max - F_0\min$ | Dynamic vocal pitch span |
| **Speech Duration** | sec | Total duration $\times$ voiced frame ratio | Phonation time |
| **Silence Duration** | sec | Total duration $\times (1 - \text{voiced ratio})$ | Unvoiced pauses and hesitations |
| **Silence Ratio** | — | Silence duration / Total duration | Proportion of hesitations/pauses |
| **Pause Frequency** | /min | Contiguous unvoiced intervals $\ge 200\text{ms}$ normalized per min | Psychomotor hesitation frequency |
| **Speaking Rate** | syl/s | Syllables detected via spectral peaks per speech sec | Articulation rate |
| **RMS Energy** | float | Overall signal root-mean-square amplitude | Vocal projection and loudness |

---

## 3. Clinical Research Significance: Acoustic Markers of Psychomotor Slowing

Depression frequently affects the somatic neuromuscular systems controlling respiration, vocal fold tension, and laryngeal motor coordination.

Key research observations in clinical literature:
1. **Reduced Pitch Variability ($F_0\text{std} < 20\text{ Hz}$):** Characterized as "monotone" or "flattened" speech.
2. **Elevated Silence Ratio ($> 30\%$):** Longer pauses between clauses reflecting psychomotor hesitation and cognitive processing latency.
3. **Reduced Articulation Rate ($< 3.2\text{ syllables/sec}$):** Slurred or slowed speech cadence.

---

## 4. Time-Series Pitch Contour Visualization

The API returns a decimate-sampled `pitch_contour` array:
```json
{
  "time_sec": 1.25,
  "pitch_hz": 174.5,
  "voiced": true
}
```
The frontend plots this time series as an interactive SVG line chart with break intervals indicating speech pauses.
