# DAIC-WOZ Dataset Documentation & Clinical Protocol

**Corpus:** Distress Analysis Interview Corpus - Wizard-of-Oz  
**Source Institution:** University of Southern California Institute for Creative Technologies (USC ICT)  
**Associated Benchmark:** Audio/Visual Emotion Challenge (AVEC 2016 / 2017)  

---

## 1. Study Protocol & Data Collection

The DAIC-WOZ database contains clinical interviews conducted by an animated virtual interviewer named **Ellie**, controlled by a human interviewer (Wizard-of-Oz protocol) located in an adjacent room.
- Total Participants: 189 individuals.
- Interview Duration: 7 to 33 minutes (mean: 16 minutes).
- Demographics: Diverse adult clinical and community cohort recruited in Los Angeles, CA.

---

## 2. Modality Representations

### 2.1 Text Transcripts (`*_TRANSCRIPT.csv`)
- Format: Tab-delimited / CSV with timestamps (`start_time`, `stop_time`), `speaker` (`Ellie` vs `Participant`), and `value` (spoken text).
- Crucial Preprocessing Rule: Interviewer prompts (`Ellie`) must be strictly stripped to avoid interviewer language bias; only participant speech is evaluated.

### 2.2 Audio Waveforms (`*_AUDIO.wav`)
- Format: 16 kHz uncompressed 16-bit PCM WAV.
- Content: Dialogue between Ellie and participant. Participant-only vocal segments are isolated via transcript timing boundaries.

### 2.3 Visual Features (`*_CLNF_features.txt` / OpenFace CSVs)
- Extracted using Constrained Local Neural Fields (CLNF) and OpenFace.
- Features include: 
  - Facial Action Units (AU01, AU02, AU04, AU06, AU07, AU10, AU12, AU14, AU15, AU17, AU23, AU25, AU28, AU45)
  - Head pose angular rotations (pitch, yaw, roll) and translations
  - Eye gaze 2D/3D vectors

---

## 3. Diagnostic Instrument: PHQ-8

The clinical target is derived from the **Patient Health Questionnaire 8-item (PHQ-8)**:
- Score Range: 0 to 24 (sum of 8 items scored 0 to 3).
- **Official AVEC Clinical Threshold:**
  $$\text{Binary Label } y = \begin{cases} 1 & \text{if } \text{PHQ-8} \ge 10 \text{ (Moderate to Severe Depressive Symptoms)} \\ 0 & \text{if } \text{PHQ-8} < 10 \text{ (None or Mild Symptoms)} \end{cases}$$
- This cut-off is validated by standard psychiatric literature (Kroenke et al., 2009).
