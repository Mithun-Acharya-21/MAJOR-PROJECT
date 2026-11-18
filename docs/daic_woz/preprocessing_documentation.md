# DAIC-WOZ Preprocessing & Feature Extraction Specification

---

## 1. Text Preprocessing

- **Speaker Isolation:** Transcripts are parsed to filter rows where `speaker.lower() == 'participant'`. Interviewer questions are discarded.
- **Normalization:** Consecutive whitespace and newline characters are collapsed. Punctuation is retained for n-gram context.
- **Vocabulary Extraction:** Sublinear TF-IDF vectorization with unigram and bigram features ($N=2,500$) and English stopword elimination.

---

## 2. Audio Preprocessing & Acoustic Parameterization

- **Segmentation:** Timestamps from participant transcript turns delineate voice activity windows.
- **Acoustic Feature Extraction:** Parameterized using the **Geneva Minimalistic Acoustic Parameter Set extended (eGeMAPS v02)** via openSMILE:
  - Frequency Parameters: Pitch ($F_0$), jitter, formant frequencies ($F_1, F_2, F_3$).
  - Energy Parameters: Shimmer, loudness, Harmonics-to-Noise Ratio (HNR).
  - Spectral Parameters: Alpha ratio, Hammarberg index, spectral slope, MFCCs 1–4.
  - Functional Aggregation: Mean, standard deviation, percentiles across active speech frames $\rightarrow$ fixed 88-dimensional vector per participant.

---

## 3. Visual Kinematics Aggregation

- **Action Units (AUs):** Dynamic activation and intensity of facial movements associated with psychological affect:
  - Upper face: Inner brow raiser (AU01), brow lowerer (AU04).
  - Lower face: Lip corner puller (AU12), lip corner depressor (AU15), chin raiser (AU17).
- **Statistical Aggregation:** Frame-level measurements are aggregated into participant-level functionals (mean and standard deviation) $\rightarrow$ 40-dimensional feature representation.
