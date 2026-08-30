# Dataset Notes — EmotiSense AI

## Text Datasets

### RSDD (Reddit Self-reported Depression Dataset)
- **Source**: Reddit posts from r/depression and r/SuicideWatch
- **Labels**: Self-reported depression (binary)
- **Size**: ~3,000 posts
- **Notes**: Real-world social media data; informal language, slang, emojis

### eRisk / CLEF
- **Source**: eRisk shared task at CLEF
- **Labels**: Depression risk (binary)
- **Size**: ~5,000+ documents
- **Notes**: Longitudinal data; some entries span weeks of posting history

### Twitter Depression Dataset (Kaggle)
- **Source**: Kaggle dataset of labeled tweets
- **Labels**: Depression / non-depression (binary)
- **Size**: ~20,000 tweets
- **Notes**: Short text; high noise; requires careful preprocessing

## Facial Expression Datasets

### FER-2013
- **Source**: Kaggle Facial Expression Recognition Challenge
- **Labels**: 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)
- **Size**: ~35,000 images (48×48 grayscale)
- **Notes**: Used as proxy for depression-related facial cues (sadness, flat affect)

### RAF-DB (Real-world Affective Faces Database)
- **Source**: Beijing Normal University
- **Labels**: 7 basic emotions + valence/arousal
- **Size**: ~30,000 images
- **Notes**: More diverse demographics and real-world conditions

### CK+ (Extended Cohn-Kanade Dataset)
- **Source**: Carnegie Mellon University
- **Labels**: 7 expressions + AU codes
- **Size**: ~600 sequences, 327 labeled
- **Notes**: Lab-controlled; high quality; limited diversity

## Preprocessing Pipeline

1. **Text**: Unicode normalization → URL/mention removal → slang expansion → emoji → text → lowercase → whitespace cleanup
2. **Face**: OpenCV Haar cascade detection → center crop with padding → resize 224×224 → ImageNet normalization
3. **Labels**: Mapped to binary (0=non-depressed, 1=depressed) regardless of original encoding

## Limitations

- Datasets are proxy signals, not clinical diagnoses
- FER-2013 emotions ≠ depression; we use sadness/flat affect as indicators
- Class imbalance exists in most datasets
- Demographic representation varies across datasets
