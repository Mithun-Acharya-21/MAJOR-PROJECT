# EmotiSense AI

# Complete Project Folder Structure

## Emotion-Aware & Explainable Deep Learning Model for Early Depression Detection




This is the full directory layout for the project — organized so text pipeline, facial pipeline, fusion, explainability, and reporting all stay cleanly separated. No code yet, just structure + purpose of each part.




```

depression-detection-project/

│

├── README.md

├── requirements.txt

├── .gitignore

├── LICENSE

│

├── data/

│   ├── raw/

│   │   ├── text/

│   │   │   ├── rsdd/

│   │   │   ├── erisk/

│   │   │   └── twitter_kaggle/

│   │   └── face/

│   │       ├── fer2013/

│   │       ├── raf_db/

│   │       └── ck_plus/

│   │

│   ├── processed/

│   │   ├── text/

│   │   │   ├── train.csv

│   │   │   ├── val.csv

│   │   │   └── test.csv

│   │   └── face/

│   │       ├── train/

│   │       ├── val/

│   │       └── test/

│   │

│   └── external/

│       └── slang_emoji_dictionaries/

│

├── notebooks/

│   ├── 01_data_exploration_text.ipynb

│   ├── 02_data_exploration_face.ipynb

│   ├── 03_text_model_prototyping.ipynb

│   ├── 04_face_model_prototyping.ipynb

│   ├── 05_fusion_experiments.ipynb

│   └── 06_explainability_experiments.ipynb

│

├── src/

│   ├── __init__.py

│   │

│   ├── text_pipeline/

│   │   ├── __init__.py

│   │   ├── preprocessing.py

│   │   ├── tokenizer.py

│   │   ├── deberta_encoder.py

│   │   ├── bilstm_attention.py

│   │   └── train_text_model.py

│   │

│   ├── face_pipeline/

│   │   ├── __init__.py

│   │   ├── face_detection.py

│   │   ├── preprocessing.py

│   │   ├── cnn_model.py

│   │   └── train_face_model.py

│   │

│   ├── fusion/

│   │   ├── __init__.py

│   │   ├── feature_fusion.py

│   │   ├── classifier_head.py

│   │   └── train_fusion_model.py

│   │

│   ├── explainability/

│   │   ├── __init__.py

│   │   ├── lime_explainer.py

│   │   ├── shap_explainer.py

│   │   └── visualization.py

│   │

│   ├── evaluation/

│   │   ├── __init__.py

│   │   ├── metrics.py

│   │   └── baseline_models.py

│   │

│   └── utils/

│       ├── __init__.py

│       ├── config_loader.py

│       ├── logger.py

│       └── seed.py

│

├── configs/

│   ├── text_config.yaml

│   ├── face_config.yaml

│   ├── fusion_config.yaml

│   └── training_config.yaml

│

├── models/

│   ├── checkpoints/

│   │   ├── text_model/

│   │   ├── face_model/

│   │   └── fusion_model/

│   └── final/

│       └── best_model.pt

│

├── results/

│   ├── metrics/

│   │   ├── text_only_results.json

│   │   ├── face_only_results.json

│   │   └── fusion_results.json

│   ├── confusion_matrices/

│   ├── roc_curves/

│   └── explainability_outputs/

│       ├── lime_examples/

│       └── shap_examples/

│

├── app/

│   ├── streamlit_app.py

│   ├── inference.py

│   └── static/

│       └── sample_images/

│

├── reports/

│   ├── phase1_synopsis/

│   │   └── Lasttt_synopsis.pdf

│   ├── phase1_ppt/

│   │   └── final_depression_detection.pptx

│   ├── literature_review/

│   │   └── papers_summary.md

│   ├── final_report/

│   │   └── project_report.docx

│   └── presentation/

│       └── final_ppt.pptx

│

├── docs/

│   ├── architecture_diagram.png

│   ├── dataset_notes.md

│   ├── limitations.md

│   └── viva_qa_prep.md

│

└── tests/

    ├── test_preprocessing.py

    ├── test_text_model.py

    ├── test_face_model.py

    └── test_fusion.py

```




---




## Folder-by-Folder Explanation




### `data/`

Holds all datasets, kept separate from code so raw data is never accidentally modified.

- **`raw/`** — original, untouched downloaded datasets (text datasets like RSDD/eRisk, face datasets like FER-2013/RAF-DB). Never edit these directly.

- **`processed/`** — cleaned, tokenized, split train/val/test versions ready to feed into models. This is what your training scripts actually read.

- **`external/`** — supporting reference files, like your slang/abbreviation-to-plain-English dictionary used in text preprocessing.




### `notebooks/`

Jupyter notebooks for exploration and prototyping — messy, iterative work happens here before it gets turned into clean scripts in `src/`. Numbered in the order you'd actually work through them (explore data → prototype text model → prototype face model → fusion → explainability).




### `src/`

The actual reusable Python source code, organized by pipeline stage:

- **`text_pipeline/`** — everything for the text branch: cleaning/preprocessing, DeBERTa-v3 tokenization and encoding, the BiLSTM+Attention layer, and the training script for this branch alone.

- **`face_pipeline/`** — everything for the facial branch: face detection/cropping, image preprocessing, the CNN model, and its training script.

- **`fusion/`** — combines outputs of both branches: the fusion layer itself (concat/cross-attention/gated), the final classification head, and the training script for the full multimodal model.

- **`explainability/`** — LIME and SHAP wrappers that generate explanations for predictions, plus code to visualize them (word highlight, face heatmaps).

- **`evaluation/`** — shared metrics calculation (accuracy, F1, ROC-AUC) and baseline model implementations (Naïve Bayes, SVM, plain BERT) for comparison tables.

- **`utils/`** — shared helper code used across the whole project: loading config files, logging, setting random seeds for reproducibility.




### `configs/`

YAML files storing hyperparameters (learning rate, batch size, epochs, model paths) separately from code, so you can change settings without touching Python files — makes experiments easier to track and reproduce.




### `models/`

Where trained model weights are saved.

- **`checkpoints/`** — intermediate saves during training (useful if training crashes or you want to compare epochs).

- **`final/`** — the final chosen best-performing model, ready for the demo app.




### `results/`

All output metrics and visualizations get saved here — this is what you'll pull screenshots/tables from for your report.

- **`metrics/`** — JSON files with accuracy/precision/recall/F1 for each model variant.

- **`confusion_matrices/`** and **`roc_curves/`** — saved plot images.

- **`explainability_outputs/`** — saved LIME/SHAP visualizations you'll include in your report and demo.




### `app/`

The demo application — a simple interface where someone types text or uploads a photo and sees the prediction + explanation. Kept separate from training code since it only needs the final trained model, not the whole training pipeline.




### `reports/`

All submission documents for your course, organized by phase — includes your existing synopsis PDF and Phase I PPT, literature review notes, and (later) your final report and final presentation.




### `docs/`

Supporting documentation: architecture diagrams, notes on dataset limitations, an explicit limitations write-up (useful for the "what if they're lying/joking" discussion), and a viva prep Q&A file your team can study from together.




### `tests/`

Basic test scripts to check that preprocessing functions, model layers, and the fusion logic work correctly before you invest time in full training runs — catches bugs early.




---




## Why This Structure Works for Your Team of 4




- **Member 1** works almost entirely inside `src/text_pipeline/`

- **Member 2** extends the same folder (BiLSTM+Attention lives there too) — pairs naturally with Member 1

- **Member 3** works almost entirely inside `src/face_pipeline/`

- **Member 4** works across `src/fusion/`, `src/explainability/`, `src/evaluation/`, and `app/`




Everyone commits to the same GitHub repo but touches different folders — minimizes merge conflicts and makes it obvious in your viva who built what.




---




*Next step, whenever you're ready: we can start filling in actual code, folder by folder — I'd suggest starting with `src/text_pipeline/preprocessing.py` since every other module depends on clean data.* make an ui for these project

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/3a9af6d5-3d5f-4ba9-8bc5-828cfdc265c5).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
