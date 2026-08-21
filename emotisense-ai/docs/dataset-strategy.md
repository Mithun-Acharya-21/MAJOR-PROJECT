# Phase 2 dataset strategy and compatibility audit

## Decision

The system separates **depression-risk learning** from **emotion representation learning**.

- DAIC-WOZ is the primary candidate for multimodal depression-risk fine-tuning and evaluation. The official USC page describes 189 Wizard-of-Oz sessions, including participant audio, transcripts, facial-feature information, questionnaire responses, and video; access is restricted to academic/non-profit researchers using an approved request. [DAIC-WOZ official access page](https://dcapswoz.ict.usc.edu/)
- E-DAIC is an extended candidate, subject to the same approval process and manual review. It is not assumed to be a drop-in replacement until the obtained documentation and labels are audited. [E-DAIC manual](https://dcapswoz.ict.usc.edu/wwwedaic/E-DAIC%20Manual.pdf)
- eRisk can support a text-only baseline or text representation study. Its labels and task protocol concern user-writing histories and early-risk detection; it cannot supply voice or face signals for fusion. [CLEF eRisk](https://erisk.irlab.org/)
- FER-2013 is facial-expression pretraining only. It must not be used to teach a depression classifier.
- RAVDESS is acted emotion pretraining only. Its official record provides audio-only, audio-video, and video-only versions with emotion/intensity labels; its license is CC BY-NC-SA 4.0. [RAVDESS official record](https://zenodo.org/records/1188976)
- CREMA-D is acted, audio-visual emotion pretraining only. Its official repository describes 7,442 clips from 91 actors, labelled for six emotions and intensity. [CREMA-D official repository](https://github.com/CheyneyComputerScience/CREMA-D)

## Explicitly forbidden construction

```text
DAIC-WOZ participant depression label
  + RAVDESS actor voice
  + FER-2013 face
  = invalid synthetic multimodal depression record
```

Cross-dataset emotion pretraining is acceptable only before fine-tuning the corresponding encoder on aligned depression data. It cannot create valid depression labels.

## Acquisition gates

1. Obtain supervisor/institution approval where required.
2. Complete the dataset provider’s agreement using an academic email where mandated.
3. Record the exact version, source, access date, and license in a data card.
4. Place files only in gitignored `data/raw/<dataset-id>/`.
5. Run the manifest validator before preprocessing.
6. Do not download or redistribute restricted data through this repository.

## Split policy

Create a frozen participant-level split manifest immediately after eligibility checks. All sessions for one participant remain in exactly one partition. Deduplication, normalization fit statistics, augmentation, and hyperparameter selection must use training/validation data only. The final test manifest is write-protected and not used until model selection is complete.

## Dataset risks

DAIC-WOZ/E-DAIC access, cohort size, interview-domain bias, and label scope may limit claims. FER-2013 labels can be noisy and both RAVDESS and CREMA-D use acted emotion, which limits naturalistic generalization. This project will report these limitations rather than translating emotion labels into clinical claims.

