# Phase 3 data preparation protocol

## Scope

No raw clinical or user inputs are included in the repository. Preparation starts only after access approval and a completed data card.

## Input index

Create one approved metadata CSV per dataset version with these required fields:

```text
record_id, participant_id, session_id, dataset_id, dataset_version,
label, purpose, alignment
```

`content_hash` is optional but strongly recommended. It must be calculated locally from the approved record before this script runs; the script does not hash or copy raw data.

## Integrity gates

The preparation command rejects:

- missing or duplicate record identifiers;
- multiple dataset versions in one run;
- blank participant/session/label metadata;
- duplicate content hashes;
- multimodal depression rows without participant-session alignment;
- a participant with conflicting labels;
- any participant appearing in more than one resulting split.

## Execution

```powershell
cd C:\Users\mh129\majorProject\emotisense-ai
backend\.venv\Scripts\python.exe training\prepare_dataset.py `
  training\examples\approved-index-example.csv `
  --output-dir data\processed\manifests\example-v1
```

The output contains `train.csv`, `validation.csv`, `test.csv`, and `split-summary.txt`. These are metadata-only and should be retained with experiment records. Do not alter them after training begins.

## Next-stage preprocessing rule

Fit text normalization statistics, audio normalization parameters, image transforms, resampling decisions, and any imputers using the training partition only. Validation is used for selection; final test stays locked.

