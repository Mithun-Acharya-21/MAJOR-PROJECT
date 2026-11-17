# EATD-Corpus — provenance and redistribution audit

Audit date: 2026-11-14 · Local copy: `data/eatd/` (1,806 files, 162 participant
folders) · Derived artifacts: `data/processed/eatd/`

**Nothing in EATD was modified by this audit.** No file, manifest, split or
label was changed. The findings below are recorded, not applied.

## 1. Provenance chain — verified

| Link | Evidence |
|---|---|
| Upstream repository | `https://github.com/Fancy-Block/EATD-Corpus` |
| Repository description | "An Emotional Audio-Textual Corpus" |
| Acquisition route | the repository's own `.zip`; README also lists a Baidu mirror (`pan.baidu.com/s/1hXPcjPcqV8ltEi255AjLdQ`, code `2022`) |
| Local layout | `train/` 83 participant folders, `validation/` 79 participant folders |
| Local file count | 1,806 files |
| Corpus description (README) | 162 volunteers "who received counseling"; 3 prompted sessions each (positive / neutral / negative) |
| Local session count | 486 = 162 × 3, consistent |

The local copy matches the upstream description of the layout
(`label.txt`, `new_label.txt`, `{positive,neutral,negative}.wav`, `_out.wav`
variants, `.txt` transcripts). Provenance is therefore **established**: this is
the EATD-Corpus v1 release, not a derivative or a re-upload.

## 2. Licence — UNRESOLVED

Checked on 2026-11-14 against the authoritative source:

| Check | Result |
|---|---|
| GitHub `license` metadata on `Fancy-Block/EATD-Corpus` | **`null`** |
| `GET /repos/Fancy-Block/EATD-Corpus/license` | **HTTP 404** — no `LICENSE`, `COPYING` or `LICENCE` file exists |
| Terms of use / data agreement in the README | **none** |
| Citation or DOI requirement stated | **none** in the repository |
| Any statement permitting redistribution | **none found** |
| Any statement restricting to research use | **none found** |

**Conclusion: the corpus ships with no licence whatsoever.** An absent licence is
not a permissive licence: under default copyright the authors retain all rights,
and no permission to copy, redistribute or republish has been granted in writing.

The only statement of intended use is indirect — it is published as an academic
corpus for "Emotional Audio-Textual" research, and it accompanies a peer-reviewed
publication — which supports a *research use* reading but does **not** authorise
**redistribution**.

## 3. Redistribution decision

| Question | Answer | Basis |
|---|---|---|
| May the project use EATD locally for research? | **Probably, but not formally granted** — unresolved | no licence, only implied academic intent |
| May the project redistribute the raw corpus (repo, release, paper appendix, demo bundle)? | **No** | default copyright; no grant exists |
| May derived aggregate results (metrics, tables, trained weights) be published? | **Probably** — results are the project's own output | standard practice; not a redistribution of the corpus |
| May transcripts or audio excerpts be quoted in slides, viva or a demo? | **No, not safely** | that is redistribution of corpus content |
| Should the corpus stay private/local? | **Yes** | unresolved licence + special-category personal data |

Practical rules adopted for this project:

1. `data/eatd/` stays out of version control and out of any public artifact.
2. No audio, transcript, participant id→label row, or excerpt leaves the local
   machine, including in the report, slides, or a live demo.
3. Published numbers are aggregate only (macro-F1, AUC, participant counts).
4. Model weights may be published only as aggregate learning artifacts, never
   alongside corpus content.
5. The demo (backend/frontend) must never be seeded with EATD content.

## 4. Privacy classification

EATD audio and transcripts are recordings of counselling sessions from
identifiable volunteers. Even though folder ids are pseudonymous, the audio is a
direct voice recording and the transcripts are verbatim speech — this is
**special-category personal data** under GDPR-style regimes. It must be treated
as private research data regardless of what the licence turns out to be.

## 5. Label provenance — a real finding

The corpus README states the authors' own class split:

> Training set contains data from 83 volunteers (**19 depressed** and 64
> non-depressed).
> Validation set contains data from 79 volunteers (**11 depressed** and 68
> non-depressed).

That is **30 depressed / 132 control (18.5 %)**. The project's local binary
label uses `SDS index ≥ 50`, which yields **53 depressed / 109 control (32.7 %)**.
The two disagree by 23 participants.

Testing every candidate threshold against the 162 locally verified SDS indices:

| Rule | Depressed | Control | Matches the authors? |
|---|---|---|---|
| SDS index ≥ 50 | 53 | 109 | No |
| **SDS index ≥ 53** | **30** | **132** | **Yes — exact match** |
| SDS index ≥ 55 | 25 | 137 | No |
| SDS index ≥ 60 | 15 | 147 | No |

**The authors' split is reproduced exactly by the standard Chinese-norm SDS
cut-off of 53, not by 50.** The README does not state which rule was used, so
this is a strong inference rather than a quoted fact — but it is an exact match
on 162 participants and no other threshold in the range reproduces it.

Consequences recorded (not applied):

- The local binary label is not the corpus authors' binary label. Any EATD
  number must state which threshold produced it.
- `docs/eatd_dataset_card.md` explains the 53/109 split as "published
  descriptions report 108/54, one-participant difference unexplained". That
  note compares against a *different published figure*; the repository README's
  30/132 is the closer and more likely definition. The discrepancy is now
  explained: **wrong threshold, not a one-participant data difference.**
- **Changing the threshold is deliberately not done here.** It would invalidate
  the locked test split (24 participants, 8 positive) and every Phase 2 baseline
  trained against the ≥50 rule. Switching is an owner decision that requires
  retraining, and belongs to a later phase.

## 6. Status summary

| Item | Status |
|---|---|
| Provenance | **Verified** — official EATD-Corpus v1 |
| Research use | **Unresolved** — implied only, never granted in writing |
| Redistribution | **Not permitted** — no licence exists |
| Privacy | **Special-category personal data** — keep local |
| Label instrument | **Verified** — Zung SDS, self-report, not a clinical diagnosis |
| Label threshold vs authors | **Mismatch, explained** — authors ≈ ≥53, project uses ≥50 |
| Action taken on EATD | **None** — read-only audit |

## 7. Owner actions

1. Decide the EATD label threshold (≥50 as used in Phase 2, or ≥53 to match the
   authors). Do not do both silently.
2. If any EATD content is ever to be shown or shared, obtain written permission
   from the corpus authors first.
3. Keep `data/eatd/` untracked and unshared.
