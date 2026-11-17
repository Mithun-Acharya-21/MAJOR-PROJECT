# Phase 4 — access-controlled dataset status

Audit date: 2026-11-14 · Sources re-checked against the official pages and the
repository's own access reports (`data/reports/{daic,rsdd,smhd}_access_report.json`).

**Nothing was requested, submitted, accepted or downloaded in this phase.** No
form was filled, no agreement signed, no credential created, no access control
touched. Everything below is preparation and status, not acquisition.

## 1. Classification

| Dataset | Acquired | Access preparation | Application | Data files present | Owner action required |
|---|---|---|---|---|---|
| DAIC-WOZ | **No** | Complete | **Not submitted** | No (`data/raw/audio/daic_woz`, `data/raw/visual/daic_woz` are empty) | **Yes** — sign + submit |
| RSDD | **No** | Complete | **Not submitted** | No (`data/raw/text/rsdd` holds only `.gitkeep`) | **Yes** — sign + submit |
| SMHD | **No** | Complete | **Not submitted** | No (`data/raw/text/smhd` is empty) | **Yes** — sign + submit |

Status for all three: **ACCESS_PENDING**. **Not acquired. Not verified.**

## 2. DAIC-WOZ

| | |
|---|---|
| Official source | https://dcapswoz.ict.usc.edu/ |
| Citation | Gratch et al. (LREC 2014); DeVault et al. (AAMAS 2014) |
| Licence | Controlled-access research agreement — not an open licence |
| Reported participants | 189 participants, 189 sessions, 7–33 min (mean 16 min) **[reported by the distributor]** |
| Modalities | Interview audio (WAV), transcripts, extracted facial features (COVAREP / CLNF-style), PHQ-8 questionnaire scores; video under the same agreement |
| Label | **PHQ-8**, sum of 8 items (0–24). The common screening cut-off is ≥ 10, but the released label convention must be read from the distributed label files, not assumed |
| Access requirements | Academic/non-profit affiliation; **institutional academic e-mail** (personal addresses refused); completed + signed data request form; signed data usage agreement |
| Redistribution | Prohibited. No sharing, publishing or re-hosting; only aggregate results |

**Why it matters:** it is the only shortlisted corpus that provides *paired*
text + audio + facial data with a questionnaire label at participant level,
which is what the project's fusion claim needs. It is therefore the primary
multimodal benchmark — and it is unavailable until the owner acts.

**Owner actions:** confirm supervisor support and the named signatory; confirm
the institutional e-mail; confirm whether an ethics/IRB statement is needed for
secondary analysis of de-identified interviews; then submit. A ready-to-send
draft e-mail is in `data/reports/daic_access_report.json` →
`email_draft_to_guide`.

**Not verified:** the exact archive size, file inventory and label-file format —
these are only visible after approval, so no size or count is claimed here.

## 3. RSDD

| | |
|---|---|
| Official source | https://ir.cs.georgetown.edu/resources/rsdd (mirror: https://georgetown-ir-lab.github.io/emnlp17-depression) |
| Citation | Yates, Cohan, Goharian (EMNLP 2017) |
| Licence | Research data usage agreement — no open licence |
| Reported users | 116,484 users (9,210 diagnosed / 107,274 control); official split 3,070/35,753 train, 3,070/35,746 validation, 3,070/35,775 test **[reported]** |
| Modalities | Text only — Reddit post collections, **user-level** |
| Label | User *self-reported* a depression diagnosis, detected by a high-precision self-disclosure pattern and crowdsource-verified; controls are subreddit-matched users with no mental-health posts. **Weak label. Not a clinical diagnosis** |
| Access requirements | Signed DUA; no contacting, deanonymising or cross-linking users; no sharing any data portion (not even example posts); request form at https://goo.gl/forms/XCqdkTlNLgbNGOXj1 |
| Redistribution | Prohibited by the agreement; aggregate results only |

**Why it matters:** it is the only text corpus with an official **user-level**
split, so it can support an external generalisation test for the text branch
without the tweet-level leakage problem that the Harvard corpus has.

**Owner actions:** sign the DUA; confirm the no-contact/no-deanonymisation
commitment in writing; store locally on an encrypted volume; record the signed
agreement path in the project records.

## 4. SMHD

| | |
|---|---|
| Official source | https://ir.cs.georgetown.edu/resources/smhd.html |
| Citation | Cohan et al. (COLING 2018) |
| Licence | Research data usage agreement — no open licence |
| Reported users | 385,476 users across nine conditions + controls; Depression 14,139 users / 1,272,000 posts, official split 2,662 / 2,574 / 2,611 (+6,292 RC) **[reported]** |
| Modalities | Text only — Reddit post collections, **user-level** |
| Label | Same self-report + crowdsource-verification scheme as RSDD. Nine conditions are **not** mutually exclusive levels of one variable: depression must be used as a one-vs-control task |
| Access requirements | Signed DUA; no contact / deanonymisation / cross-dataset linking; no redistribution; same request form |
| Redistribution | Prohibited by the agreement |

**Why it matters:** the largest text corpus available and the only one large
enough to train a text encoder from scratch, but its labels are the weakest of
the set and its sensitivity the highest.

**Owner actions:** sign the DUA; decide explicitly that the task is
depression-vs-control only (never "which of nine conditions"); confirm encrypted
local storage; never link SMHD users to RSDD, EATD or any external account.

## 5. What is actually missing before these can be used

1. A signed DUA per dataset (owner's signature — cannot be delegated).
2. An approved request from an institutional academic address.
3. Storage and processing policy: encrypted volume, no cloud upload, no quoted
   posts in reports, retention/deletion plan at project end.
4. Ethics/IRB position for secondary analysis of mental-health text (owner to
   confirm with the institution).

None of these are technical blockers that this phase can remove, so nothing in
Phase 4 stands on them.

## 6. Explicitly not done

- No application form submitted for DAIC-WOZ, RSDD or SMHD.
- No data usage agreement accepted or signed.
- No credentials, API keys or institutional accounts created.
- No scraping, mirroring or workaround of the access controls.
- No data file for these three datasets exists in the repository.
