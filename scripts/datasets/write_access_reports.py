"""Write the controlled-access reports for DAIC-WOZ, RSDD and SMHD.

All facts below were read from the official pages on 2026-11-13 and are quoted
with their source URL. Nothing here submits a form, accepts an agreement, or
creates credentials — the reports record what the project owner must do.

Outputs:
  data/reports/daic_access_report.json
  data/reports/rsdd_access_report.json
  data/reports/smhd_access_report.json

Usage:
  .venv/Scripts/python scripts/datasets/write_access_reports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EXIT_OK, REPORTS, get_logger, utcnow, write_json  # noqa: E402

STATUS = "ACCESS_PENDING"

DAIC_EMAIL_DRAFT = """\
Subject: Institutional support for DAIC-WOZ access (research use, NeuroScreen project)

Dear <Guide>,

I am working on the NeuroScreen project, a research system that studies
multimodal indicators associated with depression-related symptoms (text, voice,
facial features). The work is experimental and explicitly non-clinical.

To move from single-source experiments to a participant-level multimodal
benchmark, I would like to request access to the DAIC-WOZ database from the
USC Institute for Creative Technologies. The corpus is distributed only to
academic and non-profit researchers, and the request form requires an
institutional academic e-mail address together with a signed data usage
agreement.

I would be grateful if you could:

1. Confirm that you support this request as my project guide/supervisor.
2. Confirm which institutional e-mail address and affiliation should be used.
3. Confirm whether the department can sign the data usage agreement, and who is
   authorised to sign it on the institution's behalf.
4. Confirm whether an ethics/IRB statement is required for secondary analysis of
   de-identified interview data at our institution.

I will prepare the form and the agreement for your review and will not submit
anything without your approval. No data has been downloaded or requested yet.

Kind regards,
<Student name>
<Programme, institution>
<Institutional e-mail>
"""

DAIC = {
    "generated_at": utcnow(),
    "dataset_name": "daic_woz",
    "status": STATUS,
    "official_source": "https://dcapswoz.ict.usc.edu/",
    "citation": "Gratch et al. (LREC 2014); DeVault et al. (AAMAS 2014)",
    "license": "controlled-access research agreement (not an open licence)",
    "language": "en",
    "participants_reported": 189,
    "sessions_reported": 189,
    "session_length": "7-33 minutes, average 16 minutes",
    "modalities": "text transcripts, participant audio (WAV), facial features, questionnaire responses",
    "data_form": "raw audio + transcripts + extracted facial features (COVAREP/CLNF-style); video available under the same agreement",
    "questionnaire": {
        "primary": "PHQ-8",
        "scoring": "sum of 8 items (0-24); the standard PHQ-8 cut-off is >= 10 for moderate depression, "
                   "but the exact label convention must be taken from the released label files",
        "note": "Not interchangeable with EATD SDS bands.",
    },
    "access_requirements": [
        "Academic or non-profit researcher affiliation",
        "Institutional academic e-mail address (requests from personal addresses are not accepted)",
        "Completed, signed and submitted data request form (Apply Now on the official site)",
        "Signed data usage agreement covering non-redistribution and secure storage",
    ],
    "redistribution": "Prohibited. Data must not be shared, published, or re-hosted; only derived aggregate results may be reported.",
    "prohibited_actions_taken": [
        "No authentication bypass attempted",
        "No download attempted",
        "No institutional credentials fabricated",
        "No application submitted",
        "No agreement accepted on the owner's behalf",
    ],
    "application_checklist": [
        "Confirm supervisor support and named signatory for the agreement",
        "Use the institutional e-mail address in the request",
        "Prepare the study description: purpose, modalities used, non-clinical framing",
        "State storage plan (encrypted local disk, access limited to the project owner)",
        "State retention and deletion plan at project end",
        "Submit the form and agreement only after the owner's approval",
        "Archive the signed agreement with the project records",
    ],
    "email_draft_to_guide": DAIC_EMAIL_DRAFT,
    "research_role": "Primary paired text/audio/visual benchmark for multimodal experiments",
    "estimated_size": "tens of GB (audio + video + features); exact figure only available after approval",
}

FORMS_NOTE = "Data request form published by Georgetown IR Lab: https://goo.gl/forms/XCqdkTlNLgbNGOXj1 (form submission requires manual action)"

RSDD = {
    "generated_at": utcnow(),
    "dataset_name": "rsdd",
    "status": STATUS,
    "official_source": "https://ir.cs.georgetown.edu/resources/rsdd",
    "mirror_page": "https://georgetown-ir-lab.github.io/emnlp17-depression",
    "citation": "Yates, Cohan, Goharian (EMNLP 2017)",
    "license": "research data usage agreement (no open licence)",
    "language": "en",
    "participants_reported": 116484,
    "users": {
        "diagnosed": 9210,
        "control": 107274,
        "training": {"diagnosed": 3070, "control": 35753},
        "validation": {"diagnosed": 3070, "control": 35746},
        "testing": {"diagnosed": 3070, "control": 35775},
    },
    "modalities": "text only (Reddit posts, user-level)",
    "data_form": "per-user post collections with original train/validation/test partitions",
    "label_definition": {
        "positive": "user self-reported a depression diagnosis (high-precision pattern such as 'I was diagnosed with' "
                    "within 80 characters of a depression mention), crowdsource-verified",
        "negative": "matched control users with no mental-health posts, matched by subreddit distribution (Hellinger distance)",
        "clinical_claim": False,
        "weak_label": True,
        "derived_from_self_report": True,
    },
    "access_requirements": [
        "Signed data usage agreement",
        "No attempt to contact any user in the dataset",
        "No attempt to deanonymise or identify any user",
        "No linking dataset users to any external information",
        "No sharing of any data portion, including example posts, with third parties",
        FORMS_NOTE,
    ],
    "redistribution": "Prohibited by the agreement; results may be reported in aggregate only.",
    "application_checklist": [
        "Read and sign the data usage agreement (owner's signature required)",
        "Confirm the project's no-contact/no-deanonymisation practice in writing",
        "Store data locally on an encrypted volume, not in any cloud service",
        "Do not quote individual posts in reports or viva materials",
        "Record the signed agreement path in the project records",
    ],
    "research_role": "User-level text classification with an official participant-level split; external text generalisation",
    "privacy_notes": [
        "Posts may contain health information and identifying details",
        "Users are pseudonymous but potentially re-identifiable; treat all text as sensitive",
    ],
}

SMHD = {
    "generated_at": utcnow(),
    "dataset_name": "smhd",
    "status": STATUS,
    "official_source": "https://ir.cs.georgetown.edu/resources/smhd.html",
    "citation": "Cohan et al. (COLING 2018)",
    "license": "research data usage agreement (no open licence)",
    "language": "en",
    "participants_reported": 385476,
    "users_by_condition": {
        "ADHD": 10098, "Anxiety": 8783, "Autism": 2911, "Bipolar": 6434,
        "Depression": 14139, "Eating": 598, "OCD": 2336, "PTSD": 2894,
        "Schizophrenia": 1331, "Control": 335952,
    },
    "depression": {
        "users": 14139,
        "posts": 1272000,
        "splits": {"train": 2662, "dev": 2574, "test": 2611, "RC": 6292},
    },
    "modalities": "text only (Reddit posts, user-level, nine conditions + controls)",
    "data_form": "per-user post collections with official train/dev/test partitions",
    "label_definition": {
        "positive": "user self-reported a diagnosis of the condition (high-precision pattern, crowdsource-verified)",
        "negative": "matched control users",
        "clinical_claim": False,
        "weak_label": True,
        "note": "Nine conditions are not mutually exclusive categories of one variable; depression must be treated "
                "as a one-vs-control task, and emotion labels are never depression labels.",
    },
    "access_requirements": [
        "Signed data usage agreement",
        "No contact, deanonymisation, or cross-dataset linking of users",
        "No redistribution of posts or excerpts",
        FORMS_NOTE,
    ],
    "redistribution": "Prohibited by the agreement.",
    "application_checklist": [
        "Sign the data usage agreement (owner's signature required)",
        "Decide explicitly whether the comparison is depression-vs-control only",
        "Confirm local encrypted storage and no cloud upload",
        "Do not quote individual posts in reports",
    ],
    "research_role": "Large-scale user-level text research and external generalisation for the text branch",
    "privacy_notes": [
        "Substantially larger and more sensitive than RSDD; includes multiple mental-health conditions",
        "Never link SMHD users to RSDD, EATD, or any external account",
    ],
}


def main() -> int:
    log = get_logger("write_access_reports")
    for name, payload in (("daic_access_report", DAIC), ("rsdd_access_report", RSDD), ("smhd_access_report", SMHD)):
        path = write_json(REPORTS / f"{name}.json", payload)
        log.info("%s -> %s (%s)", payload["dataset_name"], path, payload["status"])
    log.info("no form submitted, no agreement accepted, no credentials created")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
