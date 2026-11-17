"""
Adaptive Question Service — Adaptive screening interview engine.

Selects the next screening question based on:
    - information gain (expected new structured information)
    - symptom coverage value
    - uncertainty reduction (targets low-confidence dimensions)
    - importance of missing information
    - question redundancy (exact + semantic dedup, info-already-known guard)

Conceptual score (internal only — never exposed to users):
    question_score = (information_gain + coverage_value + uncertainty_reduction)
                     / redundancy_penalty

Adaptive stopping:
    MIN_QUESTIONS = 8, MAX_QUESTIONS = 15.
    Stops when required dimensions have sufficient coverage, uncertainty is
    low enough, and important gaps are resolved — or MAX_QUESTIONS is reached.

Progress display is adaptive: "Questions answered: N" + "Assessment progress: P%"
computed from real coverage — never a fake "Question 8 of 8".
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.app.session_manager import (
    SCREENING_DIMENSIONS,
    SEVERITY_DIMENSIONS,
    ScreeningSession,
)

logger = logging.getLogger("emotiondep.adaptive_questions")

MIN_QUESTIONS = 8
MAX_QUESTIONS = 15

# ---------------------------------------------------------------------------
# Question-specific options
# ---------------------------------------------------------------------------
# Every question carries its OWN option list — options are never a shared
# universal scale ("Never/Sometimes/Often/Always") unless the question
# genuinely measures a frequency construct. Each option encodes structured
# EVIDENCE: severity (0-3) plus a ``signals`` map of dimension → severity
# contribution (0..3). Selecting an option is therefore a deterministic
# profile update, not a free-text guess.


def _opt(option_id: str, label: str, signals: Dict[str, int]) -> Dict[str, Any]:
    """Build one option dict; severity = max signal value (0 when no signals)."""
    severity = max(signals.values()) if signals else 0
    return {
        "id": option_id,
        "label": label,
        "severity": severity,
        "signals": signals,
    }


# Per-question option banks (keyed by question id). Keys must match
# QUESTION_BANK ids. Every option's evidence points at the question's own
# target dimensions — e.g. a sleep question's options move ``sleep`` severity,
# a concentration question's options move ``concentration`` severity.
QUESTION_OPTIONS: Dict[str, List[Dict[str, Any]]] = {
    # ---- Core symptom questions ----
    "q_mood_overall": [
        _opt("mood_stable", "My mood has been generally stable and positive.", {"mood": 0}),
        _opt("mood_fluctuating", "My mood has gone up and down more than usual.", {"mood": 1}),
        _opt("mood_low", "I have felt low or down on most days.", {"mood": 2}),
        _opt("mood_very_low", "I have felt persistently low, empty, or hopeless.", {"mood": 3}),
    ],
    "q_interest": [
        _opt("interest_normal", "I still enjoy the activities I normally enjoy.", {"loss_of_interest": 0}),
        _opt("interest_reduced", "Some activities feel less enjoyable than before.", {"loss_of_interest": 1}),
        _opt("interest_limited", "Only a few things still hold any enjoyment for me.", {"loss_of_interest": 2}),
        _opt("interest_none", "I have lost interest in almost everything.", {"loss_of_interest": 3}),
    ],
    "q_sleep_quality": [
        _opt("sleep_normal", "I sleep normally — falling and staying asleep is not a problem.", {"sleep": 0}),
        _opt("sleep_occasional", "I occasionally have difficulty falling or staying asleep.", {"sleep": 1}),
        _opt("sleep_frequent", "I frequently sleep badly — most nights are disturbed.", {"sleep": 2}),
        _opt("sleep_severe", "My sleep is seriously disrupted — barely any restful sleep.", {"sleep": 3}),
    ],
    "q_energy": [
        _opt("energy_normal", "My energy is about normal for me.", {"energy": 0}),
        _opt("energy_reduced", "I feel more tired than usual by the end of the day.", {"energy": 1}),
        _opt("energy_low", "I feel drained most of the day, even after rest.", {"energy": 2}),
        _opt("energy_exhausted", "I am exhausted constantly — even small tasks feel overwhelming.", {"energy": 3}),
    ],
    "q_appetite": [
        _opt("appetite_normal", "My appetite and eating habits are unchanged.", {"appetite": 0}),
        _opt("appetite_mild", "My appetite has changed slightly (more or less than usual).", {"appetite": 1}),
        _opt("appetite_marked", "My appetite has changed noticeably; eating patterns are different.", {"appetite": 2}),
        _opt("appetite_extreme", "I have lost my appetite almost completely (or eating far too much daily).", {"appetite": 3}),
    ],
    "q_concentration": [
        _opt("conc_normal", "I can concentrate on tasks about as well as ever.", {"concentration": 0}),
        _opt("conc_mild", "I get distracted a bit more easily than usual.", {"concentration": 1}),
        _opt("conc_moderate", "Staying focused is difficult — I often have to re-read or restart things.", {"concentration": 2}),
        _opt("conc_severe", "I can barely concentrate at all — even simple tasks defeat me.", {"concentration": 3}),
    ],
    "q_self_worth": [
        _opt("worth_normal", "I feel about as confident in myself as usual.", {"self_worth": 0}),
        _opt("worth_low", "I have been somewhat more self-critical than usual.", {"self_worth": 1}),
        _opt("worth_very_low", "I often feel disappointed in myself or not good enough.", {"self_worth": 2}),
        _opt("worth_none", "I feel worthless or like a burden.", {"self_worth": 3}),
    ],
    "q_guilt": [
        _opt("guilt_none", "I do not feel guilty or blame myself for things.", {"guilt": 0}),
        _opt("guilt_mild", "I occasionally feel guilty over small things.", {"guilt": 1}),
        _opt("guilt_moderate", "I often blame myself for things beyond my control.", {"guilt": 2}),
        _opt("guilt_severe", "I feel intense guilt almost constantly.", {"guilt": 3}),
    ],
    "q_psychomotor": [
        _opt("psych_normal", "My movement and pace feel normal.", {"psychomotor_changes": 0}),
        _opt("psych_mild", "I feel slightly restless or slowed at times.", {"psychomotor_changes": 1}),
        _opt("psych_moderate", "I notice clear slowing or restlessness most of the day.", {"psychomotor_changes": 2}),
        _opt("psych_severe", "I feel physically shut down or unable to stop pacing.", {"psychomotor_changes": 3}),
    ],
    "q_social": [
        _opt("social_normal", "I have been seeing or talking to people about as much as usual.", {"social_withdrawal": 0}),
        _opt("social_reduced", "I have been seeing people a bit less than usual.", {"social_withdrawal": 1}),
        _opt("social_very_reduced", "I mostly keep to myself and rarely reach out.", {"social_withdrawal": 2}),
        _opt("social_isolated", "I have cut myself off almost completely.", {"social_withdrawal": 3}),
    ],
    "q_functioning": [
        _opt("func_normal", "Daily tasks — work, study, home — are going normally.", {"daily_functioning": 0}),
        _opt("func_mild", "Some tasks take more effort than they should.", {"daily_functioning": 1}),
        _opt("func_moderate", "I am visibly falling behind; several areas are affected.", {"daily_functioning": 2}),
        _opt("func_severe", "I can barely manage basic daily activities.", {"daily_functioning": 3}),
    ],
    "q_duration": [
        _opt("dur_days", "A few days or less.", {}),
        _opt("dur_weeks", "Several weeks.", {}),
        _opt("dur_months", "Several months.", {}),
        _opt("dur_year", "A year or more.", {}),
    ],
    "q_frequency": [
        _opt("freq_rare", "Rarely or hardly ever.", {"frequency": 0}),
        _opt("freq_sometimes", "Sometimes — a few days a week.", {"frequency": 1}),
        _opt("freq_often", "Often — most days.", {"frequency": 2}),
        _opt("freq_daily", "Constantly — every day.", {"frequency": 3}),
    ],
    "q_severity_worst": [
        _opt("worst_mild", "Mild — noticeable but manageable.", {"severity": 1}),
        _opt("worst_moderate", "Moderate — clearly affects my days.", {"severity": 2}),
        _opt("worst_severe", "Severe — very hard to bear.", {"severity": 3}),
        _opt("worst_unbearable", "Unbearable at its worst moments.", {"severity": 3}),
    ],
    # ---- Follow-up questions ----
    "q_sleep_frequency": [
        _opt("sleepfreq_rare", "Rarely — maybe once a week or less.", {"sleep": 0, "frequency": 0}),
        _opt("sleepfreq_sometimes", "Some nights most weeks.", {"sleep": 1, "frequency": 1}),
        _opt("sleepfreq_often", "Most nights.", {"sleep": 2, "frequency": 2}),
        _opt("sleepfreq_daily", "Every single night.", {"sleep": 3, "frequency": 3}),
    ],
    "q_energy_duration": [
        _opt("energydur_days", "Only the last few days.", {}),
        _opt("energydur_weeks", "For several weeks.", {}),
        _opt("energydur_months", "For several months.", {}),
        _opt("energydur_year", "A year or more.", {}),
    ],
    "q_mood_duration": [
        _opt("mooddur_days", "Only a few days.", {}),
        _opt("mooddur_weeks", "For several weeks.", {}),
        _opt("mooddur_months", "For several months.", {}),
        _opt("mooddur_year", "A year or more.", {}),
    ],
 "q_social_impact": [
        _opt("socialimpact_no", "No more than usual — company feels fine.", {"social_withdrawal": 0, "energy": 0}),
        _opt("socialimpact_slightly", "Slightly more draining than before.", {"social_withdrawal": 1, "energy": 1}),
        _opt("socialimpact_very", "Much more draining; I avoid group settings.", {"social_withdrawal": 2, "energy": 2}),
        _opt("socialimpact_exhausting", "Completely exhausting — any contact wears me out.", {"social_withdrawal": 3, "energy": 3}),
    ],
    "q_functioning_work": [
        _opt("workfine", "I am working/studying and keeping up normally.", {"daily_functioning": 0}),
        _opt("workstruggle", "I am managing but it takes real effort.", {"daily_functioning": 1}),
        _opt("workbehind", "I am falling behind despite trying.", {"daily_functioning": 2}),
        _opt("worknone", "I have stopped or cannot currently manage work/study.", {"daily_functioning": 3}),
    ],
    "q_routine": [
        _opt("routine_regular", "Yes — regular sleep and meal times.", {"sleep": 0, "daily_functioning": 0}),
        _opt("routine_partial", "Somewhat — a few anchors but often irregular.", {"sleep": 1, "daily_functioning": 1}),
        _opt("routine_irregular", "No — my days are quite unstructured.", {"sleep": 2, "daily_functioning": 2}),
        _opt("routine_none", "I have no routine at all right now.", {"sleep": 3, "daily_functioning": 3}),
    ],
    "q_support": [
        _opt("support_yes", "Yes, I have people I can talk to.", {"social_withdrawal": 0}),
        _opt("support_some", "One or two people, when it matters.", {"social_withdrawal": 1}),
        _opt("support_limited", "Not really — I have nobody I confide in regularly.", {"social_withdrawal": 2}),
        _opt("support_no", "No one at all.", {"social_withdrawal": 3}),
    ],
    "q_activities": [
        _opt("act_several", "Several activities still give me enjoyment.", {"loss_of_interest": 0}),
        _opt("act_some", "A few things still help me relax.", {"loss_of_interest": 1}),
        _opt("act_few", "Almost nothing does anymore.", {"loss_of_interest": 2}),
        _opt("act_none", "Nothing interests me at all right now.", {"loss_of_interest": 3}),
    ],
    "q_morning": [
        _opt("morn_same", "Mornings feel about the same as the rest of the day.", {"mood": 0, "energy": 0}),
        _opt("morn_slightly_worse", "Mornings are slightly harder to get going.", {"mood": 1, "energy": 1}),
        _opt("morn_worse", "Mornings are clearly the worst part of my day.", {"mood": 2, "energy": 2}),
        _opt("morn_much_worse", "Mornings are unbearable compared to later in the day.", {"mood": 3, "energy": 3}),
    ],
}


def get_options(question: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the question-specific option list (never a shared scale)."""
    return QUESTION_OPTIONS.get(question["id"], [])


def find_option(
    question: Dict[str, Any], option_id: str
) -> Optional[Dict[str, Any]]:
    """Look up one option of a question by its id."""
    for opt in get_options(question):
        if opt["id"] == option_id:
            return opt
    return None

# Coverage considered "sufficient" for a dimension (0..1)
COVERAGE_SUFFICIENT = 0.8
# Dimensions required before adaptive stopping is allowed
REQUIRED_DIMENSIONS = ("mood", "loss_of_interest", "sleep", "energy", "daily_functioning", "duration")
# Mean confidence across dimensions below which we keep asking
CONFIDENCE_TARGET = 0.65
# Interview-time uncertainty threshold: stopping is allowed when the mean
# evidence uncertainty (1 - mean feature confidence over asked-about
# dimensions) falls at or below this value. This is the same information
# as CONFIDENCE_TARGET expressed as an uncertainty — deliberately ONE
# uncertainty notion at interview time (Phase 16).
UNCERTAINTY_STOP_THRESHOLD = 1.0 - CONFIDENCE_TARGET

# ---------------------------------------------------------------------------
# Question bank
# ---------------------------------------------------------------------------

# Each question: id, text, target dimensions, information it requests,
# and a focus hint used by the response analyzer.
QUESTION_BANK: List[Dict[str, Any]] = [
    {
        "id": "q_mood_overall",
        "text": "How have your mood and general spirits been over the past two weeks?",
        "target_dimensions": ["mood"],
        "information": "mood_baseline",
        "importance": 1.0,
    },
    {
        "id": "q_interest",
        "text": "Have you lost interest or pleasure in activities you usually enjoy?",
        "target_dimensions": ["loss_of_interest"],
        "information": "loss_of_interest",
        "importance": 1.0,
    },
    {
        "id": "q_sleep_quality",
        "text": "How has your sleep been recently?",
        "target_dimensions": ["sleep"],
        "information": "sleep_quality",
        "importance": 0.9,
    },
    {
        "id": "q_energy",
        "text": "How would you describe your energy levels during the day?",
        "target_dimensions": ["energy"],
        "information": "energy",
        "importance": 0.9,
    },
    {
        "id": "q_appetite",
        "text": "Have you noticed any changes in your appetite or eating habits?",
        "target_dimensions": ["appetite"],
        "information": "appetite",
        "importance": 0.7,
    },
    {
        "id": "q_concentration",
        "text": "How well have you been able to concentrate or focus on tasks?",
        "target_dimensions": ["concentration"],
        "information": "concentration",
        "importance": 0.8,
    },
    {
        "id": "q_self_worth",
        "text": "How have you been feeling about yourself and your self-worth?",
        "target_dimensions": ["self_worth", "guilt"],
        "information": "self_worth",
        "importance": 0.8,
    },
    {
        "id": "q_guilt",
        "text": "Do you find yourself feeling guilty or blaming yourself for things?",
        "target_dimensions": ["guilt"],
        "information": "guilt",
        "importance": 0.6,
    },
    {
        "id": "q_psychomotor",
        "text": "Have you noticed feeling physically slowed down, or the opposite — restless or fidgety?",
        "target_dimensions": ["psychomotor_changes"],
        "information": "psychomotor",
        "importance": 0.6,
    },
    {
        "id": "q_social",
        "text": "How much have you been seeing or talking to other people recently?",
        "target_dimensions": ["social_withdrawal"],
        "information": "social",
        "importance": 0.8,
    },
    {
        "id": "q_functioning",
        "text": "How have these difficulties been affecting your daily life — work, study, or household tasks?",
        "target_dimensions": ["daily_functioning"],
        "information": "functioning",
        "importance": 0.9,
    },
    {
        "id": "q_duration",
        "text": "How long have you been experiencing these difficulties?",
        "target_dimensions": ["duration"],
        "information": "duration",
        "importance": 1.0,
    },
    {
        "id": "q_frequency",
        "text": "How often do these experiences occur?",
        "target_dimensions": ["frequency"],
        "information": "frequency",
        "importance": 0.9,
    },
    {
        "id": "q_severity_worst",
        "text": "At their worst, how intense have these difficulties felt?",
        "target_dimensions": ["severity"],
        "information": "severity_worst",
        "importance": 0.7,
    },
    # --- Follow-up questions (target specific gaps) ---
    {
        "id": "q_sleep_frequency",
        "text": "How often have the sleep difficulties been happening?",
        "target_dimensions": ["sleep", "frequency"],
        "information": "sleep_frequency",
        "importance": 0.7,
    },
    {
        "id": "q_energy_duration",
        "text": "How long have you been feeling this lack of energy?",
        "target_dimensions": ["energy", "duration"],
        "information": "energy_duration",
        "importance": 0.7,
    },
    {
        "id": "q_mood_duration",
        "text": "How long have you been feeling this way about your mood?",
        "target_dimensions": ["mood", "duration"],
        "information": "mood_duration",
        "importance": 0.7,
    },
    {
        "id": "q_social_impact",
        "text": "Has being around other people felt more draining or stressful than usual?",
        "target_dimensions": ["social_withdrawal", "energy"],
        "information": "social_impact",
        "importance": 0.6,
    },
    {
        "id": "q_functioning_work",
        "text": "Are you currently working or studying, and how is that going?",
        "target_dimensions": ["daily_functioning"],
        "information": "functioning_context",
        "importance": 0.6,
    },
    {
        "id": "q_routine",
        "text": "Do you have a regular daily routine — consistent sleep and meal times?",
        "target_dimensions": ["sleep", "daily_functioning"],
        "information": "routine",
        "importance": 0.5,
    },
    {
        "id": "q_support",
        "text": "Do you have people around you that you can talk to about how you feel?",
        "target_dimensions": ["social_withdrawal"],
        "information": "social_support",
        "importance": 0.5,
    },
    {
        "id": "q_activities",
        "text": "Are there activities that still give you some enjoyment or help you relax?",
        "target_dimensions": ["loss_of_interest", "stress_management"],
        "information": "coping_activities",
        "importance": 0.5,
    },
    {
        "id": "q_morning",
        "text": "How do your mornings usually feel compared to the rest of the day?",
        "target_dimensions": ["mood", "energy"],
        "information": "diurnal_variation",
        "importance": 0.5,
    },
]

_QUESTION_BY_ID = {q["id"]: q for q in QUESTION_BANK}


class QuestionRedundancyError(Exception):
    """Raised internally when no non-redundant question can be selected."""


# ---------------------------------------------------------------------------
# Selection scoring
# ---------------------------------------------------------------------------

def _semantic_similarity(a: str, b: str) -> float:
    """Word-overlap Jaccard similarity (reuses the same approach as the
    existing transcript dedup in session_manager — no new infrastructure)."""
    wa = set(re.findall(r"[a-z']+", a.lower()))
    wb = set(re.findall(r"[a-z']+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def is_redundant(
    question: Dict[str, Any],
    asked_question_ids: List[str],
    question_semantics: List[str],
    features: Dict[str, Any],
    similarity_threshold: float = 0.55,
) -> Tuple[bool, str]:
    """Check the 4 redundancy rules from the spec.

    1. exact duplicate id
    2. semantic similarity with a previously asked question
    3. the required information already exists in the feature state
    4. reject redundant questions
    """
    qid = question.get("id", "")
    if qid in asked_question_ids:
        return True, "exact_duplicate"

    qtext = question.get("text", "")
    for prev in question_semantics:
        if _semantic_similarity(qtext, prev) >= similarity_threshold:
            return True, "semantic_duplicate"

    # Info-already-known: if every target dimension already has solid
    # coverage AND confidence, asking again adds nothing.
    targets = question.get("target_dimensions", [])
    if targets and all(
        features.get(t, {}).get("confidence", 0) >= 0.8
        for t in targets
        if t in SEVERITY_DIMENSIONS
    ):
        if all(
            t in features
            for t in targets
            if t in SEVERITY_DIMENSIONS
        ):
            return True, "information_already_known"

    return False, ""


def compute_dimension_coverage(features: Dict[str, Any]) -> Dict[str, float]:
    """Compute 0..1 coverage per dimension from current feature state."""
    coverage: Dict[str, float] = {}
    for dim in SCREENING_DIMENSIONS:
        info = features.get(dim)
        if info is None:
            coverage[dim] = 0.0
            continue
        if dim in SEVERITY_DIMENSIONS:
            confidence = float(info.get("confidence", 0.0))
            observations = int(info.get("observations", 1))
            coverage[dim] = min(1.0, confidence * (1.0 + 0.15 * (observations - 1)))
        else:
            coverage[dim] = min(1.0, float(info.get("confidence", 0.5)))
    return coverage


def score_question(
    question: Dict[str, Any],
    features: Dict[str, Any],
    dimension_coverage: Dict[str, float],
    missing_information: List[str],
    recent_dimensions: Optional[List[str]] = None,
) -> float:
    """Compute the adaptive selection score (internal only).

    question_score = (information_gain + coverage_value + uncertainty_reduction)
                     / redundancy_penalty

    ``recent_dimensions``: dimensions detected in the user's most recent
    response. Follow-up questions targeting them get a strong bonus so the
    next question directly follows up on what the user just described
    (e.g. after "I've been tired recently" → ask about the energy gap).
    """
    targets = question.get("target_dimensions", [])
    information = question.get("information", "")

    # --- information_gain: how much NEW info would this question provide? ---
    # (normalized per target so multi-target questions don't double-dip)
    info_gain = 0.0
    for t in targets:
        if t not in features:
            info_gain += 0.8
        else:
            conf = features[t].get("confidence", 0.0)
            info_gain += max(0.0, 0.8 - conf)  # diminishing returns on known info
    if targets:
        info_gain /= len(targets)
    # Follow-up bonus only when a real gap exists (missing info, or a targeted
    # dimension has evidence but low confidence)
    recent = set(recent_dimensions or [])
    if information in missing_information:
        info_gain += 0.5
    elif information.endswith("_frequency") or information.endswith("_duration"):
        focus = targets[0] if targets else None
        if focus and focus in recent:
            info_gain += 1.8  # follow up directly on the user's latest statement
        elif focus and focus in features and float(features[focus].get("confidence", 0)) < 0.8:
            info_gain += 0.5
        elif focus and focus not in features:
            info_gain -= 0.4  # follow-up about an unknown dimension is premature
    info_gain *= question.get("importance", 0.5)

    # --- coverage_value: fills gaps in important dimensions ---
    coverage_value = 0.0
    for t in targets:
        cov = dimension_coverage.get(t, 0.0)
        coverage_value += (1.0 - cov)
    if targets:
        coverage_value /= len(targets)
        # Required dimensions get a coverage bonus
        if any(t in REQUIRED_DIMENSIONS for t in targets):
            coverage_value += 0.4

    # --- uncertainty_reduction: targets low-confidence dimensions ---
    uncertainty_reduction = 0.0
    for t in targets:
        info = features.get(t)
        if info is None:
            uncertainty_reduction += 0.4
        else:
            uncertainty_reduction += max(0.0, CONFIDENCE_TARGET - float(info.get("confidence", 1.0)))
    if targets:
        uncertainty_reduction /= len(targets)

    # --- redundancy_penalty: repeats / known info ---
    redundancy = 1.0
    qtext = question.get("text", "")
    for prev in question.get("_prev_semantics", []):
        sim = _semantic_similarity(qtext, prev)
        if sim > 0.3:
            redundancy += sim

    return (info_gain + coverage_value + uncertainty_reduction) / redundancy


def select_next_question(
    session: ScreeningSession,
) -> Optional[Dict[str, Any]]:
    """Select the next question for the session, or None if done."""
    features = session.features
    coverage = session.dimension_coverage or compute_dimension_coverage(features)

    # Re-derive missing info from the last response analysis (if any)
    missing: List[str] = []
    if session.responses:
        missing = session.responses[-1].get("analysis", {}).get("missing_information", [])

    # Dimensions detected in the most recent response drive follow-up priority
    recent_dimensions: List[str] = []
    if session.responses:
        recent_dimensions = list(
            session.responses[-1].get("analysis", {}).get("dimensions", {}).keys()
        )

    best_q: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for question in QUESTION_BANK:
        redundant, _reason = is_redundant(
            question,
            session.asked_question_ids,
            session.question_semantics,
            features,
        )
        if redundant:
            continue
        # Score with redundancy context
        q = dict(question)
        q["_prev_semantics"] = session.question_semantics
        score = score_question(q, features, coverage, missing, recent_dimensions)
        if score > best_score:
            best_score = score
            best_q = question

    return best_q


def interview_uncertainty(session: "ScreeningSession") -> float:
    """Interview-time evidence uncertainty in [0, 1] (Phase 16).

    uncertainty = 1 - mean(confidence of structured features that have been
    asked about so far). This is the ONLY uncertainty measure used by the
    stopping logic. It is related to, but distinct from, the post-hoc
    fused-probability uncertainty in ``uncertainty_service`` (predictive
    entropy + modality disagreement), which is only computable once the
    multimodal pipeline runs at completion time.
    """
    confidences = [
        float(info.get("confidence", 0.0))
        for d, info in session.features.items()
        if d in SEVERITY_DIMENSIONS
    ]
    if not confidences:
        return 1.0
    return round(1.0 - sum(confidences) / len(confidences), 4)


def should_stop(session: "ScreeningSession") -> Tuple[bool, str]:
    """Adaptive stopping decision (Phase 13).

    ONE coverage source of truth: ``compute_dimension_coverage(features)``.

    Decision order:
        1. answered <  MIN_QUESTIONS        → continue (min_questions_not_reached)
        2. answered >= MAX_QUESTIONS        → stop    (max_questions_reached)
        3. required dimensions sufficiently covered
                                            → stop    (sufficient_information)
        4. interview uncertainty <= threshold AND duration/frequency gaps resolved
                                            → stop    (low_uncertainty_gaps_resolved)
        5. otherwise                        → continue

    The sufficient-information paths (3 and 4) are reachable: coverage now
    comes from feature confidence (one strong mention → 0.9 ≥ 0.8), not
    from the old fixed +0.4 increments that could never reach the 0.8
    threshold in a single mention.
    """
    answered = len(session.responses)
    if answered >= MAX_QUESTIONS:
        return True, "max_questions_reached"
    if answered < MIN_QUESTIONS:
        return False, "min_questions_not_reached"

    features = session.features
    coverage = compute_dimension_coverage(features)

    # 3. Required-dimension coverage path
    missing_required = [
        d for d in REQUIRED_DIMENSIONS if coverage.get(d, 0.0) < COVERAGE_SUFFICIENT
    ]
    if not missing_required:
        return True, "sufficient_information"

    # 4. Low-uncertainty + resolved-gaps path
    uncertainty = interview_uncertainty(session)
    gaps_resolved = "duration" in features and "frequency" in features
    if uncertainty <= UNCERTAINTY_STOP_THRESHOLD and gaps_resolved:
        return True, "low_uncertainty_gaps_resolved"

    # 5. Continue — report why
    if not gaps_resolved:
        return False, "information_gaps: duration/frequency"
    return False, f"required_dimensions_uncovered: {','.join(missing_required)}"


def get_progress(session: ScreeningSession) -> Dict[str, Any]:
    """Meaningful adaptive progress (never a fake 'Question N of M').

    progress = mean coverage across all dimensions, blended with a
    questions-answered component, reported as 0-100%.
    """
    coverage = compute_dimension_coverage(session.features)
    mean_coverage = sum(coverage.values()) / len(coverage) if coverage else 0.0

    required_cov = [coverage.get(d, 0.0) for d in REQUIRED_DIMENSIONS]
    required_mean = sum(required_cov) / len(required_cov) if required_cov else 0.0

    answered = len(getattr(session, "responses", []) or [])
    answered_component = min(1.0, answered / MIN_QUESTIONS)
    progress = 0.6 * required_mean + 0.25 * mean_coverage + 0.15 * answered_component
    progress_pct = int(round(max(0.0, min(1.0, progress)) * 100))

    return {
        "questions_answered": answered,
        "min_questions": MIN_QUESTIONS,
        "max_questions": MAX_QUESTIONS,
        "progress_percent": progress_pct,
        "required_dimensions_covered": all(
            coverage.get(d, 0.0) >= COVERAGE_SUFFICIENT for d in REQUIRED_DIMENSIONS
        ),
        "dimension_coverage": {k: round(v, 2) for k, v in coverage.items()},
        "missing_information": sorted(
            d for d, v in coverage.items() if v < COVERAGE_SUFFICIENT
        ),
    }


def get_question_by_id(question_id: str) -> Optional[Dict[str, Any]]:
    return _QUESTION_BY_ID.get(question_id)


# ---------------------------------------------------------------------------
# Shared questionnaire → probability mapping (Phase 12 — single definition)
# ---------------------------------------------------------------------------

def questionnaire_probability(features: Dict[str, Any]) -> Dict[str, float]:
    """Documented RULE-BASED mapping: structured features → screening score.

    severity-weighted mean (0..3) → logistic mapping centered at 1.5:
        P = sigmoid((mean_severity - 1.5) * 1.4), clipped to [0.05, 0.95]
        (0 → ~0.20, 1.5 → 0.50, 3 → ~0.80)

    This is a QUESTIONNAIRE-DERIVED RESEARCH SCORE — a hand-designed formula,
    NOT a trained ML probability and NOT a validated clinical instrument
    (no validated screening instrument is administered verbatim).
    Used by build_questionnaire_result in screening_service.
    """
    severity_dims = {
        d: info for d, info in features.items()
        if d not in ("duration", "frequency", "severity")
    }
    if not severity_dims:
        return {"probability": 0.5, "mean_severity": 0.0}

    weighted = 0.0
    weight_sum = 0.0
    for d, info in severity_dims.items():
        sev = float(info.get("severity", 0.0))
        conf = float(info.get("confidence", 0.5))
        weighted += sev * conf
        weight_sum += conf
    mean_severity = weighted / weight_sum if weight_sum else 0.0
    probability = 1.0 / (1.0 + math.exp(-(mean_severity - 1.5) * 1.4))
    probability = min(max(probability, 0.05), 0.95)
    return {"probability": probability, "mean_severity": round(mean_severity, 3)}


def build_question_payload(question: Dict[str, Any]) -> Dict[str, Any]:
    """Public question payload for the frontend (no internal scoring fields).

    Includes the question-SPECIFIC option list. Options differ per question
    because each question measures a different construct (sleep options
    describe sleep, energy options describe energy, ...). Free-text answers
    remain supported: ``options`` is empty for questions without a bank.
    """
    return {
        "id": question["id"],
        "text": question["text"],
        "target_dimensions": list(question.get("target_dimensions", [])),
        "options": [
            {"id": o["id"], "label": o["label"]}
            for o in get_options(question)
        ],
    }
