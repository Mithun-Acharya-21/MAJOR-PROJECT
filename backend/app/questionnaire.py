"""
Questionnaire configuration + train/inference-consistent preprocessing.

The questionnaire collects structured emotional/behavioural answers and
free text. The structured answers are mapped to a normalized *Chinese*
text template — the exact input distribution the EATD-trained text model
was trained on (Chinese emotional statements). This same mapper is used
for every prediction, so there is no train/inference mismatch.

The free-text answer (typed by the user, typically English) is scored by
an English depression-lexicon heuristic (``proxy_scorer.score_text``) and
blended into the final signal with a small weight — the primary prediction
always comes from the trained model.

Stable question IDs (per master prompt §19): mood_01, interest_01, ...,
free_text_01, free_text_02.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Questionnaire definition — served to the frontend via GET /api/questions
# ---------------------------------------------------------------------------

QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "mood_01",
        "category": "Mood",
        "question": "How would you describe your overall mood recently?",
        "type": "single",
        "required": True,
        "options": [
            {"value": "very_positive", "label": "Very positive", "zh": "心情非常好"},
            {"value": "mostly_positive", "label": "Mostly positive", "zh": "心情大部分时候不错"},
            {"value": "mixed", "label": "Mixed", "zh": "心情时好时坏"},
            {"value": "often_low", "label": "Often low", "zh": "心情经常低落"},
            {"value": "very_low", "label": "Very low", "zh": "心情非常低落"},
        ],
    },
    {
        "id": "interest_01",
        "category": "Interest",
        "question": "How interested have you been in activities you normally enjoy?",
        "type": "single",
        "required": True,
        "options": [
            {"value": "very_interested", "label": "Very interested", "zh": "对平时喜欢的活动非常有兴趣"},
            {"value": "somewhat_interested", "label": "Somewhat interested", "zh": "对平时喜欢的活动有些兴趣"},
            {"value": "same_as_usual", "label": "About the same as usual", "zh": "对活动的兴趣和平时一样"},
            {"value": "less_interested", "label": "Less interested than usual", "zh": "对平时喜欢的活动兴趣减少了"},
            {"value": "almost_no_interest", "label": "Almost no interest", "zh": "对平时喜欢的活动几乎没有兴趣"},
        ],
    },
    {
        "id": "energy_01",
        "category": "Energy",
        "question": "How would you describe your energy during the day?",
        "type": "single",
        "required": True,
        "options": [
            {"value": "very_energetic", "label": "Very energetic", "zh": "精力非常充沛"},
            {"value": "normal", "label": "Normal", "zh": "精力正常"},
            {"value": "slightly_low", "label": "Slightly low", "zh": "精力稍微有点低"},
            {"value": "frequently_tired", "label": "Frequently tired", "zh": "经常感到疲惫"},
            {"value": "exhausted", "label": "Exhausted most of the time", "zh": "大部分时间都精疲力尽"},
        ],
    },
    {
        "id": "sleep_01",
        "category": "Sleep",
        "question": "How has your sleep been recently?",
        "type": "single",
        "required": True,
        "options": [
            {"value": "sleeping_well", "label": "Sleeping well", "zh": "睡眠很好"},
            {"value": "fairly_well", "label": "Fairly well", "zh": "睡眠还可以"},
            {"value": "some_trouble", "label": "Some trouble falling or staying asleep", "zh": "睡眠有些问题，入睡或保持睡眠有点困难"},
            {"value": "often_disturbed", "label": "Often disturbed", "zh": "经常睡不好"},
            {"value": "very_poor", "label": "Very poor", "zh": "睡眠非常差，很难入睡"},
        ],
    },
    {
        "id": "social_01",
        "category": "Social",
        "question": "How often have you wanted to spend time with other people recently?",
        "type": "single",
        "required": True,
        "options": [
            {"value": "more_than_usual", "label": "More than usual", "zh": "比平时更想和别人相处"},
            {"value": "about_the_same", "label": "About the same", "zh": "想和别人相处的时间和平时差不多"},
            {"value": "slightly_less", "label": "Slightly less", "zh": "稍微不太想和别人相处"},
            {"value": "much_less", "label": "Much less", "zh": "很不想和别人相处"},
            {"value": "prefer_alone", "label": "I usually prefer to be alone", "zh": "我通常更喜欢一个人待着"},
        ],
    },
    {
        "id": "concentration_01",
        "category": "Concentration",
        "question": "How easy is it for you to concentrate on everyday tasks?",
        "type": "single",
        "required": True,
        "options": [
            {"value": "very_easy", "label": "Very easy", "zh": "集中注意力非常容易"},
            {"value": "mostly_easy", "label": "Mostly easy", "zh": "大部分时候能集中注意力"},
            {"value": "sometimes_difficult", "label": "Sometimes difficult", "zh": "有时难以集中注意力"},
            {"value": "frequently_difficult", "label": "Frequently difficult", "zh": "经常难以集中注意力"},
            {"value": "very_difficult", "label": "Very difficult", "zh": "很难集中注意力"},
        ],
    },
    {
        "id": "motivation_01",
        "category": "Motivation",
        "question": "How difficult is it to start everyday activities?",
        "type": "single",
        "required": True,
        "options": [
            {"value": "very_easy", "label": "Very easy", "zh": "开始做日常事情非常容易"},
            {"value": "usually_easy", "label": "Usually easy", "zh": "通常比较容易开始做事"},
            {"value": "sometimes_difficult", "label": "Sometimes difficult", "zh": "有时很难开始做事"},
            {"value": "often_difficult", "label": "Often difficult", "zh": "经常很难开始做事情"},
            {"value": "very_difficult", "label": "Very difficult", "zh": "做什么事情都非常困难"},
        ],
    },
    {
        "id": "daily_01",
        "category": "Daily Life",
        "question": "How would you describe your daily activities lately?",
        "type": "single",
        "required": True,
        "options": [
            {"value": "fully_active", "label": "Fully active", "zh": "日常生活非常充实活跃"},
            {"value": "mostly_normal", "label": "Mostly normal", "zh": "日常生活基本正常"},
            {"value": "reduced", "label": "Somewhat reduced", "zh": "日常活动比以前少了"},
            {"value": "greatly_reduced", "label": "Greatly reduced", "zh": "日常活动比以前少了很多"},
            {"value": "almost_none", "label": "Almost none", "zh": "几乎不想做任何日常活动"},
        ],
    },
    {
        "id": "stress_01",
        "category": "Stress",
        "question": "How stressed have you been feeling recently?",
        "type": "single",
        "required": True,
        "options": [
            {"value": "not_stressed", "label": "Not stressed", "zh": "最近没什么压力"},
            {"value": "slightly_stressed", "label": "Slightly stressed", "zh": "最近有一点压力"},
            {"value": "moderately_stressed", "label": "Moderately stressed", "zh": "最近压力比较大"},
            {"value": "very_stressed", "label": "Very stressed", "zh": "最近压力很大"},
            {"value": "overwhelmed", "label": "Overwhelmed", "zh": "最近压力大到难以承受"},
        ],
    },
    {
        "id": "isolation_01",
        "category": "Wellbeing",
        "question": "How often have you felt lonely or isolated recently?",
        "type": "single",
        "required": True,
        "options": [
            {"value": "rarely", "label": "Rarely", "zh": "很少感到孤独"},
            {"value": "sometimes", "label": "Sometimes", "zh": "有时感到孤独"},
            {"value": "often", "label": "Often", "zh": "经常感到孤独"},
            {"value": "very_often", "label": "Very often", "zh": "常常感到孤独"},
            {"value": "almost_always", "label": "Almost always", "zh": "几乎总是感到孤独"},
        ],
    },
    {
        "id": "free_text_01",
        "category": "Open-ended",
        "question": "Tell us a little about how your recent days have been.",
        "type": "text",
        "required": False,
        "placeholder": "Anything you feel comfortable sharing…",
    },
    {
        "id": "free_text_02",
        "category": "Open-ended",
        "question": "Is there anything about your recent emotional state that you would like to describe?",
        "type": "text",
        "required": False,
        "placeholder": "Optional — describe how you have been feeling…",
    },
]

_QUESTION_BY_ID = {q["id"]: q for q in QUESTIONS}
_OPTION_ZH: Dict[str, Dict[str, str]] = {
    q["id"]: {o["value"]: o["zh"] for o in q.get("options", [])}
    for q in QUESTIONS
}
# Chinese label for each category (used in the template prefix)
_CATEGORY_ZH = {
    "Mood": "心情",
    "Interest": "兴趣",
    "Energy": "精力",
    "Sleep": "睡眠",
    "Social": "社交",
    "Concentration": "注意力",
    "Motivation": "动力",
    "Daily Life": "日常生活",
    "Stress": "压力",
    "Wellbeing": "孤独感",
}


def get_questions() -> List[Dict[str, Any]]:
    """Return the questionnaire config for the frontend (no zh mappings)."""
    return [
        {k: v for k, v in q.items() if k != "options"} | {"options": [
            {"value": o["value"], "label": o["label"]} for o in q.get("options", [])
        ]}
        if q["type"] == "single" else {k: v for k, v in q.items()}
        for q in QUESTIONS
    ]


def questionnaire_to_model_text(answers: Dict[str, Any]) -> str:
    """Convert structured questionnaire answers into the normalized text
    template consumed by the trained text model.

    Args:
        answers: {question_id: option_value_or_text, ...}

    Returns:
        A Chinese sentence-per-question template, e.g.
        "心情：经常低落。兴趣：对平时喜欢的活动兴趣减少了。……"
    """
    parts: List[str] = []
    for q in QUESTIONS:
        qid = q["id"]
        if q["type"] != "single" or qid not in answers:
            continue
        value = answers[qid]
        zh = _OPTION_ZH.get(qid, {}).get(str(value))
        if not zh:
            continue
        label = _CATEGORY_ZH.get(q["category"], q["category"])
        parts.append(f"{label}：{zh}")
    text = "。".join(parts)
    if text and not text.endswith("。"):
        text += "。"
    return text


def validate_answers(answers: Dict[str, Any]) -> Optional[str]:
    """Return an error message if required questions are unanswered."""
    missing = []
    for q in QUESTIONS:
        if q["type"] == "single" and q.get("required"):
            val = answers.get(q["id"])
            if val is None or str(val).strip() == "":
                missing.append(q["question"])
    if missing:
        return "Please answer the required questions: " + "; ".join(missing[:3])
    return None
