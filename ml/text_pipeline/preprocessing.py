"""
Text Preprocessing — Clean and normalise text for the DeBERTa encoder.

Handles:
  - Unicode normalisation
  - URL and mention replacement
  - Slang / abbreviation expansion
  - Emoji → text conversion
  - Excess whitespace cleanup

Exports:
  preprocess_text — raw text → cleaned text
  load_slang_dictionary — load slang/emoji mappings from external files
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Built-in slang / abbreviation dictionary
# ---------------------------------------------------------------------------

_BUILTIN_SLANG: Dict[str, str] = {
    "idk": "i do not know",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "tbh": "to be honest",
    "afk": "away from keyboard",
    "brb": "be right back",
    "smh": "shaking my head",
    "fwiw": "for what it is worth",
    "iirc": "if i recall correctly",
    "irl": "in real life",
    "nvm": "never mind",
    "omw": "on my way",
    "rn": "right now",
    "ngl": "not going to lie",
    "fr": "for real",
    "lmao": "laughing out loud",
    "rofl": "rolling on the floor laughing",
    "tyt": "take your time",
    "wbu": "what about you",
    "wtf": "what the heck",
    "ttyl": "talk to you later",
    "hmu": "hit me up",
    "bc": "because",
    "b/c": "because",
    "w/": "with",
    "w/o": "without",
    "cus": "because",
    "cos": "because",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "do not know",
    "lemme": "let me",
    "gimme": "give me",
    "ight": "alright",
    "ayy": "hey",
    "lil": "little",
    "dw": "do not worry",
    "ily": "i love you",
    "haha": "haha",
    "hehe": "hehe",
    "hmm": "hmm",
    "ugh": "ugh",
    "sigh": "sigh",
}

# Common emoji → emotion text mapping
_EMOJI_MAP: Dict[str, str] = {
    "😀": " happy ",
    "😃": " happy ",
    "😄": " happy ",
    "😁": " smiling ",
    "😆": " laughing ",
    "😅": " nervous laugh ",
    "🤣": " laughing hard ",
    "😂": " crying laughing ",
    "🙂": " slightly smiling ",
    "🙃": " upside down ",
    "😉": " winking ",
    "😊": " blushing happy ",
    "😇": " innocent ",
    "🥰": " loving ",
    "😍": " heart eyes love ",
    "🤩": " starstruck ",
    "😘": " kissing ",
    "😗": " kissing ",
    "😚": " kissing closed eyes ",
    "😙": " kissing smiling ",
    "🥲": " sad smile ",
    "😋": " yummy ",
    "😛": " playful ",
    "😜": " wacky ",
    "🤪": " crazy ",
    "😝": " tongue out ",
    "🤑": " money mouth ",
    "🤗": " hugging ",
    "🤭": " oops ",
    "🤫": " shushing ",
    "🤔": " thinking ",
    "🤐": " zipped mouth ",
    "🤨": " raised eyebrow ",
    "😐": " neutral face ",
    "😑": " expressionless ",
    "😶": " no mouth ",
    "😏": " smirk ",
    "😒": " unamused ",
    "🙄": " eye roll ",
    "😬": " grimace ",
    "🤥": " lying ",
    "😌": " relieved ",
    "😔": " pensive sad ",
    "😪": " sleepy ",
    "🤤": " drooling ",
    "😴": " sleeping ",
    "😷": " sick mask ",
    "🤒": " sick fever ",
    "🤕": " injured ",
    "🤢": " nauseous ",
    "🤮": " vomiting ",
    "🥵": " hot ",
    "🥶": " cold ",
    "🥴": " woozy ",
    "😵": " dizzy ",
    "🤯": " mind blown ",
    "🤠": " cowboy ",
    "🥳": " partying ",
    "🥸": " disguised ",
    "😎": " cool sunglasses ",
    "🤓": " nerd ",
    "🧐": " monocle ",
    "😕": " confused ",
    "😟": " worried ",
    "🙁": " slightly frowning ",
    "☹️": " frowning ",
    "😮": " surprised mouth open ",
    "😯": " hushed surprised ",
    "😲": " astonished ",
    "😳": " flushed embarrassed ",
    "🥺": " pleading sad ",
    "😦": " frowning mouth open ",
    "😧": " anguished ",
    "😨": " fearful ",
    "😰": " anxious sweat ",
    "😥": " sad but relieved ",
    "😢": " crying sad ",
    "😭": " loudly crying ",
    "😱": " screaming fear ",
    "😖": " confounded ",
    "😣": " persevering ",
    "😞": " disappointed sad ",
    "😓": " downcast sweat ",
    "😩": " weary tired ",
    "😫": " tired exhausted ",
    "🥱": " yawning bored ",
    "😤": " angry triumphant ",
    "😡": " angry rage ",
    "😠": " angry ",
    "🤬": " cursing angry ",
    "💔": " broken heart sad ",
    "❤️": " love heart ",
    "🧡": " orange heart love ",
    "💛": " yellow heart love ",
    "💚": " green heart love ",
    "💙": " blue heart love ",
    "💜": " purple heart love ",
    "🖤": " black heart ",
    "🤍": " white heart ",
    "💯": " hundred percent perfect ",
    "🔥": " fire hot ",
    "✨": " sparkles ",
    "🌟": " glowing star ",
    "💪": " strong flex ",
    "👍": " thumbs up approve ",
    "👎": " thumbs down disapprove ",
    "🙏": " please thanks prayer ",
    "😢": " crying sad ",
    "🥺": " pleading sad ",
    "😞": " disappointed sad ",
    "😔": " pensive sad ",
    "💀": " skull dead ",
    "☠️": " skull dead ",
}


# ---------------------------------------------------------------------------
# Slang dictionary loader
# ---------------------------------------------------------------------------

def load_slang_dictionary(path: Optional[str] = None) -> Dict[str, str]:
    """Load slang/abbreviation dictionary from a JSON file.

    Falls back to the built-in dictionary if the file doesn't exist.

    Args:
        path: Path to a JSON file mapping abbreviations to expansions.
              If None, uses the built-in dictionary.

    Returns:
        Dict mapping lowercase abbreviations to their expansions.
    """
    slang = dict(_BUILTIN_SLANG)

    if path is not None:
        p = Path(path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    user_slang = json.load(f)
                if isinstance(user_slang, dict):
                    slang.update({k.lower(): v for k, v in user_slang.items()})
            except (json.JSONDecodeError, OSError):
                pass

    return slang


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(
    text: str,
    slang_dict: Optional[Dict[str, str]] = None,
    expand_slang: bool = True,
    convert_emoji: bool = True,
) -> str:
    """Normalise text while preserving emotionally meaningful content.

    Steps:
        1. Unicode normalisation (NFKC)
        2. Lowercase
        3. Replace URLs with <URL> placeholder
        4. Replace @mentions with <USER> placeholder
        5. Convert emojis to text descriptions (if enabled)
        6. Expand slang/abbreviations (if enabled)
        7. Clean excess whitespace

    Args:
        text: Raw input text.
        slang_dict: Optional custom slang dictionary. Uses built-in if None.
        expand_slang: Whether to expand slang abbreviations.
        convert_emoji: Whether to convert emojis to text.

    Returns:
        Cleaned text string.
    """
    if not text or not text.strip():
        return ""

    # Unicode normalisation
    text = unicodedata.normalize("NFKC", text).strip()

    # Lowercase
    text = text.lower()

    # Replace URLs
    text = re.sub(r"https?://\S+|www\.\S+", " <URL> ", text)

    # Replace @mentions
    text = re.sub(r"@\w+", " <USER> ", text)

    # Convert emojis to text
    if convert_emoji:
        for emoji, replacement in _EMOJI_MAP.items():
            text = text.replace(emoji, replacement)

    # Remove remaining non-ASCII characters (after emoji conversion)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Expand slang
    if expand_slang:
        if slang_dict is None:
            slang_dict = load_slang_dictionary()
        words = text.split()
        expanded = []
        for word in words:
            clean_word = re.sub(r"[^a-z0-9'/]", "", word.lower())
            if clean_word in slang_dict:
                expanded.append(slang_dict[clean_word])
            else:
                expanded.append(word)
        text = " ".join(expanded)

    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
