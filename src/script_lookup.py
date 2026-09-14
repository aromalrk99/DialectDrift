"""
Manual reference table: writing system per language.

NOT a dataset pulled from any API — this is a small, hand-compiled lookup
based on well-known public facts about which script each language uses
(the kind of thing you'd find on Wikipedia/Omniglot). Included because
URIEL/lang2vec does not provide a "script" feature.
"""

SCRIPTS = {
    "ces": "Latin",
    "swe": "Latin",
    "nor": "Latin",
    "dan": "Latin",
    "ron": "Latin",
    "ukr": "Cyrillic",
    "ben": "Bengali",
    "urd": "Arabic",
    "pan": "Gurmukhi",
    "tam": "Tamil",
    "tel": "Telugu",
    "mar": "Devanagari",
    "jav": "Latin",
    "mya": "Myanmar",
    "khm": "Khmer",
    "amh": "Geez",
}

def same_script(lang1: str, lang2: str) -> int:
    """Returns 1 if both languages use the same script, else 0."""
    return int(SCRIPTS.get(lang1) == SCRIPTS.get(lang2))