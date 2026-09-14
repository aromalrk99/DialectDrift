"""
Manual reference table: writing system per language.

NOT a dataset pulled from any API — this is a small, hand-compiled lookup
based on well-known public facts about which script each language uses
(the kind of thing you'd find on Wikipedia/Omniglot). Included because
URIEL/lang2vec does not provide a "script" feature.
"""

SCRIPTS = {
    "eng": "Latin",
    "spa": "Latin",
    "fra": "Latin",
    "deu": "Latin",
    "ita": "Latin",
    "por": "Latin",
    "nld": "Latin",
    "rus": "Cyrillic",
    "pol": "Latin",
    "cmn": "Han",
    "jpn": "Japanese",       # Kanji + Hiragana/Katakana, treated as its own system
    "kor": "Hangul",
    "arb": "Arabic",
    "hin": "Devanagari",
    "tur": "Latin",
    "vie": "Latin",          # Latin script with diacritics (Chữ Quốc Ngữ)
    "tha": "Thai",
    "swh": "Latin",
    "fin": "Latin",
    "hun": "Latin",
    "ell": "Greek",
    "heb": "Hebrew",
    "ind": "Latin",
    "tgl": "Latin",
}

def same_script(lang1: str, lang2: str) -> int:
    """Returns 1 if both languages use the same script, else 0."""
    return int(SCRIPTS.get(lang1) == SCRIPTS.get(lang2))