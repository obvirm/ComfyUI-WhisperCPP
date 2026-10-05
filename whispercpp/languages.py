"""Language selector handling for the WhisperCPP node (issue #16).

ComfyUI validates combo-box inputs against the list returned by ``INPUT_TYPES``
BEFORE the node code runs, so a value that is not in the list never reaches
``transcribe()``. Other extensions (e.g. MiniMax Music Production Toolkit) push
their own labels such as ``日本語 (Japanese)`` or ``Multilingual / mixed`` into
dropdowns that share the widget name ``language``, and stale workflows keep
those values - the prompt then dies with::

    Value not in list: language: '日本語 (Japanese)' not in (list of length 203)

Everything language-related lives here so it can be unit-tested without the
ComfyUI runtime:

* :func:`language_options` - the values offered by the node dropdown
* :func:`resolve_language` - maps ANY selector (code, English name, native
  name, foreign label, auto synonyms) to a whisper.cpp language code, or to
  ``None`` for auto-detect. Returns a ``recognized`` flag so callers can warn
  about garbage instead of failing the prompt.
"""
import re

# Full language codes for whisper (100 languages) - whisper.cpp native only
WHISPER_LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian", "ko": "korean", "fr": "french",
    "ja": "japanese", "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan", "nl": "dutch",
    "ar": "arabic", "sv": "swedish", "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish",
    "vi": "vietnamese", "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech",
    "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian", "th": "thai",
    "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak", "te": "telugu", "fa": "persian",
    "lv": "latvian", "bn": "bengali", "sr": "serbian", "az": "azerbaijani", "sl": "slovenian",
    "kn": "kannada", "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "bs": "bosnian",
    "kk": "kazakh", "sq": "albanian", "sw": "swahili", "gl": "galician", "mr": "marathi",
    "pa": "punjabi", "si": "sinhala", "km": "khmer", "sn": "shona", "yo": "yoruba",
    "so": "somali", "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic", "yi": "yiddish",
    "lo": "lao", "uz": "uzbek", "fo": "faroese", "ht": "haitian creole", "ps": "pashto",
    "tk": "turkmen", "nn": "nynorsk", "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish",
    "my": "myanmar", "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa", "ba": "bashkir",
    "jw": "javanese", "su": "sundanese", "yue": "cantonese", "nb": "bokmal", "mn": "mongolian",
}

# whisper.cpp native languages - always available
LANGUAGES = WHISPER_LANGUAGES
TO_LANGUAGE_CODE = {v: k for k, v in WHISPER_LANGUAGES.items()}
LANG_BY_NAME = {name.lower(): code for code, name in WHISPER_LANGUAGES.items()}

# Native-script names (used by other toolkits' dropdowns, e.g. MiniMax).
LANG_BY_NATIVE = {
    "日本語": "ja", "中文": "zh", "汉语": "zh", "粵語": "yue", "한국어": "ko",
    "français": "fr", "deutsch": "de", "español": "es", "português": "pt",
    "русский": "ru", "українська": "uk", "italiano": "it", "العربية": "ar",
    "हिन्दी": "hi", "bahasa indonesia": "id", "tiếng việt": "vi", "ไทย": "th",
    "türkçe": "tr", "polski": "pl", "nederlands": "nl",
}

# Selectors that mean "figure the language out yourself".
AUTO_SELECTORS = frozenset({
    "", "none", "auto", "autodetect", "auto-detect", "auto detect",
    "detect", "auto language", "auto detect language", "language detection",
    "any", "mixed",
})

# Any selector carrying the word "multilingual" means auto-detect
# (e.g. "Multilingual / mixed", "multilingual").
_MULTILINGUAL_RE = re.compile(r"multilingual", re.IGNORECASE)

# "Label (English name)" / "日本語 (Japanese)" style selectors.
_PAREN_RE = re.compile(r"^\s*(.*?)\s*[（(]([^()]+)[)）]\s*$")

# "en-US" / "zh-Hans" style locale suffixes.
_LOCALE_RE = re.compile(r"[-_]")

# Keep in sync with LANG_VALUE_RE in js/whispercpp_node.js
# (tests/test_languages.py asserts both accept exactly the same options).
CANONICAL_OPTION_RE = re.compile(r"^(None|auto|[a-z]{2,3}|[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)$")


def language_options() -> list:
    """Dropdown values for the ``language`` input.

    ``None`` and ``auto`` both mean auto-detect, then every whisper.cpp code
    plus the English language name (both spellings ComfyUI has always
    accepted, so old workflows keep working).
    """
    opts = ["None", "auto"]
    opts.extend(sorted(WHISPER_LANGUAGES))
    opts.extend(sorted(name.title() for name in WHISPER_LANGUAGES.values()))
    return list(dict.fromkeys(opts))


def _match_known(value: str) -> str:
    """Return the whisper.cpp code for a plain selector, or "" if unknown."""
    low = value.lower()
    if low in WHISPER_LANGUAGES:
        return low
    if low in LANG_BY_NAME:
        return LANG_BY_NAME[low]
    if low in LANG_BY_NATIVE:
        return LANG_BY_NATIVE[low]
    # Locale suffix: "en-US", "zh-Hans" -> base language.
    if _LOCALE_RE.search(value):
        base = _LOCALE_RE.split(value)[0].lower()
        if base in WHISPER_LANGUAGES:
            return base
    return ""


def resolve_language(value) -> tuple:
    """Map any language selector to ``(code_or_None, recognized)``.

    * ``code_or_None`` - a whisper.cpp language code, or ``None`` for
      auto-detect. Never raises, never returns an unusable string: unknown
      input degrades to auto-detect instead of crashing whisper.cpp with
      "unknown language".
    * ``recognized`` - ``True`` when the selector actually meant something
      (auto synonyms, known code/name/label), ``False`` for garbage so the
      caller can log a warning.

    Examples::

        resolve_language("ja")                     -> ("ja", True)
        resolve_language("Japanese")               -> ("ja", True)
        resolve_language("日本語 (Japanese)")        -> ("ja", True)
        resolve_language("Multilingual / mixed")   -> (None, True)
        resolve_language("None")                   -> (None, True)
        resolve_language("Klingon")                -> (None, False)
    """
    if value is None:
        return None, True
    if not isinstance(value, str):
        return None, False

    s = value.strip()
    if not s:
        return None, True

    low = s.lower()
    if low in AUTO_SELECTORS or _MULTILINGUAL_RE.search(s):
        return None, True

    code = _match_known(s)
    if code:
        return code, True

    m = _PAREN_RE.match(s)
    if m:
        inner, outer = m.group(2).strip(), m.group(1).strip()
        for cand in (inner, outer):
            if not cand:
                continue
            if cand.lower() in AUTO_SELECTORS or _MULTILINGUAL_RE.search(cand):
                return None, True
            code = _match_known(cand)
            if code:
                return code, True
        return None, False

    return None, False


def normalize_language(value) -> str:
    """``resolve_language`` minus the flag - handy for simple call sites."""
    return resolve_language(value)[0]
