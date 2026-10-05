"""Unit tests for whispercpp.languages (issue #16).

Covers the selector normalization the node relies on, plus a three-way sync
check between the values the backend offers and the frontend's whitelist
regex (js/whispercpp_node.js).
"""
import re
from functools import lru_cache
from pathlib import Path

import pytest

from whispercpp.languages import (
    CANONICAL_OPTION_RE,
    WHISPER_LANGUAGES,
    language_options,
    normalize_language,
    resolve_language,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
JS_FILE = REPO_ROOT / "js" / "whispercpp_node.js"

# Labels reported in issue #16 (injected by MiniMax Music Production Toolkit).
POLLUTED_LABELS = ["日本語 (Japanese)", "Multilingual / mixed"]


@lru_cache(maxsize=1)
def _js_lang_regex() -> re.Pattern:
    """Extract LANG_VALUE_RE from the frontend source so both stay in sync."""
    src = JS_FILE.read_text(encoding="utf-8")
    m = re.search(r"const\s+LANG_VALUE_RE\s*=\s*/(.+)/;", src)
    assert m, "LANG_VALUE_RE not found in js/whispercpp_node.js"
    return re.compile(m.group(1))


class TestResolveLanguage:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("ja", "ja"),
            ("en", "en"),
            ("haw", "haw"),
            ("Japanese", "ja"),
            ("japanese", "ja"),
            ("Haitian Creole", "ht"),
            ("  French  ", "fr"),
            ("日本語 (Japanese)", "ja"),
            ("日本語", "ja"),
            ("Deutsch (German)", "de"),
            ("en-US", "en"),
            ("zh-Hans", "zh"),
        ],
    )
    def test_known_selectors(self, value, expected):
        code, recognized = resolve_language(value)
        assert recognized is True
        assert code == expected

    @pytest.mark.parametrize(
        "value",
        ["None", "none", "auto", "AUTO", "", "   ", "detect", "Mixed", "Multilingual / mixed", "multilingual"],
    )
    def test_auto_selectors(self, value):
        code, recognized = resolve_language(value)
        assert recognized is True
        assert code is None

    @pytest.mark.parametrize("value", ["Klingon", "language id", "12345", 12345, ["ja"]])
    def test_unknown_falls_back_to_auto(self, value):
        code, recognized = resolve_language(value)
        assert code is None
        assert recognized is (value is None)

    def test_none_input(self):
        assert resolve_language(None) == (None, True)

    def test_normalize_language_matches_resolve(self):
        for raw in ["ja", "Japanese", "日本語 (Japanese)", "Multilingual / mixed", "None", "Klingon"]:
            assert normalize_language(raw) == resolve_language(raw)[0]

    @pytest.mark.parametrize("code", sorted(WHISPER_LANGUAGES))
    def test_every_code_and_name_roundtrip(self, code):
        name = WHISPER_LANGUAGES[code]
        assert resolve_language(code) == (code, True)
        assert resolve_language(name) == (code, True)
        assert resolve_language(name.title()) == (code, True)


class TestLanguageOptions:
    def test_contains_defaults_codes_and_names(self):
        opts = language_options()
        assert opts[:2] == ["None", "auto"]
        for code in ("en", "ja", "zh", "haw"):
            assert code in opts
        assert "Japanese" in opts
        assert "Haitian Creole" in opts

    def test_length_and_uniqueness(self):
        opts = language_options()
        assert len(opts) == 2 + 2 * len(WHISPER_LANGUAGES)
        assert len(set(opts)) == len(opts)

    def test_does_not_offer_polluted_labels(self):
        opts = set(language_options())
        for label in POLLUTED_LABELS:
            assert label not in opts

    @pytest.mark.parametrize("opt", language_options())
    def test_every_option_matches_backend_regex(self, opt):
        assert CANONICAL_OPTION_RE.match(opt)

    @pytest.mark.parametrize("opt", language_options())
    def test_every_option_matches_frontend_regex(self, opt):
        assert _js_lang_regex().match(opt), f"{opt!r} rejected by js/whispercpp_node.js regex"


def test_frontend_regex_rejects_polluted_labels():
    rx = _js_lang_regex()
    for label in POLLUTED_LABELS:
        assert not rx.match(label)


def test_backend_and_frontend_regex_agree():
    """Same pattern on both sides: any value the dropdown can hold passes."""
    js = _js_lang_regex().pattern
    py = CANONICAL_OPTION_RE.pattern
    assert js == py, "js/whispercpp_node.js LANG_VALUE_RE drifted from whispercpp.languages"
