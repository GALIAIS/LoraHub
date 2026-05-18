"""Internationalization for the Anima LoRA GUI.

Per-language string tables live in sibling modules (``en.py``, ``ko.py``,
``cn.py``). Add a new language by dropping in ``<code>.py`` exporting
``STRINGS: dict[str, str]`` and registering it in ``TRANSLATIONS`` below.
Missing keys fall back to English via ``t()``.
"""

from __future__ import annotations

from gui.i18n import cn, en, ko

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": en.STRINGS,
    "ko": ko.STRINGS,
    "cn": cn.STRINGS,
}

_current_lang = "en"
_SETTINGS_FILE = None


def _settings_path():
    global _SETTINGS_FILE
    if _SETTINGS_FILE is None:
        from pathlib import Path

        # __file__ is gui/i18n/__init__.py — settings live in gui/
        _SETTINGS_FILE = Path(__file__).resolve().parent.parent / "gui_settings.json"
    return _SETTINGS_FILE


def load_language() -> str:
    """Load saved language preference."""
    global _current_lang
    import json

    p = _settings_path()
    if p.exists():
        try:
            _current_lang = json.loads(p.read_text(encoding="utf-8")).get(
                "language", "en"
            )
        except (json.JSONDecodeError, OSError):
            _current_lang = "en"
    return _current_lang


def save_language(lang: str):
    """Persist language preference."""
    global _current_lang
    import json

    _current_lang = lang
    p = _settings_path()
    settings = {}
    if p.exists():
        try:
            settings = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    settings["language"] = lang
    p.write_text(json.dumps(settings), encoding="utf-8")


def set_language(lang: str):
    global _current_lang
    _current_lang = lang


def t(key: str, **kwargs) -> str:
    """Translate a key using the current language."""
    s = TRANSLATIONS.get(_current_lang, TRANSLATIONS["en"]).get(key)
    if s is None:
        s = TRANSLATIONS["en"].get(key, key)
    if kwargs:
        s = s.format(**kwargs)
    return s


def current_language() -> str:
    return _current_lang


def available_languages() -> list[str]:
    return list(TRANSLATIONS.keys())
