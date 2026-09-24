"""Every language the product can be read in, built from CLDR rather than typed by hand.

Babel ships the Unicode CLDR data. English names come from CLDR's English display names
(every code CLDR can name), autonyms from each language's own locale where CLDR has one,
and direction from the locale — or, where there is no locale, from the script CLDR says the
language is most likely written in. The only hand-kept data are the support tiers (they
describe a model, not a language) and a few colloquial aliases CLDR does not carry; both
live in config/i18n/.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from babel import Locale, UnknownLocaleError
from babel.core import get_global

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config" / "i18n"

# Scripts written right to left (Unicode). A fact about scripts, not a language list.
_RTL_SCRIPTS = frozenset({"Arab", "Hebr", "Syrc", "Thaa", "Nkoo", "Adlm", "Rohg", "Mand", "Samr", "Mend", "Yezi"})
# CLDR codes that are not languages anyone reads a screen in.
_NOT_LANGUAGES = frozenset({"und", "mul", "zxx", "mis"})
_CUSTOM_MAX = 60


def fold(text: str) -> str:
    """Case- and accent-insensitive form: 'Español' and 'espanol' fold the same."""
    decomposed = unicodedata.normalize("NFKD", (text or "").casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).strip()


def _norm_code(code: str) -> str:
    return (code or "").strip().replace("_", "-").lower()


def is_source_language(code: str) -> bool:
    """English is the source; any English variant is served the source text."""
    return _norm_code(code).split("-")[0] == "en"


@dataclass(frozen=True)
class Language:
    code: str
    english: str
    native: str
    aliases: tuple[str, ...]
    dir: str
    tier: int
    custom: bool = False

    def label(self) -> str:
        return self.english if self.native == self.english else f"{self.native} — {self.english}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["aliases"] = list(self.aliases)
        d["label"] = self.label()
        return d


def _locale(code: str) -> Locale | None:
    try:
        return Locale.parse(code.replace("-", "_"))
    except (UnknownLocaleError, ValueError):
        return None


def _native(code: str, english: str) -> str:
    loc = _locale(code)
    name = loc.get_display_name(loc) if loc else None
    if not name:
        return english
    return name[:1].upper() + name[1:]


def _direction(code: str) -> str:
    loc = _locale(code)
    if loc is not None:
        return "rtl" if loc.character_order == "right-to-left" else "ltr"
    likely = get_global("likely_subtags").get(code.replace("-", "_"), "")
    return "rtl" if any(part in _RTL_SCRIPTS for part in likely.split("_")) else "ltr"


def _bcp47(cldr_code: str) -> str:
    """'zh_Hant' -> 'zh-Hant', 'es_419' -> 'es-419' (CLDR already cases subtags correctly)."""
    return cldr_code.replace("_", "-")


def _load_json(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _tier_map(model: str) -> tuple[dict[str, int], int]:
    cfg = _load_json("tiers.json")
    default = int(cfg.get("default_tier", 3))
    tiers: dict[str, int] = {}
    for tier, codes in (cfg.get("models", {}).get(model) or {}).items():
        for c in codes:
            tiers[_norm_code(c)] = int(tier)
    return tiers, default


class LanguageRegistry:
    def __init__(self, languages: Iterable[Language]):
        self._langs = sorted(languages, key=lambda L: L.english.casefold())
        self._by_code = {_norm_code(L.code): L for L in self._langs}
        self._fields = {L.code: [fold(x) for x in (L.code, L.english, L.native, *L.aliases)] for L in self._langs}

    def all(self) -> list[Language]:
        return list(self._langs)

    def get(self, code: str) -> Language | None:
        return self._by_code.get(_norm_code(code))

    def _exact(self, text: str) -> Language | None:
        f = fold(text)
        for L in self._langs:
            if f in self._fields[L.code]:
                return L
        return None

    def search(self, query: str, *, pinned: Sequence[str] = (), limit: int = 50) -> list[Language]:
        pins = [p for p in (self.get(c) for c in pinned) if p]
        pin_rank = {L.code: i for i, L in enumerate(pins)}
        q = fold(query)
        if not q:
            rest = [L for L in self._langs if L.code not in pin_rank]
            return (pins + rest)[:limit]
        scored = []
        for L in self._langs:
            best = None
            for f in self._fields[L.code]:
                if f == q:
                    s = 0
                elif f.startswith(q):
                    s = 1
                elif any(w.startswith(q) for w in re.split(r"[\s\-()]+", f)):
                    s = 2
                elif q in f:
                    s = 3
                else:
                    continue
                best = s if best is None else min(best, s)
            if best is not None:
                scored.append((best, pin_rank.get(L.code, len(pin_rank)), L.english.casefold(), L))
        scored.sort(key=lambda t: t[:3])
        return [t[3] for t in scored[:limit]]

    def pin_codes(self, accept_language: str) -> list[str]:
        """English first, then the browser's languages in preference order, as registry codes."""
        weighted = []
        for i, part in enumerate((accept_language or "").split(",")):
            tag, _, q = part.strip().partition(";q=")
            if not tag or tag == "*":
                continue
            try:
                weight = float(q) if q else 1.0
            except ValueError:
                weight = 0.0
            weighted.append((-weight, i, tag))
        out = ["en"]
        for _, _, tag in sorted(weighted):
            L = self.get(tag) or self.get(tag.split("-")[0])
            if L and L.code not in out:
                out.append(L.code)
        return out

    def resolve(self, text: str) -> Language:
        """A typed language name -> a registry entry, else a custom private-use code."""
        name = " ".join((text or "").split())
        if not name or len(name) > _CUSTOM_MAX:
            raise ValueError(f"a language name must be 1-{_CUSTOM_MAX} characters")
        hit = self.get(name) or self._exact(name)
        if hit:
            return hit
        slug = re.sub(r"[^a-z0-9]+", "-", fold(name)).strip("-")
        subtags = [s[:8] for s in slug.split("-") if s][:4]
        if not subtags:
            subtags = [hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]]
        return Language(code="x-" + "-".join(subtags), english=name, native=name,
                        aliases=(), dir="ltr", tier=3, custom=True)


def build_registry(model: str) -> LanguageRegistry:
    names = Locale("en").languages
    extra = {_norm_code(k): v for k, v in _load_json("aliases.json").items() if not k.startswith("_")}
    deprecated: dict[str, list[str]] = {}
    for old, new in get_global("language_aliases").items():
        deprecated.setdefault(_norm_code(new), []).append(old)
    tiers, default = _tier_map(model)

    langs = []
    for cldr_code, english in names.items():
        if cldr_code in _NOT_LANGUAGES:
            continue
        code = _bcp47(cldr_code)
        key = _norm_code(code)
        base = key.split("-")[0]
        tier = 1 if base == "en" else tiers.get(key, tiers.get(base, default))
        langs.append(Language(
            code=code, english=english, native=_native(code, english),
            aliases=tuple(deprecated.get(key, []) + extra.get(key, [])),
            dir=_direction(code), tier=tier,
        ))
    return LanguageRegistry(langs)
