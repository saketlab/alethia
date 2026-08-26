"""Generate deterministic dirty variants as label-free positive pairs."""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

_KEYBOARD_NEIGHBORS: dict[str, str] = {
    "q": "wa",
    "w": "qeas",
    "e": "wrsd",
    "r": "etdf",
    "t": "ryfg",
    "y": "tugh",
    "u": "yihj",
    "i": "uojk",
    "o": "ipkl",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}


def _adjacent(ch: str) -> str:
    lower = ch.lower()
    neighbors = _KEYBOARD_NEIGHBORS.get(lower)
    if not neighbors:
        return ch
    # randomness comes from the choice of position, not the substitute
    pick = neighbors[ord(lower) % len(neighbors)]
    return pick.upper() if ch.isupper() else pick


def _typo(text: str, rng) -> str:
    if len(text) < 2:
        return text
    i = rng.randrange(len(text))
    return text[:i] + _adjacent(text[i]) + text[i + 1 :]


def _transpose(text: str, rng) -> str:
    if len(text) < 3:
        return text
    i = rng.randrange(len(text) - 1)
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


def _delete_char(text: str, rng) -> str:
    if len(text) < 3:
        return text
    i = rng.randrange(len(text))
    return text[:i] + text[i + 1 :]


def _case_change(text: str, rng) -> str:
    choice = rng.randrange(3)
    if choice == 0:
        return text.lower()
    if choice == 1:
        return text.upper()
    return text.title()


def _drop_punct(text: str, rng) -> str:
    return "".join(c for c in text if c.isalnum() or c.isspace())


def _abbreviate(text: str, rng) -> str:
    tokens = text.split()
    if len(tokens) < 2:
        return text[: max(2, len(text) // 2)] if len(text) > 4 else text
    i = rng.randrange(1, len(tokens))
    if rng.random() < 0.5:
        tokens.pop(i)
    else:
        tokens[i] = tokens[i][0]
    return " ".join(tokens)


def _extra_space(text: str, rng) -> str:
    if len(text) < 2:
        return text
    i = rng.randrange(1, len(text))
    return text[:i] + " " + text[i:]


def _uppercase(text: str, rng) -> str:
    return text.upper()


def _lowercase(text: str, rng) -> str:
    return text.lower()


def _drop_comma_clause(text: str, rng) -> str:
    """Drop a trailing comma-delimited qualifier ("Cholera, unspecified" -> "Cholera")."""
    if "," not in text:
        return text
    head, _, _tail = text.partition(",")
    return head.strip() or text


def _drop_parenthetical(text: str, rng) -> str:
    stripped = re.sub(r"\s*\([^)]*\)", "", text).strip()
    return stripped or text


def _add_parenthetical(text: str, rng) -> str:
    """Append a status qualifier, the way clinical free text annotates a diagnosis."""
    note = _PAREN_NOTES[rng.randrange(len(_PAREN_NOTES))]
    return f"{text} ({note})"


def _truncate_tokens(text: str, rng) -> str:
    """Keep a leading run of tokens; real queries are markedly shorter than references."""
    tokens = text.split()
    if len(tokens) < 3:
        return text
    keep = rng.randint(max(1, len(tokens) // 2), len(tokens) - 1)
    return " ".join(tokens[:keep])


def _acronymize(text: str, rng) -> str:
    """Replace a run of capitalizable words with its initialism (UTI, ARDS)."""
    tokens = text.split()
    words = [i for i, t in enumerate(tokens) if t.isalpha() and len(t) > 3]
    if len(words) < 2:
        return text
    start_idx = rng.randrange(len(words) - 1)
    span = words[start_idx : start_idx + rng.randint(2, min(4, len(words) - start_idx))]
    if len(span) < 2 or span[-1] - span[0] != len(span) - 1:
        return text
    acronym = "".join(tokens[i][0].upper() for i in span)
    return " ".join(tokens[: span[0]] + [acronym] + tokens[span[-1] + 1 :])


def _mojibake(text: str, rng) -> str:
    """Reproduce UTF-8 bytes mis-decoded as Latin-1; a no-op on pure ASCII."""
    try:
        return text.encode("utf-8").decode("latin-1")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


_PAREN_NOTES = ("RESOLVED", "SEVERE", "ACUTE", "SUSPECTED", "R/O", "CHRONIC")

_PERTURBATION_REGISTRY: dict[str, Callable[[str, object], str]] = {
    "typo": _typo,
    "transpose": _transpose,
    "delete_char": _delete_char,
    "case_change": _case_change,
    "uppercase": _uppercase,
    "lowercase": _lowercase,
    "drop_punct": _drop_punct,
    "abbreviate": _abbreviate,
    "extra_space": _extra_space,
    "drop_comma_clause": _drop_comma_clause,
    "drop_parenthetical": _drop_parenthetical,
    "add_parenthetical": _add_parenthetical,
    "truncate_tokens": _truncate_tokens,
    "acronymize": _acronymize,
    "mojibake": _mojibake,
}


#: used when no profile is supplied
_DEFAULT_PROFILE_WEIGHTS = {
    "typo": 1.0,
    "transpose": 1.0,
    "delete_char": 1.0,
    "case_change": 1.0,
    "drop_punct": 1.0,
    "abbreviate": 1.0,
    "extra_space": 1.0,
}


@dataclass
class NoiseProfile:
    """A weighted mix of perturbations describing how a corpus is actually dirty.

    ``weights`` names must exist in :data:`_PERTURBATION_REGISTRY`; ``evidence`` records what
    the estimator measured.
    """

    weights: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_PROFILE_WEIGHTS)
    )
    evidence: dict[str, float] = field(default_factory=dict)

    def population(self) -> tuple[list[Callable], list[float]]:
        """Return the (perturbation, weight) population for ``random.Random.choices``."""
        pairs = [(_PERTURBATION_REGISTRY[n], w) for n, w in self.weights.items()
                 if w > 0 and n in _PERTURBATION_REGISTRY]
        if not pairs:
            pairs = [(_PERTURBATION_REGISTRY[n], w)
                     for n, w in _DEFAULT_PROFILE_WEIGHTS.items()]
        fns, weights = zip(*pairs)
        return list(fns), list(weights)

    def describe(self) -> str:
        parts = sorted(
            ((n, w) for n, w in self.weights.items() if w > 0), key=lambda kv: -kv[1]
        )
        return ", ".join(f"{n}={w:.2f}" for n, w in parts)


def _rate(strings: Sequence[str], pred) -> float:
    strings = [s for s in strings if isinstance(s, str) and s.strip()]
    if not strings:
        return 0.0
    return sum(1 for s in strings if pred(s)) / len(strings)


def estimate_noise_profile(
    queries: Sequence[str], references: Sequence[str]
) -> NoiseProfile:
    """Infer which perturbations to apply by comparing query and reference surface form."""
    queries = [q for q in queries if isinstance(q, str) and q.strip()]
    references = [r for r in references if isinstance(r, str) and r.strip()]
    if not queries or not references:
        return NoiseProfile()

    def has_alpha(s):
        return any(c.isalpha() for c in s)

    def rates(pred):
        """The feature's rate in the queries and in the references."""
        return _rate(queries, pred), _rate(references, pred)

    q_upper, r_upper = rates(lambda s: s.isupper() and has_alpha(s))
    q_lower, r_lower = rates(lambda s: s.islower() and has_alpha(s))
    q_nonascii, r_nonascii = rates(lambda s: any(ord(c) > 127 for c in s))
    q_paren, r_paren = rates(lambda s: "(" in s)
    q_comma, r_comma = rates(lambda s: "," in s)
    q_acronym, r_acronym = rates(
        lambda s: bool(re.search(r"\b[A-Z]{2,6}\b", s)) and not s.isupper()
    )

    upper = max(0.0, q_upper - r_upper)
    lower = max(0.0, q_lower - r_lower)
    nonascii = max(0.0, q_nonascii - r_nonascii)
    paren = max(0.0, q_paren - r_paren)
    acronym = max(0.0, q_acronym - r_acronym)

    comma_loss = max(0.0, r_comma - q_comma)
    paren_loss = max(0.0, r_paren - q_paren)

    q_tokens = sum(len(q.split()) for q in queries) / len(queries)
    r_tokens = sum(len(r.split()) for r in references) / len(references)
    shortening = max(0.0, (r_tokens - q_tokens) / r_tokens) if r_tokens else 0.0

    weights = {
        "uppercase": upper,
        "lowercase": lower,
        "mojibake": nonascii,
        "add_parenthetical": paren,
        "acronymize": acronym,
        "drop_comma_clause": comma_loss,
        "drop_parenthetical": paren_loss,
        "truncate_tokens": shortening,
        # floor keeps character noise represented when measured signals are weak
        "typo": 0.05,
        "transpose": 0.05,
        "delete_char": 0.05,
        "extra_space": 0.05,
        "abbreviate": 0.05,
    }
    weights = {k: round(v, 4) for k, v in weights.items() if v > 0}
    if not weights:
        return NoiseProfile()

    return NoiseProfile(
        weights=weights,
        evidence={
            "query_upper_rate": q_upper,
            "reference_upper_rate": r_upper,
            "query_mean_tokens": q_tokens,
            "reference_mean_tokens": r_tokens,
            "query_nonascii_rate": q_nonascii,
            "reference_comma_rate": r_comma,
            "query_comma_rate": q_comma,
        },
    )


def make_dirty_variant(
    text: str, rng, n_edits: int = 1, profile: NoiseProfile | None = None
) -> str:
    """Apply ``n_edits`` perturbations to ``text``, drawn according to ``profile``."""
    fns, weights = (profile or NoiseProfile()).population()
    out = text
    for perturb in rng.choices(fns, weights=weights, k=max(1, n_edits)):
        candidate = perturb(out, rng)
        if candidate:
            out = candidate
    return out


def generate_positive_pairs(
    references: list[str],
    n_variants: int = 2,
    max_edits: int = 2,
    seed: int = 0,
    max_references: int | None = None,
    profile: NoiseProfile | None = None,
) -> tuple[list[str], list[str]]:
    """Generate ``(sources, variants)`` positive pairs from clean references."""
    import random

    rng = random.Random(seed)
    refs = [r for r in references if isinstance(r, str) and r.strip()]
    if max_references is not None and len(refs) > max_references:
        refs = rng.sample(refs, max_references)

    sources: list[str] = []
    variants: list[str] = []
    for ref in refs:
        for _ in range(n_variants):
            n_edits = rng.randint(1, max(1, max_edits))
            variant = make_dirty_variant(ref, rng, n_edits=n_edits, profile=profile)
            if variant != ref:
                sources.append(ref)
                variants.append(variant)
    return sources, variants
