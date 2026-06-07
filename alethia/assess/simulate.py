"""Generate dirty variants of clean strings as label-free positive pairs (deterministic)."""

from typing import Dict, List, Optional, Tuple

# QWERTY adjacency for realistic typo substitution.
_KEYBOARD_NEIGHBORS: Dict[str, str] = {
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
    # Deterministic pick driven by the char itself; randomness comes from position choice.
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
        # Truncate a single long token.
        return text[: max(2, len(text) // 2)] if len(text) > 4 else text
    # Drop a random non-first token, or initialize it.
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


_PERTURBATIONS = [
    _typo,
    _transpose,
    _delete_char,
    _case_change,
    _drop_punct,
    _abbreviate,
    _extra_space,
]


def make_dirty_variant(text: str, rng, n_edits: int = 1) -> str:
    """Apply ``n_edits`` random perturbations to ``text`` using ``rng``."""
    out = text
    for _ in range(max(1, n_edits)):
        perturb = _PERTURBATIONS[rng.randrange(len(_PERTURBATIONS))]
        candidate = perturb(out, rng)
        if candidate:  # never collapse to empty
            out = candidate
    return out


def generate_positive_pairs(
    references: List[str],
    n_variants: int = 2,
    max_edits: int = 2,
    seed: int = 0,
    max_references: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Generate (source, dirty_variant) positive pairs from clean references.

    Args:
        references: Clean canonical strings.
        n_variants: Number of dirty variants to synthesize per reference.
        max_edits: Upper bound on edits applied per variant (1..max_edits, chosen per variant).
        seed: RNG seed for reproducibility.
        max_references: If set, sample at most this many references (deterministically).

    Returns:
        ``(sources, variants)`` parallel lists where ``variants[i]`` is a dirty version of
        ``sources[i]``. Variants identical to their source are skipped.
    """
    import random

    rng = random.Random(seed)
    refs = [r for r in references if isinstance(r, str) and r.strip()]
    if max_references is not None and len(refs) > max_references:
        refs = rng.sample(refs, max_references)

    sources: List[str] = []
    variants: List[str] = []
    for ref in refs:
        for _ in range(n_variants):
            n_edits = rng.randint(1, max(1, max_edits))
            variant = make_dirty_variant(ref, rng, n_edits=n_edits)
            if variant != ref:
                sources.append(ref)
                variants.append(variant)
    return sources, variants
