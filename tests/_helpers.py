"""Dependency-free embedders shared across the test suite (no model downloads)."""

import numpy as np

_VOCAB = "abcdefghijklmnopqrstuvwxyz "


def char_bag(texts):
    """Character-count bag embedder over a-z and space."""
    out = []
    for t in texts:
        v = np.zeros(len(_VOCAB), dtype=np.float32)
        for ch in str(t).lower():
            if ch in _VOCAB:
                v[_VOCAB.index(ch)] += 1.0
        out.append(v)
    return np.array(out)


def _stable_seed(text):
    """Per-string seed independent of PYTHONHASHSEED, for reproducible noise."""
    h = 0
    for ch in str(text):
        h = (h * 31 + ord(ch)) % (2**31)
    return h


def random_embed(texts):
    """A deliberately content-blind 'bad' embedder: reproducible pseudo-random vectors."""
    out = []
    for t in texts:
        rng = np.random.RandomState(_stable_seed(t))
        out.append(rng.randn(len(_VOCAB)).astype(np.float32))
    return np.array(out)
