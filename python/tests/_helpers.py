import numpy as np

REFS = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
    "Dallas",
    "San Jose",
]

_VOCAB = "abcdefghijklmnopqrstuvwxyz "
_VOCAB_INDEX = {ch: i for i, ch in enumerate(_VOCAB)}


def char_bag(texts):
    out = []
    for t in texts:
        v = np.zeros(len(_VOCAB), dtype=np.float32)
        for ch in str(t).lower():
            if ch in _VOCAB_INDEX:
                v[_VOCAB_INDEX[ch]] += 1.0
        out.append(v)
    return np.array(out)


def _stable_seed(text):
    h = 0
    for ch in str(text):
        h = (h * 31 + ord(ch)) % (2**31)
    return h


def random_embed(texts):
    out = []
    for t in texts:
        rng = np.random.RandomState(_stable_seed(t))
        out.append(rng.randn(len(_VOCAB)).astype(np.float32))
    return np.array(out)
