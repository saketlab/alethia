"""Matching for composite entries: split on a separator, match each piece, keep the best."""

import pandas as pd

from .alethia import _is_nan_entry, alethia
from .embedder import rank_key


def _split_parts(entry, split_on: str) -> list:
    if _is_nan_entry(entry):
        return [entry]
    parts = [s for p in str(entry).split(split_on) if (s := p.strip())]
    return parts or [entry]


def alethia_split(
    dirty_entries: list[str],
    reference_entries: list[str],
    split_on: str,
    **kwargs,
) -> pd.DataFrame:
    """Match composite entries by splitting them on ``split_on`` before matching.

    Each entry is split into pieces on the literal separator, every piece is matched
    independently against ``reference_entries`` via :func:`alethia`, and the
    highest-scoring piece becomes the entry's prediction. Ties break on first
    occurrence, same as the rest of the package.

    Args:
        split_on: Literal separator each entry is split on (e.g. ``","``, ``";"``,
            ``" and "``). Entries with no separator, or that split to nothing usable,
            are matched whole.
        **kwargs: Forwarded to :func:`alethia` for the per-piece matching (``model``,
            ``threshold``, ``backend``, ...).

    Returns:
        A frame of ``given_entity``, ``alethia_prediction``, ``alethia_score``,
        ``alethia_matched_part`` (the winning piece), and ``alethia_parts`` (every
        piece tried, as a list of ``{"part", "prediction", "score"}`` dicts, best
        first).
    """
    parts_per_entry = [_split_parts(entry, split_on) for entry in dirty_entries]
    flat_parts = [part for parts in parts_per_entry for part in parts]

    # a repeated piece is matched once
    codes, uniques = pd.Index(flat_parts).factorize(use_na_sentinel=False)
    part_results = alethia(list(uniques), reference_entries, **kwargs)
    given = part_results["given_entity"].to_numpy()
    predictions = part_results["alethia_prediction"].to_numpy()
    scores = part_results["alethia_score"].to_numpy()

    rows = []
    cursor = 0
    for entry, parts in zip(dirty_entries, parts_per_entry):
        idx = codes[cursor : cursor + len(parts)]
        cursor += len(parts)

        candidates = sorted(
            (
                {"part": part, "prediction": pred, "score": score}
                for part, pred, score in zip(given[idx], predictions[idx], scores[idx])
            ),
            key=lambda c: -rank_key(c["score"]) if pd.notna(c["score"]) else float("inf"),
        )
        best = candidates[0]

        rows.append(
            {
                "given_entity": entry,
                "alethia_prediction": best["prediction"],
                "alethia_score": best["score"],
                "alethia_matched_part": best["part"],
                "alethia_parts": candidates,
            }
        )

    return pd.DataFrame(rows)
