"""Tests for splitting composite entries before matching."""

import pandas as pd

from alethia import alethia_split

from ._helpers import char_bag


class TestAlethiaSplit:
    def test_picks_the_best_scoring_piece(self):
        res = alethia_split(
            ["apple, zzzzzzzzzz"],
            ["apple", "banana"],
            split_on=",",
            model=char_bag,
        )
        row = res.iloc[0]
        assert row["alethia_prediction"] == "apple"
        assert row["alethia_matched_part"] == "apple"

    def test_provenance_keeps_every_piece(self):
        res = alethia_split(
            ["apple, banana"],
            ["apple", "banana"],
            split_on=",",
            model=char_bag,
        )
        parts = res.iloc[0]["alethia_parts"]
        assert {c["part"] for c in parts} == {"apple", "banana"}
        assert parts[0]["score"] >= parts[1]["score"]

    def test_repeated_parts_across_entries_stay_aligned(self):
        # the deduped "aple" result has to be looked up by every entry that used it
        res = alethia_split(
            ["aple, aple", "aple, bananna"],
            ["apple", "banana"],
            split_on=",",
            model="rapidfuzz",
        )
        assert res.iloc[0]["alethia_prediction"] == "apple"
        assert res.iloc[1]["alethia_prediction"] == "banana"
        assert res.iloc[1]["alethia_matched_part"] == "bananna"

    def test_no_separator_matches_whole_entry(self):
        res = alethia_split(["apple"], ["apple", "banana"], split_on=",", model=char_bag)
        assert res.iloc[0]["alethia_matched_part"] == "apple"
        assert res.iloc[0]["alethia_prediction"] == "apple"

    def test_multiword_entry_without_the_separator_is_not_split_on_words(self):
        res = alethia_split(
            ["Cns Tuberculosis"],
            ["Tuberculosis of nervous system"],
            split_on=",",
            model=char_bag,
        )
        row = res.iloc[0]
        assert len(row["alethia_parts"]) == 1
        assert row["alethia_matched_part"] == "Cns Tuberculosis"

    def test_nan_entry_passes_through(self):
        res = alethia_split([float("nan")], ["apple"], split_on=",", model=char_bag)
        assert pd.isna(res.iloc[0]["alethia_score"])
        assert pd.isna(res.iloc[0]["alethia_prediction"])
