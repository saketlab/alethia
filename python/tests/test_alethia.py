#!/usr/bin/env python

"""Tests for `alethia` package."""

import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alethia import alethia
from alethia.alethia import (
    _filter_nan_entries,
    _find_exact_matches,
    _is_nan_entry,
    _preprocess_entries_with_nans,
    check_optional_dependencies,
    get_best_available_backend,
    run_gemini_matching,
    run_openai_matching,
    run_rapidfuzz_matching,
)

alethia_mod = importlib.import_module("alethia.alethia")


@pytest.fixture
def sample_dirty_entries():
    return ["NY", "LA", "Chiacgo", "Houston City", "San Fran"]


@pytest.fixture
def sample_reference_entries():
    return ["New York", "Los Angeles", "Chicago", "Houston", "San Francisco"]


@pytest.fixture
def sample_entries_with_nans():
    return ["NY", "LA", np.nan, "Chicago", None, "Houston", ""]


@pytest.fixture
def sample_reference_with_nans():
    return ["New York", "Los Angeles", np.nan, "Chicago", "Houston", None]


class TestDependencyChecking:
    """Test dependency checking functionality"""

    def test_check_optional_dependencies(self):
        deps = check_optional_dependencies()
        expected_keys = {
            "SENTENCE_TRANSFORMERS_AVAILABLE",
            "FASTEMBED_AVAILABLE",
            "RAPIDFUZZ_AVAILABLE",
            "OPENAI_AVAILABLE",
            "GEMINI_AVAILABLE",
        }
        assert set(deps.keys()) == expected_keys
        assert all(isinstance(v, bool) for v in deps.values())

    def test_get_best_available_backend(self):
        backend = get_best_available_backend()
        assert backend in [
            "sentence-transformers",
            "fastembed",
            "openai",
            "gemini",
            "rapidfuzz",
            "exact",
        ]

        backend_cpu = get_best_available_backend(prefer_cpu=True)
        assert backend_cpu in [
            "fastembed",
            "sentence-transformers",
            "openai",
            "gemini",
            "rapidfuzz",
            "exact",
        ]


class TestRapidFuzzMatching:
    """Test RapidFuzz-based matching functionality"""

    @pytest.mark.skipif(
        not check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"],
        reason="RapidFuzz not available",
    )
    def test_run_rapidfuzz_matching_basic(
        self, sample_dirty_entries, sample_reference_entries
    ):
        result = run_rapidfuzz_matching(sample_dirty_entries, sample_reference_entries)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dirty_entries)
        assert list(result.columns) == [
            "given_entity",
            "alethia_prediction",
            "alethia_score",
        ]

        assert all(0 <= score <= 1 for score in result["alethia_score"])

        assert len(result["given_entity"].unique()) == len(sample_dirty_entries)

    @pytest.mark.skipif(
        not check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"],
        reason="RapidFuzz not available",
    )
    def test_rapidfuzz_exact_matches(self):
        dirty = ["New York", "Los Angeles"]
        reference = ["New York", "Los Angeles", "Chicago"]

        result = run_rapidfuzz_matching(dirty, reference)

        for _, row in result.iterrows():
            if row["given_entity"] == row["alethia_prediction"]:
                assert row["alethia_score"] == 1.0

    @pytest.mark.skipif(
        not check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"],
        reason="RapidFuzz not available",
    )
    def test_rapidfuzz_with_empty_inputs(self):
        result = run_rapidfuzz_matching([], [])
        assert len(result) == 0

        result = run_rapidfuzz_matching(["test"], [])
        assert len(result) == 1
        assert result.iloc[0]["given_entity"] == "test"


class TestSentenceTransformersMatching:
    """Test Sentence Transformers-based matching functionality"""

    @pytest.mark.skipif(
        not check_optional_dependencies()["SENTENCE_TRANSFORMERS_AVAILABLE"],
        reason="Sentence Transformers not available",
    )
    def test_alethia_sentence_transformers(
        self, sample_dirty_entries, sample_reference_entries
    ):
        result = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="sentence-transformers",
            force_cpu=True,
            verbose=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dirty_entries)
        assert list(result.columns) == [
            "given_entity",
            "alethia_prediction",
            "alethia_score",
            "alethia_method",
            "alethia_backend",
        ]

        scored = result["alethia_score"].dropna()
        assert all(0 <= score <= 1 for score in scored)
        unmatched = result["alethia_score"].isna()
        assert result.loc[unmatched, "alethia_prediction"].isna().all()

        assert all(
            backend == "sentence-transformers" for backend in result["alethia_backend"]
        )

    @pytest.mark.skipif(
        not check_optional_dependencies()["SENTENCE_TRANSFORMERS_AVAILABLE"],
        reason="Sentence Transformers not available",
    )
    def test_sentence_transformers_batch_optimization(
        self, sample_dirty_entries, sample_reference_entries
    ):
        result_batch = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="sentence-transformers",
            use_batch_optimization=True,
            force_cpu=True,
            verbose=False,
        )

        result_standard = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="sentence-transformers",
            use_batch_optimization=False,
            force_cpu=True,
            verbose=False,
        )

        assert len(result_batch) == len(result_standard)
        assert list(result_batch.columns) == list(result_standard.columns)

    @pytest.mark.skipif(
        not check_optional_dependencies()["SENTENCE_TRANSFORMERS_AVAILABLE"],
        reason="Sentence Transformers not available",
    )
    def test_sentence_transformers_with_threshold(
        self, sample_dirty_entries, sample_reference_entries
    ):
        result_high = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="sentence-transformers",
            threshold=0.9,
            force_cpu=True,
            verbose=False,
        )

        result_low = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="sentence-transformers",
            threshold=0.1,
            force_cpu=True,
            verbose=False,
        )

        unchanged_high = sum(
            result_high["given_entity"] == result_high["alethia_prediction"]
        )
        unchanged_low = sum(
            result_low["given_entity"] == result_low["alethia_prediction"]
        )

        assert unchanged_high >= unchanged_low


class TestFastEmbedMatching:
    """Test FastEmbed-based matching functionality"""

    @pytest.mark.skipif(
        not check_optional_dependencies()["FASTEMBED_AVAILABLE"],
        reason="FastEmbed not available",
    )
    def test_alethia_fastembed(self, sample_dirty_entries, sample_reference_entries):
        result = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="fastembed",
            verbose=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dirty_entries)
        assert list(result.columns) == [
            "given_entity",
            "alethia_prediction",
            "alethia_score",
            "alethia_method",
            "alethia_backend",
        ]

        scored = result["alethia_score"].dropna()
        assert all(0 <= score <= 1 for score in scored)
        unmatched = result["alethia_score"].isna()
        assert result.loc[unmatched, "alethia_prediction"].isna().all()

        assert all(backend == "fastembed" for backend in result["alethia_backend"])

    @pytest.mark.skipif(
        not check_optional_dependencies()["FASTEMBED_AVAILABLE"],
        reason="FastEmbed not available",
    )
    def test_fastembed_batch_vs_standard(
        self, sample_dirty_entries, sample_reference_entries
    ):
        result_batch = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="fastembed",
            use_batch_optimization=True,
            verbose=False,
        )

        result_standard = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="fastembed",
            use_batch_optimization=False,
            verbose=False,
        )

        assert len(result_batch) == len(result_standard)
        assert list(result_batch.columns) == list(result_standard.columns)


class TestAutoBackendSelection:
    """Test automatic backend selection"""

    def test_auto_backend_selection(
        self, sample_dirty_entries, sample_reference_entries
    ):
        result = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            backend="auto",
            verbose=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dirty_entries)

        if "alethia_backend" in result.columns:
            assert result["alethia_backend"].iloc[0] in [
                "sentence-transformers",
                "fastembed",
                "openai",
                "gemini",
                "rapidfuzz",
                "auto",
            ]

    def test_fallback_to_rapidfuzz(
        self, sample_dirty_entries, sample_reference_entries
    ):
        with patch.object(
            alethia_mod, "load_sentence_transformer_model", return_value=None
        ):
            with patch.object(alethia_mod, "load_fastembed_model", return_value=None):
                result = alethia(
                    sample_dirty_entries,
                    sample_reference_entries,
                    model="all-MiniLM-L6-v2",
                    backend="sentence-transformers",
                    verbose=False,
                )

                assert isinstance(result, pd.DataFrame)
                assert len(result) == len(sample_dirty_entries)


class TestExactMatching:
    """Test exact matching functionality"""

    def test_find_exact_matches_case_sensitive(self):
        dirty = ["New York", "new york", "LA", "Chicago"]
        reference = ["New York", "Los Angeles", "Chicago"]

        exact_matches, remaining, remaining_indices = _find_exact_matches(
            dirty, reference, case_sensitive=True
        )

        assert len(exact_matches) == 2
        assert len(remaining) == 2
        assert 0 in exact_matches
        assert 3 in exact_matches

    def test_find_exact_matches_case_insensitive(self):
        dirty = ["New York", "new york", "LA", "Chicago"]
        reference = ["New York", "Los Angeles", "Chicago"]

        exact_matches, remaining, remaining_indices = _find_exact_matches(
            dirty, reference, case_sensitive=False
        )

        assert len(exact_matches) == 3
        assert len(remaining) == 1
        assert 0 in exact_matches
        assert 1 in exact_matches
        assert 3 in exact_matches

    def test_exact_matching_integration(
        self, sample_dirty_entries, sample_reference_entries
    ):
        dirty_with_exact = sample_dirty_entries + ["Chicago", "Houston"]

        result = alethia(
            dirty_with_exact,
            sample_reference_entries,
            use_exact_matching=True,
            verbose=False,
        )

        exact_matches = result[result["given_entity"] == result["alethia_prediction"]]
        for _, row in exact_matches.iterrows():
            if row["given_entity"] in sample_reference_entries:
                assert row["alethia_score"] == 1.0


class TestNaNHandling:
    """Test NaN and None value handling"""

    def test_is_nan_entry(self):
        assert _is_nan_entry(None)
        assert _is_nan_entry(np.nan)
        assert _is_nan_entry("")
        assert _is_nan_entry("nan")
        assert _is_nan_entry("NaN")
        assert _is_nan_entry("null")
        assert _is_nan_entry("none")
        assert _is_nan_entry("na")
        assert _is_nan_entry("n/a")

        assert not _is_nan_entry("New York")
        assert not _is_nan_entry("0")
        assert not _is_nan_entry(0)

    def test_preprocess_entries_with_nans(self, sample_entries_with_nans):
        processed, nan_mask, original_indices = _preprocess_entries_with_nans(
            sample_entries_with_nans
        )

        assert len(processed) == 4
        assert len(nan_mask) == len(sample_entries_with_nans)
        assert sum(nan_mask) == 3
        assert len(original_indices) == 4

    def test_filter_nan_entries(self, sample_reference_with_nans):
        filtered = _filter_nan_entries(sample_reference_with_nans)

        assert len(filtered) == 4
        assert np.nan not in filtered
        assert None not in filtered
        assert "New York" in filtered
        assert "Los Angeles" in filtered

    def test_alethia_with_nan_entries(
        self, sample_entries_with_nans, sample_reference_entries
    ):
        result = alethia(
            sample_entries_with_nans, sample_reference_entries, verbose=False
        )

        assert len(result) <= len(sample_entries_with_nans)

        nan_results = result[
            result["given_entity"].isna()
            | result["given_entity"].isin(["", "nan", "null", "none"])
        ]

        for _, row in nan_results.iterrows():
            if pd.isna(row["given_entity"]) or str(row["given_entity"]).lower() in [
                "nan",
                "null",
                "none",
                "",
            ]:
                assert pd.isna(row["alethia_prediction"]) or pd.isna(
                    row["alethia_score"]
                )


class TestAPIBackends:
    """Test API-based backends (OpenAI, Gemini)"""

    @pytest.mark.api
    @pytest.mark.skipif(
        not check_optional_dependencies()["OPENAI_AVAILABLE"],
        reason="OpenAI not available",
    )
    def test_openai_matching_mock(self, sample_dirty_entries, sample_reference_entries):
        with patch.object(alethia_mod, "setup_openai_client") as mock_client_setup:
            with patch.object(alethia_mod, "get_openai_embedding") as mock_embedding:
                mock_client = MagicMock()
                mock_client_setup.return_value = mock_client

                def mock_embedding_func(client, text, model):
                    return [0.1] * 100

                mock_embedding.side_effect = mock_embedding_func

                result = run_openai_matching(
                    sample_dirty_entries, sample_reference_entries, threshold=0.5
                )

                assert isinstance(result, pd.DataFrame)
                assert len(result) == len(sample_dirty_entries)
                assert list(result.columns) == [
                    "given_entity",
                    "alethia_prediction",
                    "alethia_score",
                ]

    @pytest.mark.api
    @pytest.mark.skipif(
        not check_optional_dependencies()["GEMINI_AVAILABLE"],
        reason="Gemini not available",
    )
    def test_gemini_matching_mock(self, sample_dirty_entries, sample_reference_entries):
        with patch.object(alethia_mod, "setup_gemini_client") as mock_client_setup:
            with patch.object(alethia_mod, "get_gemini_embedding") as mock_embedding:
                mock_client_setup.return_value = MagicMock()

                def mock_embedding_func(text, model):
                    return [0.1] * 100

                mock_embedding.side_effect = mock_embedding_func

                result = run_gemini_matching(
                    sample_dirty_entries, sample_reference_entries, threshold=0.5
                )

                assert isinstance(result, pd.DataFrame)
                assert len(result) == len(sample_dirty_entries)
                assert list(result.columns) == [
                    "given_entity",
                    "alethia_prediction",
                    "alethia_score",
                ]


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_inputs(self):
        result = alethia([], [], verbose=False)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_single_entry(self):
        result = alethia(["test"], ["test"], verbose=False)
        assert len(result) == 1
        assert result.iloc[0]["given_entity"] == "test"
        assert result.iloc[0]["alethia_prediction"] == "test"
        assert result.iloc[0]["alethia_score"] == 1.0

    def test_no_reference_entries(self):
        result = alethia(["test"], [], verbose=False)
        assert len(result) == 1
        assert result.iloc[0]["given_entity"] == "test"

    def test_all_nan_entries(self):
        result = alethia([np.nan, None, ""], ["test"], verbose=False)
        assert len(result) == 3
        for _, row in result.iterrows():
            assert pd.isna(row["alethia_prediction"]) or pd.isna(row["alethia_score"])

    def test_duplicate_handling(self):
        dirty = ["NY", "NY", "LA"]
        reference = ["New York", "Los Angeles"]

        result_drop = alethia(dirty, reference, drop_duplicates=True, verbose=False)

        result_keep = alethia(dirty, reference, drop_duplicates=False, verbose=False)

        assert len(result_keep) == 3
        assert len(result_drop) <= len(result_keep)

    def test_remove_identical_hits(self):
        dirty = ["New York", "NY"]
        reference = ["New York", "Los Angeles"]

        result_remove = alethia(
            dirty, reference, remove_identical_hits=True, verbose=False
        )

        result_keep = alethia(
            dirty, reference, remove_identical_hits=False, verbose=False
        )

        assert len(result_keep) == 2
        assert len(result_remove) <= len(result_keep)

    def test_invalid_backend(self, sample_dirty_entries, sample_reference_entries):
        try:
            result = alethia(
                sample_dirty_entries,
                sample_reference_entries,
                backend="invalid_backend",
                verbose=False,
            )
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_dirty_entries)
        except ValueError:
            pass

    def test_return_model_attrs(self, sample_dirty_entries, sample_reference_entries):
        result_with_attrs = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            return_model_attrs=True,
            verbose=False,
        )

        result_without_attrs = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            return_model_attrs=False,
            verbose=False,
        )

        base_columns = ["given_entity", "alethia_prediction", "alethia_score"]
        attr_columns = ["alethia_method", "alethia_backend"]

        assert all(
            col in result_with_attrs.columns for col in base_columns + attr_columns
        )
        assert all(col in result_without_attrs.columns for col in base_columns)
        assert not any(col in result_without_attrs.columns for col in attr_columns)


class TestPerformanceAttributes:
    """Test performance attributes and metadata"""

    def test_result_attributes(self, sample_dirty_entries, sample_reference_entries):
        result = alethia(sample_dirty_entries, sample_reference_entries, verbose=False)

        expected_attrs = [
            "acceleration",
            "backend",
            "processing_time",
            "model",
            "nan_entries_count",
            "processed_entries_count",
        ]

        for attr in expected_attrs:
            assert attr in result.attrs

        assert isinstance(result.attrs["processing_time"], (int, float))
        assert isinstance(result.attrs["nan_entries_count"], int)
        assert isinstance(result.attrs["processed_entries_count"], int)
        assert result.attrs["processing_time"] >= 0

    def test_verbose_mode(self, sample_dirty_entries, sample_reference_entries, capsys):
        alethia(sample_dirty_entries, sample_reference_entries, verbose=True)

        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestThresholdContract:
    """`threshold` must reach every backend, and mean the same thing in each."""

    @pytest.mark.skipif(
        not check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"],
        reason="RapidFuzz not available",
    )
    def test_rapidfuzz_default_returns_best_guess(self):
        result = run_rapidfuzz_matching(["Bombay"], ["Mumbai", "Delhi"])
        assert result.iloc[0]["alethia_prediction"] == "Mumbai"

    @pytest.mark.skipif(
        not check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"],
        reason="RapidFuzz not available",
    )
    def test_rapidfuzz_honours_an_explicit_threshold(self):
        result = run_rapidfuzz_matching(["Bombay"], ["Mumbai", "Delhi"], threshold=0.95)
        assert pd.isna(result.iloc[0]["alethia_prediction"])
        assert pd.isna(result.iloc[0]["alethia_score"])

    @pytest.mark.skipif(
        not check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"],
        reason="RapidFuzz not available",
    )
    def test_threshold_reaches_rapidfuzz_through_alethia(self):
        loose = alethia(["Bombay"], ["Mumbai", "Delhi"], model="rapidfuzz")
        strict = alethia(
            ["Bombay"], ["Mumbai", "Delhi"], model="rapidfuzz", threshold=0.95
        )
        assert loose.iloc[0]["alethia_prediction"] == "Mumbai"
        assert pd.isna(strict.iloc[0]["alethia_prediction"])

    @pytest.mark.skipif(
        not check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"],
        reason="RapidFuzz not available",
    )
    def test_default_call_is_unchanged_for_existing_callers(self):
        result = alethia(["Bombay"], ["Mumbai", "Delhi"])
        assert result.iloc[0]["alethia_prediction"] == "Mumbai"

    def test_apply_threshold_is_a_no_op_when_unset(self):
        from alethia.alethia import apply_threshold

        frame = pd.DataFrame(
            {
                "given_entity": ["a"],
                "alethia_prediction": ["b"],
                "alethia_score": [0.1],
            }
        )
        assert apply_threshold(frame, None).iloc[0]["alethia_prediction"] == "b"

    def test_apply_threshold_blanks_prediction_and_score_together(self):
        from alethia.alethia import apply_threshold

        frame = pd.DataFrame(
            {
                "given_entity": ["low", "high"],
                "alethia_prediction": ["x", "y"],
                "alethia_score": [0.10, 0.99],
            }
        )
        out = apply_threshold(frame, 0.5)
        assert pd.isna(out.iloc[0]["alethia_prediction"])
        assert pd.isna(out.iloc[0]["alethia_score"])
        assert out.iloc[1]["alethia_prediction"] == "y"

    def test_apply_threshold_handles_an_empty_frame(self):
        from alethia.alethia import apply_threshold

        assert apply_threshold(pd.DataFrame(), 0.5).empty


class TestNoReferenceIsNotAMatch:
    def test_empty_reference_list_reports_no_match(self):
        result = alethia(["anything"], [])
        assert pd.isna(result.iloc[0]["alethia_prediction"])
        assert pd.isna(result.iloc[0]["alethia_score"])


class TestFallbackRunsOnce:
    """A failed backend must hand off exactly once, not re-run the fallback."""

    @pytest.mark.skipif(
        not check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"],
        reason="RapidFuzz not available",
    )
    def test_load_failure_does_not_match_twice(self, monkeypatch):
        import importlib

        mod = importlib.import_module("alethia.alethia")

        calls = []
        real = mod.run_rapidfuzz_matching

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(mod, "run_rapidfuzz_matching", counting)
        monkeypatch.setattr(
            mod, "load_sentence_transformer_model", lambda *a, **k: None
        )

        result = mod.alethia(
            ["Bombay"],
            ["Mumbai", "Delhi"],
            model="all-MiniLM-L6-v2",
            backend="sentence-transformers",
        )
        assert len(calls) == 1
        assert result.attrs["effective_backend"] == "rapidfuzz"

    @pytest.mark.skipif(
        not check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"],
        reason="RapidFuzz not available",
    )
    def test_fallback_is_recorded_not_reported_as_the_requested_backend(
        self, monkeypatch
    ):
        import importlib

        mod = importlib.import_module("alethia.alethia")

        monkeypatch.setattr(
            mod, "load_sentence_transformer_model", lambda *a, **k: None
        )
        result = mod.alethia(
            ["Bombay"],
            ["Mumbai"],
            model="all-MiniLM-L6-v2",
            backend="sentence-transformers",
        )
        assert result.attrs["backend"] == "sentence-transformers"
        assert result.attrs["effective_backend"] == "rapidfuzz"

    def test_no_backend_available_raises_rather_than_silently_degrading(
        self, monkeypatch
    ):
        import importlib

        mod = importlib.import_module("alethia.alethia")

        monkeypatch.setattr(mod, "RAPIDFUZZ_AVAILABLE", False)
        monkeypatch.setattr(mod, "OPENAI_AVAILABLE", False)
        monkeypatch.setattr(mod, "GEMINI_AVAILABLE", False)
        monkeypatch.setattr(
            mod, "load_sentence_transformer_model", lambda *a, **k: None
        )
        with pytest.raises(ValueError):
            mod.alethia(
                ["Bombay"],
                ["Mumbai"],
                model="all-MiniLM-L6-v2",
                backend="sentence-transformers",
            )


class TestCpuFirstBackendPreference:
    """Most users run on CPUs, so the ONNX runtime must win by default."""

    @staticmethod
    def _module():
        import importlib

        return importlib.import_module("alethia.alethia")

    def test_onnx_runtime_preferred_when_both_are_installed(self, monkeypatch):
        mod = self._module()
        monkeypatch.setattr(mod, "FASTEMBED_AVAILABLE", True)
        monkeypatch.setattr(mod, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
        assert mod.get_best_available_backend(prefer_cpu=True) == "fastembed"

    def test_alethia_defaults_to_the_cpu_path(self):
        import inspect

        sig = inspect.signature(self._module().alethia)
        assert sig.parameters["force_cpu"].default is True

    def test_torch_backend_still_reachable_when_asked_for(self, monkeypatch):
        mod = self._module()
        monkeypatch.setattr(mod, "FASTEMBED_AVAILABLE", True)
        monkeypatch.setattr(mod, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
        assert (
            mod.get_best_available_backend(prefer_cpu=False) == "sentence-transformers"
        )

    def test_falls_back_when_the_onnx_runtime_is_absent(self, monkeypatch):
        mod = self._module()
        monkeypatch.setattr(mod, "FASTEMBED_AVAILABLE", False)
        monkeypatch.setattr(mod, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
        assert (
            mod.get_best_available_backend(prefer_cpu=True) == "sentence-transformers"
        )

    def test_named_models_resolve_through_onnx_first(self):
        import inspect

        from alethia.embedder import _resolve_named_embedder

        source = inspect.getsource(_resolve_named_embedder)
        assert source.index("try_fastembed()") < source.index(
            "try_sentence_transformers()"
        )

    def test_onnx_extra_is_declared_and_torch_free(self):
        import pathlib

        tomllib = pytest.importorskip("tomllib", reason="stdlib from 3.11")

        root = pathlib.Path(__file__).resolve().parents[1]
        cfg = tomllib.loads((root / "pyproject.toml").read_text())
        extras = cfg["project"]["optional-dependencies"]
        assert "onnx" in extras
        joined = " ".join(extras["onnx"]).lower()
        assert "fastembed" in joined
        assert "torch" not in joined
        assert "sentence-transformers" not in joined


class TestResultShapeIsStable:
    """The column set must not depend on what the input happened to contain."""

    def test_empty_query_list_still_has_columns(self):
        frame = alethia([], ["New York", "Chicago"])
        assert list(frame.columns)[:3] == [
            "given_entity",
            "alethia_prediction",
            "alethia_score",
        ]

    def test_all_exact_matches_return_the_same_columns_as_a_mixed_run(self):
        exact = alethia(["a", "b"], ["a", "b", "c"])
        mixed = alethia(["a", "bx"], ["a", "b", "c"])
        assert list(exact.columns) == list(mixed.columns)

    def test_all_exact_matches_honour_return_model_attrs(self):
        frame = alethia(["a", "b"], ["a", "b", "c"], return_model_attrs=False)
        assert "alethia_method" not in frame.columns
        assert "alethia_backend" not in frame.columns
