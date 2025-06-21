#!/usr/bin/env python

"""Tests for `alethia` package."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from alethia import alethia
from alethia.alethia import (
    check_optional_dependencies,
    get_best_available_backend,
    run_rapidfuzz_matching,
    run_openai_matching,
    run_gemini_matching,
    optimized_batch_matching,
    standard_matching,
    _find_exact_matches,
    _is_nan_entry,
    _preprocess_entries_with_nans,
    _filter_nan_entries,
)


# Test data fixtures
@pytest.fixture
def sample_dirty_entries():
    """Sample dirty entries for testing"""
    return ["NY", "LA", "Chiacgo", "Houston City", "San Fran"]


@pytest.fixture
def sample_reference_entries():
    """Sample reference entries for testing"""
    return ["New York", "Los Angeles", "Chicago", "Houston", "San Francisco"]


@pytest.fixture
def sample_entries_with_nans():
    """Sample entries with NaN values"""
    return ["NY", "LA", np.nan, "Chicago", None, "Houston", ""]


@pytest.fixture
def sample_reference_with_nans():
    """Sample reference entries with NaN values"""
    return ["New York", "Los Angeles", np.nan, "Chicago", "Houston", None]


class TestDependencyChecking:
    """Test dependency checking functionality"""
    
    def test_check_optional_dependencies(self):
        """Test dependency checking returns proper structure"""
        deps = check_optional_dependencies()
        expected_keys = {
            "SENTENCE_TRANSFORMERS_AVAILABLE",
            "FASTEMBED_AVAILABLE", 
            "RAPIDFUZZ_AVAILABLE",
            "FAISS_AVAILABLE",
            "NUMBA_AVAILABLE",
            "OPENAI_AVAILABLE",
            "GEMINI_AVAILABLE",
        }
        assert set(deps.keys()) == expected_keys
        assert all(isinstance(v, bool) for v in deps.values())
    
    def test_get_best_available_backend(self):
        """Test backend selection logic"""
        backend = get_best_available_backend()
        assert backend in ["sentence-transformers", "fastembed", "openai", "gemini", "rapidfuzz", "exact"]
        
        backend_cpu = get_best_available_backend(prefer_cpu=True)
        assert backend_cpu in ["fastembed", "sentence-transformers", "openai", "gemini", "rapidfuzz", "exact"]


class TestRapidFuzzMatching:
    """Test RapidFuzz-based matching functionality"""
    
    @pytest.mark.skipif(not check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"], 
                       reason="RapidFuzz not available")
    def test_run_rapidfuzz_matching_basic(self, sample_dirty_entries, sample_reference_entries):
        """Test basic RapidFuzz matching"""
        result = run_rapidfuzz_matching(sample_dirty_entries, sample_reference_entries)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dirty_entries)
        assert list(result.columns) == ["given_entity", "alethia_prediction", "alethia_score"]
        
        # Check that scores are between 0 and 1
        assert all(0 <= score <= 1 for score in result["alethia_score"])
        
        # Check that all entries are matched
        assert len(result["given_entity"].unique()) == len(sample_dirty_entries)
    
    @pytest.mark.skipif(not check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"], 
                       reason="RapidFuzz not available")
    def test_rapidfuzz_exact_matches(self):
        """Test that exact matches get perfect scores"""
        dirty = ["New York", "Los Angeles"]
        reference = ["New York", "Los Angeles", "Chicago"]
        
        result = run_rapidfuzz_matching(dirty, reference)
        
        # Exact matches should have score 1.0
        for _, row in result.iterrows():
            if row["given_entity"] == row["alethia_prediction"]:
                assert row["alethia_score"] == 1.0
    
    @pytest.mark.skipif(not check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"], 
                       reason="RapidFuzz not available")
    def test_rapidfuzz_with_empty_inputs(self):
        """Test RapidFuzz with empty inputs"""
        result = run_rapidfuzz_matching([], [])
        assert len(result) == 0
        
        # Test with empty reference - rapidfuzz returns None when no matches
        result = run_rapidfuzz_matching(["test"], [])
        assert len(result) == 1
        # Should return the original entry when no matches possible
        assert result.iloc[0]["given_entity"] == "test"


class TestSentenceTransformersMatching:
    """Test Sentence Transformers-based matching functionality"""
    
    @pytest.mark.skipif(not check_optional_dependencies()["SENTENCE_TRANSFORMERS_AVAILABLE"], 
                       reason="Sentence Transformers not available")
    def test_alethia_sentence_transformers(self, sample_dirty_entries, sample_reference_entries):
        """Test alethia with sentence-transformers backend"""
        result = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="sentence-transformers",
            force_cpu=True,
            verbose=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dirty_entries)
        assert list(result.columns) == ["given_entity", "alethia_prediction", "alethia_score", "alethia_method", "alethia_backend"]
        
        # Check that scores are between 0 and 1
        assert all(0 <= score <= 1 for score in result["alethia_score"])
        
        # Check backend is correctly set
        assert all(backend == "sentence-transformers" for backend in result["alethia_backend"])
    
    @pytest.mark.skipif(not check_optional_dependencies()["SENTENCE_TRANSFORMERS_AVAILABLE"], 
                       reason="Sentence Transformers not available")
    def test_sentence_transformers_batch_optimization(self, sample_dirty_entries, sample_reference_entries):
        """Test batch optimization with sentence transformers"""
        # Test with batch optimization enabled
        result_batch = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="sentence-transformers",
            use_batch_optimization=True,
            force_cpu=True,
            verbose=False
        )
        
        # Test with batch optimization disabled
        result_standard = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="sentence-transformers",
            use_batch_optimization=False,
            force_cpu=True,
            verbose=False
        )
        
        # Both should have same structure
        assert len(result_batch) == len(result_standard)
        assert list(result_batch.columns) == list(result_standard.columns)
    
    @pytest.mark.skipif(not check_optional_dependencies()["SENTENCE_TRANSFORMERS_AVAILABLE"], 
                       reason="Sentence Transformers not available")
    def test_sentence_transformers_with_threshold(self, sample_dirty_entries, sample_reference_entries):
        """Test sentence transformers with different thresholds"""
        # High threshold - should return more original entries
        result_high = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="sentence-transformers",
            threshold=0.9,
            force_cpu=True,
            verbose=False
        )
        
        # Low threshold - should return more matches
        result_low = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="sentence-transformers",
            threshold=0.1,
            force_cpu=True,
            verbose=False
        )
        
        # Count entries that stayed the same (given_entity == alethia_prediction)
        unchanged_high = sum(result_high["given_entity"] == result_high["alethia_prediction"])
        unchanged_low = sum(result_low["given_entity"] == result_low["alethia_prediction"])
        
        # High threshold should have more unchanged entries
        assert unchanged_high >= unchanged_low


class TestFastEmbedMatching:
    """Test FastEmbed-based matching functionality"""
    
    @pytest.mark.skipif(not check_optional_dependencies()["FASTEMBED_AVAILABLE"], 
                       reason="FastEmbed not available")
    def test_alethia_fastembed(self, sample_dirty_entries, sample_reference_entries):
        """Test alethia with fastembed backend"""
        result = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="fastembed",
            verbose=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dirty_entries)
        assert list(result.columns) == ["given_entity", "alethia_prediction", "alethia_score", "alethia_method", "alethia_backend"]
        
        # Check that scores are between 0 and 1
        assert all(0 <= score <= 1 for score in result["alethia_score"])
        
        # Check backend is correctly set
        assert all(backend == "fastembed" for backend in result["alethia_backend"])
    
    @pytest.mark.skipif(not check_optional_dependencies()["FASTEMBED_AVAILABLE"], 
                       reason="FastEmbed not available")
    def test_fastembed_batch_vs_standard(self, sample_dirty_entries, sample_reference_entries):
        """Test FastEmbed batch vs standard processing"""
        # Test with batch optimization
        result_batch = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="fastembed",
            use_batch_optimization=True,
            verbose=False
        )
        
        # Test with standard processing
        result_standard = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            model="all-MiniLM-L6-v2",
            backend="fastembed",
            use_batch_optimization=False,
            verbose=False
        )
        
        # Both should produce similar results
        assert len(result_batch) == len(result_standard)
        assert list(result_batch.columns) == list(result_standard.columns)


class TestAutoBackendSelection:
    """Test automatic backend selection"""
    
    def test_auto_backend_selection(self, sample_dirty_entries, sample_reference_entries):
        """Test that auto backend selection works"""
        result = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            backend="auto",
            verbose=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dirty_entries)
        
        # Should have selected some backend
        if "alethia_backend" in result.columns:
            assert result["alethia_backend"].iloc[0] in [
                "sentence-transformers", "fastembed", "openai", "gemini", "rapidfuzz", "auto"
            ]
    
    def test_fallback_to_rapidfuzz(self, sample_dirty_entries, sample_reference_entries):
        """Test fallback to rapidfuzz when other backends fail"""
        # This test mocks a failure in the primary backend to test fallback
        with patch('alethia.alethia.load_sentence_transformer_model', return_value=None):
            with patch('alethia.alethia.load_fastembed_model', return_value=None):
                result = alethia(
                    sample_dirty_entries,
                    sample_reference_entries,
                    model="all-MiniLM-L6-v2",
                    backend="sentence-transformers",
                    verbose=False
                )
                
                assert isinstance(result, pd.DataFrame)
                assert len(result) == len(sample_dirty_entries)


class TestExactMatching:
    """Test exact matching functionality"""
    
    def test_find_exact_matches_case_sensitive(self):
        """Test exact matching with case sensitivity"""
        dirty = ["New York", "new york", "LA", "Chicago"]
        reference = ["New York", "Los Angeles", "Chicago"]
        
        exact_matches, remaining, remaining_indices = _find_exact_matches(
            dirty, reference, case_sensitive=True
        )
        
        assert len(exact_matches) == 2  # "New York" and "Chicago"
        assert len(remaining) == 2  # "new york" and "LA"
        assert 0 in exact_matches  # "New York" at index 0
        assert 3 in exact_matches  # "Chicago" at index 3
    
    def test_find_exact_matches_case_insensitive(self):
        """Test exact matching without case sensitivity"""
        dirty = ["New York", "new york", "LA", "Chicago"]
        reference = ["New York", "Los Angeles", "Chicago"]
        
        exact_matches, remaining, remaining_indices = _find_exact_matches(
            dirty, reference, case_sensitive=False
        )
        
        assert len(exact_matches) == 3  # "New York", "new york", "Chicago"
        assert len(remaining) == 1  # "LA"
        assert 0 in exact_matches
        assert 1 in exact_matches
        assert 3 in exact_matches
    
    def test_exact_matching_integration(self, sample_dirty_entries, sample_reference_entries):
        """Test exact matching integration in main alethia function"""
        # Add some exact matches to test data
        dirty_with_exact = sample_dirty_entries + ["Chicago", "Houston"]
        
        result = alethia(
            dirty_with_exact,
            sample_reference_entries,
            use_exact_matching=True,
            verbose=False
        )
        
        # Check that exact matches have score 1.0
        exact_matches = result[result["given_entity"] == result["alethia_prediction"]]
        for _, row in exact_matches.iterrows():
            if row["given_entity"] in sample_reference_entries:
                assert row["alethia_score"] == 1.0


class TestNaNHandling:
    """Test NaN and None value handling"""
    
    def test_is_nan_entry(self):
        """Test NaN detection function"""
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
        """Test preprocessing of entries with NaN values"""
        processed, nan_mask, original_indices = _preprocess_entries_with_nans(sample_entries_with_nans)
        
        # Should filter out NaN entries
        assert len(processed) == 4  # "NY", "LA", "Chicago", "Houston"
        assert len(nan_mask) == len(sample_entries_with_nans)
        assert sum(nan_mask) == 3  # np.nan, None, ""
        assert len(original_indices) == 4
    
    def test_filter_nan_entries(self, sample_reference_with_nans):
        """Test filtering NaN entries from reference list"""
        filtered = _filter_nan_entries(sample_reference_with_nans)
        
        assert len(filtered) == 4  # Should remove np.nan and None
        assert np.nan not in filtered
        assert None not in filtered
        assert "New York" in filtered
        assert "Los Angeles" in filtered
    
    def test_alethia_with_nan_entries(self, sample_entries_with_nans, sample_reference_entries):
        """Test alethia function with NaN entries"""
        result = alethia(
            sample_entries_with_nans,
            sample_reference_entries,
            verbose=False
        )
        
        # The result might have fewer entries if duplicates are dropped by default
        assert len(result) <= len(sample_entries_with_nans)
        
        # Check that at least some NaN entries are preserved
        nan_results = result[result["given_entity"].isna() | 
                           result["given_entity"].isin(["", "nan", "null", "none"])]
        
        # NaN entries should have NaN predictions
        for _, row in nan_results.iterrows():
            if pd.isna(row["given_entity"]) or str(row["given_entity"]).lower() in ["nan", "null", "none", ""]:
                assert pd.isna(row["alethia_prediction"]) or pd.isna(row["alethia_score"])


class TestAPIBackends:
    """Test API-based backends (OpenAI, Gemini)"""
    
    @pytest.mark.api
    @pytest.mark.skipif(not check_optional_dependencies()["OPENAI_AVAILABLE"], 
                       reason="OpenAI not available")
    def test_openai_matching_mock(self, sample_dirty_entries, sample_reference_entries):
        """Test OpenAI matching with mocked API calls"""
        with patch('alethia.alethia.setup_openai_client') as mock_client_setup:
            with patch('alethia.alethia.get_openai_embedding') as mock_embedding:
                # Mock the client setup
                mock_client = MagicMock()
                mock_client_setup.return_value = mock_client
                
                # Mock embeddings - return same size vectors for all inputs
                def mock_embedding_func(client, text, model):
                    # Return fixed-size vectors to avoid dimension mismatch
                    return [0.1] * 100
                
                mock_embedding.side_effect = mock_embedding_func
                
                result = run_openai_matching(
                    sample_dirty_entries,
                    sample_reference_entries,
                    threshold=0.5
                )
                
                assert isinstance(result, pd.DataFrame)
                assert len(result) == len(sample_dirty_entries)
                assert list(result.columns) == ["given_entity", "alethia_prediction", "alethia_score"]
    
    @pytest.mark.api
    @pytest.mark.skipif(not check_optional_dependencies()["GEMINI_AVAILABLE"], 
                       reason="Gemini not available")
    def test_gemini_matching_mock(self, sample_dirty_entries, sample_reference_entries):
        """Test Gemini matching with mocked API calls"""
        with patch('alethia.alethia.setup_gemini_client') as mock_client_setup:
            with patch('alethia.alethia.get_gemini_embedding') as mock_embedding:
                # Mock the client setup
                mock_client_setup.return_value = MagicMock()
                
                # Mock embeddings - return same size vectors
                def mock_embedding_func(text, model):
                    return [0.1] * 100
                
                mock_embedding.side_effect = mock_embedding_func
                
                result = run_gemini_matching(
                    sample_dirty_entries,
                    sample_reference_entries,
                    threshold=0.5
                )
                
                assert isinstance(result, pd.DataFrame)
                assert len(result) == len(sample_dirty_entries)
                assert list(result.columns) == ["given_entity", "alethia_prediction", "alethia_score"]


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_inputs(self):
        """Test with empty inputs"""
        result = alethia([], [], verbose=False)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_single_entry(self):
        """Test with single entry"""
        result = alethia(["test"], ["test"], verbose=False)
        assert len(result) == 1
        assert result.iloc[0]["given_entity"] == "test"
        assert result.iloc[0]["alethia_prediction"] == "test"
        assert result.iloc[0]["alethia_score"] == 1.0
    
    def test_no_reference_entries(self):
        """Test with no reference entries"""
        result = alethia(["test"], [], verbose=False)
        assert len(result) == 1
        # Should return the original entry when no references available
        assert result.iloc[0]["given_entity"] == "test"
    
    def test_all_nan_entries(self):
        """Test with all NaN entries"""
        result = alethia([np.nan, None, ""], ["test"], verbose=False)
        assert len(result) == 3
        # All should have NaN predictions
        for _, row in result.iterrows():
            assert pd.isna(row["alethia_prediction"]) or pd.isna(row["alethia_score"])
    
    def test_duplicate_handling(self):
        """Test duplicate handling"""
        dirty = ["NY", "NY", "LA"]
        reference = ["New York", "Los Angeles"]
        
        # Test with drop_duplicates=True
        result_drop = alethia(dirty, reference, drop_duplicates=True, verbose=False)
        
        # Test with drop_duplicates=False
        result_keep = alethia(dirty, reference, drop_duplicates=False, verbose=False)
        
        assert len(result_keep) == 3  # Should keep all entries
        assert len(result_drop) <= len(result_keep)  # Should have fewer or equal entries
    
    def test_remove_identical_hits(self):
        """Test removing identical hits"""
        dirty = ["New York", "NY"]
        reference = ["New York", "Los Angeles"]
        
        # Test with remove_identical_hits=True
        result_remove = alethia(dirty, reference, remove_identical_hits=True, verbose=False)
        
        # Test with remove_identical_hits=False
        result_keep = alethia(dirty, reference, remove_identical_hits=False, verbose=False)
        
        assert len(result_keep) == 2
        # result_remove should have fewer entries (exact matches removed)
        assert len(result_remove) <= len(result_keep)
    
    def test_invalid_backend(self, sample_dirty_entries, sample_reference_entries):
        """Test with invalid backend"""
        # Invalid backend should either raise ValueError or fallback to available backend
        try:
            result = alethia(
                sample_dirty_entries,
                sample_reference_entries,
                backend="invalid_backend",
                verbose=False
            )
            # If it doesn't raise, it should have fallen back to a valid backend
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_dirty_entries)
        except ValueError:
            # This is also acceptable behavior
            pass
    
    def test_return_model_attrs(self, sample_dirty_entries, sample_reference_entries):
        """Test return_model_attrs parameter"""
        # With return_model_attrs=True
        result_with_attrs = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            return_model_attrs=True,
            verbose=False
        )
        
        # With return_model_attrs=False
        result_without_attrs = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            return_model_attrs=False,
            verbose=False
        )
        
        # Check column differences
        base_columns = ["given_entity", "alethia_prediction", "alethia_score"]
        attr_columns = ["alethia_method", "alethia_backend"]
        
        assert all(col in result_with_attrs.columns for col in base_columns + attr_columns)
        assert all(col in result_without_attrs.columns for col in base_columns)
        assert not any(col in result_without_attrs.columns for col in attr_columns)


class TestPerformanceAttributes:
    """Test performance attributes and metadata"""
    
    def test_result_attributes(self, sample_dirty_entries, sample_reference_entries):
        """Test that results have proper attributes"""
        result = alethia(
            sample_dirty_entries,
            sample_reference_entries,
            verbose=False
        )
        
        # Check that result has performance attributes
        expected_attrs = [
            "acceleration",
            "backend", 
            "processing_time",
            "model",
            "nan_entries_count",
            "processed_entries_count"
        ]
        
        for attr in expected_attrs:
            assert attr in result.attrs
        
        # Check types
        assert isinstance(result.attrs["processing_time"], (int, float))
        assert isinstance(result.attrs["nan_entries_count"], int)
        assert isinstance(result.attrs["processed_entries_count"], int)
        assert result.attrs["processing_time"] >= 0
    
    def test_verbose_mode(self, sample_dirty_entries, sample_reference_entries, capsys):
        """Test verbose mode output"""
        alethia(
            sample_dirty_entries,
            sample_reference_entries,
            verbose=True
        )
        
        captured = capsys.readouterr()
        # Should have some verbose output
        assert len(captured.out) > 0
