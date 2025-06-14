"""
fuzzy matching for alethia
"""

import logging
import sys
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class FuzzyLibraryManager:
    """Manages fuzzy matching library imports and availability"""

    def __init__(self):
        self.rapidfuzz_available = False
        self.fuzzywuzzy_available = False
        self.jellyfish_available = False

        self.rapidfuzz = None
        self.fuzzywuzzy = None
        self.jellyfish = None

        self._check_imports()

    def _check_imports(self):
        """Check which libraries are available"""

        # Check RapidFuzz
        try:
            import rapidfuzz
            from rapidfuzz import fuzz as rf_fuzz
            from rapidfuzz import process as rf_process

            self.rapidfuzz = {
                "module": rapidfuzz,
                "fuzz": rf_fuzz,
                "process": rf_process,
            }
            self.rapidfuzz_available = True
            logger.info("✅ RapidFuzz available")
        except ImportError:
            logger.debug("RapidFuzz not available")

        # Check FuzzyWuzzy
        try:
            import fuzzywuzzy
            from fuzzywuzzy import fuzz as fw_fuzz
            from fuzzywuzzy import process as fw_process

            self.fuzzywuzzy = {
                "module": fuzzywuzzy,
                "fuzz": fw_fuzz,
                "process": fw_process,
            }
            self.fuzzywuzzy_available = True
            logger.info("✅ FuzzyWuzzy available")
        except ImportError:
            logger.debug("FuzzyWuzzy not available")

        # Check Jellyfish
        try:
            import jellyfish

            self.jellyfish = jellyfish
            self.jellyfish_available = True
            logger.debug("✅ Jellyfish available")
        except ImportError:
            logger.debug("Jellyfish not available")

        # Issue warnings if nothing is available
        if not self.rapidfuzz_available and not self.fuzzywuzzy_available:
            warnings.warn(
                "No fuzzy matching library available. Install rapidfuzz or fuzzywuzzy.",
                ImportWarning,
            )
        elif not self.rapidfuzz_available and self.fuzzywuzzy_available:
            warnings.warn(
                "Using FuzzyWuzzy. Install rapidfuzz for better performance: pip install rapidfuzz",
                UserWarning,
            )

    def get_primary_library(self):
        """Get the primary library to use (prefer RapidFuzz)"""
        if self.rapidfuzz_available:
            return "rapidfuzz", self.rapidfuzz
        elif self.fuzzywuzzy_available:
            return "fuzzywuzzy", self.fuzzywuzzy
        else:
            return None, None

    def get_algorithm_function(self, algorithm: str):
        """Get the appropriate algorithm function"""
        lib_name, lib = self.get_primary_library()

        if lib is None:
            return None

        # Map algorithm names to functions
        algorithm_map = {
            "ratio": "ratio",
            "partial_ratio": "partial_ratio",
            "token_sort_ratio": "token_sort_ratio",
            "token_set_ratio": "token_set_ratio",
            "WRatio": "WRatio",
        }

        func_name = algorithm_map.get(algorithm, "ratio")

        try:
            return getattr(lib["fuzz"], func_name)
        except AttributeError:
            logger.warning(
                f"Algorithm {algorithm} not available in {lib_name}, using ratio"
            )
            return getattr(lib["fuzz"], "ratio")

    def get_process_extract(self):
        """Get the process.extract function"""
        lib_name, lib = self.get_primary_library()

        if lib is None:
            return None

        return lib["process"].extract


# Global library manager
lib_manager = FuzzyLibraryManager()


class RobustFuzzyMatcher:
    """
    Robust fuzzy matcher that works with available libraries
    """

    def __init__(self, algorithm: str = "ratio", preprocessor: str = "simple"):
        """
        Initialize the matcher

        Args:
            algorithm: Algorithm to use ('ratio', 'partial_ratio', 'token_sort_ratio',
                      'token_set_ratio', 'WRatio')
            preprocessor: Preprocessing method ('none', 'simple', 'smart')
        """
        self.algorithm = algorithm
        self.preprocessor_name = preprocessor

        # Get algorithm function
        self.algorithm_func = lib_manager.get_algorithm_function(algorithm)
        self.process_extract = lib_manager.get_process_extract()

        # Set up preprocessor
        self.preprocessor = self._get_preprocessor(preprocessor)

        # Check if we can actually do fuzzy matching
        if self.algorithm_func is None:
            logger.warning(
                "No fuzzy matching available - only exact matching will work"
            )

    def _get_preprocessor(self, preprocessor: str):
        """Get preprocessing function"""
        if preprocessor == "none":
            return lambda x: x
        elif preprocessor == "simple":
            return self._simple_preprocess
        elif preprocessor == "smart":
            return self._smart_preprocess
        else:
            logger.warning(f"Unknown preprocessor {preprocessor}, using simple")
            return self._simple_preprocess

    def _simple_preprocess(self, text: str) -> str:
        """Simple preprocessing"""
        if not text:
            return ""
        return text.lower().strip()

    def _smart_preprocess(self, text: str) -> str:
        """Smart preprocessing"""
        if not text:
            return ""

        import re

        # Convert to lowercase and strip
        text = text.lower().strip()

        # Remove extra punctuation and normalize whitespace
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text.strip())

        return text

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if self.algorithm_func is None:
            # Exact matching fallback
            proc1 = self.preprocessor(text1)
            proc2 = self.preprocessor(text2)
            return 100.0 if proc1 == proc2 else 0.0

        # Preprocess
        proc1 = self.preprocessor(text1)
        proc2 = self.preprocessor(text2)

        if not proc1 and not proc2:
            return 100.0
        if not proc1 or not proc2:
            return 0.0

        try:
            return float(self.algorithm_func(proc1, proc2))
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def match_single(
        self,
        query: str,
        candidates: List[str],
        limit: int = 5,
        score_cutoff: float = 0.0,
    ) -> List[Dict]:
        """
        Find best matches for a single query
        """
        if not candidates:
            return []

        results = []

        # Try using optimized process.extract if available
        if self.process_extract and self.algorithm_func:
            try:
                processed_candidates = [self.preprocessor(c) for c in candidates]
                processed_query = self.preprocessor(query)

                extracted = self.process_extract(
                    processed_query,
                    processed_candidates,
                    scorer=self.algorithm_func,
                    limit=limit,
                    score_cutoff=score_cutoff,
                )

                # Convert to our format
                for match in extracted:
                    if len(match) >= 3:  # (text, score, index)
                        results.append(
                            {
                                "text": candidates[match[2]],
                                "score": match[1],
                                "index": match[2],
                            }
                        )
                    else:
                        # Handle different return formats
                        idx = (
                            processed_candidates.index(match[0])
                            if match[0] in processed_candidates
                            else 0
                        )
                        results.append(
                            {"text": candidates[idx], "score": match[1], "index": idx}
                        )

                return results

            except Exception as e:
                logger.warning(f"process.extract failed: {e}, using manual calculation")

        # Manual calculation fallback
        scores = []
        for i, candidate in enumerate(candidates):
            score = self.calculate_similarity(query, candidate)
            if score >= score_cutoff:
                scores.append({"text": candidate, "score": score, "index": i})

        # Sort by score and return top results
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:limit]

    def match_batch(
        self,
        queries: List[str],
        candidates: List[str],
        limit: int = 1,
        score_cutoff: float = 0.0,
    ) -> pd.DataFrame:
        """
        Batch matching
        """
        results = []

        for query in queries:
            matches = self.match_single(query, candidates, limit, score_cutoff)

            if matches:
                best_match = matches[0]
                results.append(
                    {
                        "query": query,
                        "match": best_match["text"],
                        "score": best_match["score"],
                        "index": best_match["index"],
                        "all_matches": matches,
                    }
                )
            else:
                results.append(
                    {
                        "query": query,
                        "match": None,  # No match found
                        "score": 0.0,
                        "index": -1,
                        "all_matches": [],
                    }
                )

        return pd.DataFrame(results)

    def get_library_info(self) -> Dict:
        """Get information about available libraries"""
        return {
            "rapidfuzz_available": lib_manager.rapidfuzz_available,
            "fuzzywuzzy_available": lib_manager.fuzzywuzzy_available,
            "jellyfish_available": lib_manager.jellyfish_available,
            "primary_library": lib_manager.get_primary_library()[0],
            "algorithm": self.algorithm,
            "algorithm_available": self.algorithm_func is not None,
        }


def robust_fuzzy_match(
    queries: List[str],
    candidates: List[str],
    algorithm: str = "ratio",
    threshold: float = 70.0,
    preprocessor: str = "simple",
) -> pd.DataFrame:
    """
    High-level robust fuzzy matching function

    Args:
        queries: List of query strings
        candidates: List of candidate strings
        algorithm: Algorithm to use
        threshold: Minimum score threshold
        preprocessor: Preprocessing method

    Returns:
        DataFrame with results
    """
    matcher = RobustFuzzyMatcher(algorithm=algorithm, preprocessor=preprocessor)

    results = matcher.match_batch(queries, candidates, limit=1, score_cutoff=threshold)

    # Log library info
    info = matcher.get_library_info()
    logger.info(
        f"Using {info['primary_library']} library with {info['algorithm']} algorithm"
    )

    return results


def alethia_fuzzy_baseline(
    incorrect_entries: List[str], reference_entries: List[str], threshold: float = 70.0
) -> pd.DataFrame:
    """
    Create Alethia-compatible fuzzy baseline results

    Args:
        incorrect_entries: Entries to match
        reference_entries: Reference entries to match against
        threshold: Similarity threshold (0-100)

    Returns:
        DataFrame in Alethia-compatible format
    """
    # Use best available algorithm
    best_algorithm = "WRatio"
    if not lib_manager.rapidfuzz_available and not lib_manager.fuzzywuzzy_available:
        best_algorithm = "exact"

    results = robust_fuzzy_match(
        incorrect_entries,
        reference_entries,
        algorithm=best_algorithm,
        threshold=threshold,
        preprocessor="smart",
    )

    # Convert to Alethia format
    alethia_format = pd.DataFrame(
        {
            "given_entity": results["query"],
            "fuzzy_prediction": results["match"],
            "fuzzy_similarity": results["score"] / 100.0,  # Normalize to 0-1 range
            "fuzzy_algorithm": best_algorithm,
            "fuzzy_library": lib_manager.get_primary_library()[0] or "exact_only",
        }
    )

    return alethia_format


def compare_algorithms(queries: List[str], candidates: List[str]) -> pd.DataFrame:
    """
    Compare available algorithms

    Args:
        queries: Test queries
        candidates: Test candidates

    Returns:
        Comparison DataFrame
    """
    algorithms = [
        "ratio",
        "partial_ratio",
        "token_sort_ratio",
        "token_set_ratio",
        "WRatio",
    ]
    results = {}

    for algorithm in algorithms:
        try:
            matcher = RobustFuzzyMatcher(algorithm=algorithm)
            if matcher.algorithm_func is not None:
                algo_results = matcher.match_batch(queries, candidates)
                results[algorithm] = algo_results[["query", "match", "score"]].copy()
            else:
                logger.warning(f"Algorithm {algorithm} not available")
        except Exception as e:
            logger.error(f"Error with algorithm {algorithm}: {e}")

    if not results:
        logger.error("No algorithms available for comparison")
        return pd.DataFrame({"query": queries})

    # Combine results
    base_df = pd.DataFrame({"query": queries})

    for algo_name, df in results.items():
        base_df[f"{algo_name}_match"] = df["match"].values
        base_df[f"{algo_name}_score"] = df["score"].values

    return base_df
