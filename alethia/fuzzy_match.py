import re
import warnings
from typing import Dict, List

import pandas as pd


class FuzzyLibraryManager:
    def __init__(self):
        self.rapidfuzz = self.fuzzywuzzy = self.jellyfish = None
        self.rapidfuzz_available = self.fuzzywuzzy_available = (
            self.jellyfish_available
        ) = False
        self._check_imports()

    def _check_imports(self):
        try:
            from rapidfuzz import fuzz, process

            self.rapidfuzz = {"fuzz": fuzz, "process": process}
            self.rapidfuzz_available = True
        except ImportError:
            pass

        try:
            from fuzzywuzzy import fuzz, process

            self.fuzzywuzzy = {"fuzz": fuzz, "process": process}
            self.fuzzywuzzy_available = True
        except ImportError:
            pass

        try:
            import jellyfish

            self.jellyfish = jellyfish
            self.jellyfish_available = True
        except ImportError:
            pass

        if not self.rapidfuzz_available and not self.fuzzywuzzy_available:
            warnings.warn(
                "No fuzzy matching library available. Install rapidfuzz or fuzzywuzzy.",
                ImportWarning,
            )

    def get_primary_library(self):
        if self.rapidfuzz_available:
            return "rapidfuzz", self.rapidfuzz
        if self.fuzzywuzzy_available:
            return "fuzzywuzzy", self.fuzzywuzzy
        return None, None

    def get_algorithm_function(self, algorithm: str):
        _, lib = self.get_primary_library()
        if lib is None:
            return None
        algo_map = {
            "ratio": "ratio",
            "partial_ratio": "partial_ratio",
            "token_sort_ratio": "token_sort_ratio",
            "token_set_ratio": "token_set_ratio",
            "WRatio": "WRatio",
        }
        return getattr(lib["fuzz"], algo_map.get(algorithm, "ratio"), None)

    def get_process_extract(self):
        _, lib = self.get_primary_library()
        return lib["process"].extract if lib else None


lib_manager = FuzzyLibraryManager()


class RobustFuzzyMatcher:
    def __init__(self, algorithm: str = "ratio", preprocessor: str = "simple"):
        self.algorithm = algorithm
        self.algorithm_func = lib_manager.get_algorithm_function(algorithm)
        self.process_extract = lib_manager.get_process_extract()
        self.preprocessor = self._get_preprocessor(preprocessor)

    def _get_preprocessor(self, name):
        if name == "none":
            return lambda x: x
        if name == "smart":
            return self._smart_preprocess
        return self._simple_preprocess

    def _simple_preprocess(self, text: str) -> str:
        return text.lower().strip() if text else ""

    def _smart_preprocess(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text.strip())

    def calculate_similarity(self, text1: str, text2: str) -> float:
        proc1, proc2 = self.preprocessor(text1), self.preprocessor(text2)
        if self.algorithm_func is None:
            return 100.0 if proc1 == proc2 else 0.0
        if not proc1 and not proc2:
            return 100.0
        if not proc1 or not proc2:
            return 0.0
        return float(self.algorithm_func(proc1, proc2))

    def match_single(
        self,
        query: str,
        candidates: List[str],
        limit: int = 5,
        score_cutoff: float = 0.0,
    ) -> List[Dict]:
        if not candidates:
            return []

        if self.process_extract and self.algorithm_func:
            processed = [self.preprocessor(c) for c in candidates]
            extracted = self.process_extract(
                self.preprocessor(query),
                processed,
                scorer=self.algorithm_func,
                limit=limit,
                score_cutoff=score_cutoff,
            )
            results = []
            for match in extracted:
                idx = (
                    match[2]
                    if len(match) >= 3
                    else processed.index(match[0]) if match[0] in processed else 0
                )
                results.append(
                    {"text": candidates[idx], "score": match[1], "index": idx}
                )
            return results

        scores = [
            {"text": c, "score": self.calculate_similarity(query, c), "index": i}
            for i, c in enumerate(candidates)
            if self.calculate_similarity(query, c) >= score_cutoff
        ]
        return sorted(scores, key=lambda x: x["score"], reverse=True)[:limit]

    def match_batch(
        self,
        queries: List[str],
        candidates: List[str],
        limit: int = 1,
        score_cutoff: float = 0.0,
    ) -> pd.DataFrame:
        results = []
        for query in queries:
            matches = self.match_single(query, candidates, limit, score_cutoff)
            if matches:
                results.append(
                    {
                        "query": query,
                        "match": matches[0]["text"],
                        "score": matches[0]["score"],
                        "index": matches[0]["index"],
                        "all_matches": matches,
                    }
                )
            else:
                results.append(
                    {
                        "query": query,
                        "match": None,
                        "score": 0.0,
                        "index": -1,
                        "all_matches": [],
                    }
                )
        return pd.DataFrame(results)


def robust_fuzzy_match(
    queries: List[str],
    candidates: List[str],
    algorithm: str = "ratio",
    threshold: float = 70.0,
    preprocessor: str = "simple",
) -> pd.DataFrame:
    matcher = RobustFuzzyMatcher(algorithm=algorithm, preprocessor=preprocessor)
    return matcher.match_batch(queries, candidates, limit=1, score_cutoff=threshold)


def alethia_fuzzy_baseline(
    incorrect_entries: List[str], reference_entries: List[str], threshold: float = 70.0
) -> pd.DataFrame:
    algorithm = (
        "WRatio"
        if lib_manager.rapidfuzz_available or lib_manager.fuzzywuzzy_available
        else "exact"
    )
    results = robust_fuzzy_match(
        incorrect_entries,
        reference_entries,
        algorithm=algorithm,
        threshold=threshold,
        preprocessor="smart",
    )
    return pd.DataFrame(
        {
            "given_entity": results["query"],
            "fuzzy_prediction": results["match"],
            "fuzzy_similarity": results["score"] / 100.0,
            "fuzzy_algorithm": algorithm,
            "fuzzy_library": lib_manager.get_primary_library()[0] or "exact_only",
        }
    )
