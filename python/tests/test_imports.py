"""Import-cost guarantees for the package surface."""

import subprocess
import sys

import pytest

HEAVY = ("torch", "sentence_transformers", "matplotlib", "umap")


def _loaded_after(statement):
    probe = (
        "import sys\n"
        f"{statement}\n"
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    return [m for m in out.stdout.strip().split(",") if m]


class TestLazyExports:
    def test_every_public_name_resolves(self):
        import alethia

        unresolvable = [n for n in alethia.__all__ if not hasattr(alethia, n)]
        assert not unresolvable

    def test_star_import_works(self):
        namespace = {}
        exec("from alethia import *", namespace)
        assert "alethia" in namespace and "cluster_entities" in namespace

    def test_unknown_attribute_still_raises(self):
        import alethia

        with pytest.raises(AttributeError):
            _ = alethia.definitely_not_a_real_name

    def test_dunder_version_available_without_submodules(self):
        import alethia

        assert alethia.__version__


class TestImportCost:
    @pytest.mark.parametrize(
        "statement",
        [
            "import alethia",
            "import alethia.cli",
            "import alethia; alethia.cluster_entities",
            "import alethia; alethia.RAPIDFUZZ_AVAILABLE",
        ],
    )
    def test_no_heavy_backend_is_imported(self, statement):
        assert _loaded_after(statement) == []

    def test_probe_agrees_with_real_importability(self):
        import importlib

        from alethia.alethia import _DEPENDENCY_MODULES, check_optional_dependencies

        deps = check_optional_dependencies()
        for flag, module in _DEPENDENCY_MODULES.items():
            if not deps[flag]:
                continue
            importlib.import_module(module)  # raises if the probe was wrong

    def test_dependency_probe_does_not_import_what_it_probes(self):
        from alethia.alethia import check_optional_dependencies

        deps = check_optional_dependencies()
        assert set(deps) == {
            "SENTENCE_TRANSFORMERS_AVAILABLE",
            "FASTEMBED_AVAILABLE",
            "RAPIDFUZZ_AVAILABLE",
            "OPENAI_AVAILABLE",
            "GEMINI_AVAILABLE",
        }
        assert all(isinstance(v, bool) for v in deps.values())


class TestSubmoduleShadowing:
    """`alethia.alethia` is both a submodule and the matching function."""

    def test_function_wins_whichever_is_imported_first(self):
        for statement in (
            "from alethia import alethia as f",
            "import alethia.alethia; from alethia import alethia as f",
            "from alethia.alethia import check_optional_dependencies\n"
            "from alethia import alethia as f",
        ):
            probe = f"{statement}\nprint(type(f).__name__)"
            out = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                check=True,
            )
            assert out.stdout.strip() == "function", statement

    def test_the_submodule_is_still_reachable(self):
        import importlib

        module = importlib.import_module("alethia.alethia")
        assert hasattr(module, "check_optional_dependencies")

    def test_shadow_fix_does_not_import_backends(self):
        assert _loaded_after("from alethia import alethia") == []


class TestPackagingContract:
    """What a `pip install` has to keep true, and what CI cannot see until PyPI."""

    def test_importing_the_library_writes_nothing_to_stdout(self):
        probe = "import alethia.alethia, alethia.cli, alethia.assess"
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert out.stdout == ""

    def test_version_matches_the_installed_distribution(self):
        from importlib.metadata import PackageNotFoundError, version

        import alethia

        try:
            installed = version("alethia")
        except PackageNotFoundError:
            pytest.skip("alethia is not installed in this environment")
        assert alethia.__version__ == installed

    def test_every_shipped_module_imports_without_optional_backends(self):
        import importlib
        import pkgutil

        import alethia

        broken = []
        for info in pkgutil.walk_packages(alethia.__path__, "alethia."):
            try:
                importlib.import_module(info.name)
            except Exception as exc:
                broken.append(f"{info.name}: {type(exc).__name__}: {exc}")
        assert not broken

    def test_the_default_matcher_is_a_hard_dependency(self):
        import pathlib

        tomllib = pytest.importorskip("tomllib", reason="stdlib from 3.11")
        root = pathlib.Path(__file__).resolve().parents[1]
        cfg = tomllib.loads((root / "pyproject.toml").read_text())
        required = " ".join(cfg["project"]["dependencies"]).lower()
        assert "rapidfuzz" in required

        from alethia.cli import DEFAULT_MATCH_MODEL

        assert DEFAULT_MATCH_MODEL == "rapidfuzz"
