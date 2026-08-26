"""CLI behaviour tests."""

import pandas as pd
import pytest
from typer.testing import CliRunner

from alethia.alethia import check_optional_dependencies
from alethia.cli import app

runner = CliRunner()

RAPIDFUZZ = check_optional_dependencies()["RAPIDFUZZ_AVAILABLE"]
needs_rapidfuzz = pytest.mark.skipif(not RAPIDFUZZ, reason="RapidFuzz not available")


@pytest.fixture
def messy(tmp_path):
    path = tmp_path / "messy.txt"
    path.write_text("Bombay\nCalcutta\nMumbai\n")
    return path


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "reference.txt"
    path.write_text("Mumbai\nKolkata\nDelhi\n")
    return path


class TestDiscovery:
    """A user who types `alethia` and nothing else must land somewhere useful."""

    def test_bare_invocation_shows_commands_not_an_error(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        for command in ("match", "cluster", "assess", "check"):
            assert command in result.output

    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "alethia" in result.output

    @pytest.mark.parametrize(
        "command", ["match", "cluster", "assess", "check", "models"]
    )
    def test_every_command_has_help(self, command):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert result.output.strip()

    def test_match_help_includes_a_worked_example(self):
        result = runner.invoke(app, ["match", "--help"])
        assert "alethia match" in result.output

    def test_check_reports_status(self):
        result = runner.invoke(app, ["check"])
        assert "Feature" in result.output


class TestMatching:
    @needs_rapidfuzz
    def test_matches_and_summarises(self, messy, reference):
        result = runner.invoke(app, ["match", str(messy), str(reference)])
        assert result.exit_code == 0
        assert "Bombay" in result.output
        assert "3 entries processed" in result.output

    @needs_rapidfuzz
    def test_writes_output_file(self, messy, reference, tmp_path):
        out = tmp_path / "out.csv"
        result = runner.invoke(
            app, ["match", str(messy), str(reference), "-o", str(out)]
        )
        assert result.exit_code == 0
        assert out.exists()
        frame = pd.read_csv(out)
        assert list(frame["given_entity"]) == ["Bombay", "Calcutta", "Mumbai"]

    @needs_rapidfuzz
    def test_reads_csv_with_named_column(self, tmp_path, reference):
        source = tmp_path / "data.csv"
        source.write_text("id,city\n1,Bombay\n2,Calcutta\n")
        result = runner.invoke(
            app, ["match", str(source), str(reference), "--column", "city"]
        )
        assert result.exit_code == 0
        assert "Bombay" in result.output

    @needs_rapidfuzz
    def test_column_name_is_case_insensitive(self, tmp_path, reference):
        source = tmp_path / "data.csv"
        source.write_text("id,City\n1,Bombay\n")
        result = runner.invoke(
            app, ["match", str(source), str(reference), "--column", "city"]
        )
        assert result.exit_code == 0

    @needs_rapidfuzz
    def test_single_column_csv_needs_no_column_flag(self, tmp_path, reference):
        source = tmp_path / "one.csv"
        source.write_text("city\nBombay\nCalcutta\n")
        result = runner.invoke(app, ["match", str(source), str(reference)])
        assert result.exit_code == 0

    @needs_rapidfuzz
    def test_conventionally_named_column_is_found_automatically(
        self, tmp_path, reference
    ):
        source = tmp_path / "auto.csv"
        source.write_text("id,name,region\n1,Bombay,W\n")
        result = runner.invoke(app, ["match", str(source), str(reference)])
        assert result.exit_code == 0
        assert "name" in result.output

    @needs_rapidfuzz
    def test_below_threshold_reports_no_match_rather_than_a_self_match(
        self, tmp_path, reference
    ):
        source = tmp_path / "unmatchable.txt"
        source.write_text("zzzzzzzzzz\n")
        result = runner.invoke(
            app, ["match", str(source), str(reference), "--threshold", "0.95"]
        )
        assert result.exit_code == 0
        assert "no match" in result.output

    @needs_rapidfuzz
    def test_threshold_is_honoured_on_the_default_backend(self, tmp_path, reference):
        source = tmp_path / "far.txt"
        source.write_text("zzzzzzzzzz\n")
        loose = runner.invoke(app, ["match", str(source), str(reference)])
        strict = runner.invoke(
            app, ["match", str(source), str(reference), "--threshold", "0.95"]
        )
        assert loose.exit_code == strict.exit_code == 0
        assert "no match" not in loose.output
        assert "no match" in strict.output


class TestErrorsAreActionable:
    """Each failure must name its own fix. These assert on the fix, not the wording."""

    def test_missing_file_names_the_path(self, tmp_path, reference):
        result = runner.invoke(
            app, ["match", str(tmp_path / "absent.csv"), str(reference)]
        )
        assert result.exit_code == 1
        assert "absent.csv" in result.output

    def test_ambiguous_column_lists_the_columns_and_the_flag(self, tmp_path, reference):
        source = tmp_path / "wide.csv"
        source.write_text("a,b,c\n1,2,3\n")
        result = runner.invoke(app, ["match", str(source), str(reference)])
        assert result.exit_code == 1
        assert "--column" in result.output
        for column in ("a", "b", "c"):
            assert column in result.output

    def test_unknown_column_lists_the_real_ones(self, tmp_path, reference):
        source = tmp_path / "data.csv"
        source.write_text("id,city\n1,Bombay\n")
        result = runner.invoke(
            app, ["match", str(source), str(reference), "--column", "town"]
        )
        assert result.exit_code == 1
        assert "city" in result.output

    def test_suggested_column_skips_id_columns(self, tmp_path, reference):
        source = tmp_path / "wide.csv"
        source.write_text("id,hospital,ward\n1,x,y\n")
        result = runner.invoke(app, ["match", str(source), str(reference)])
        assert "--column 'hospital'" in result.output.replace("\n", "")

    def test_missing_arguments_suggest_the_full_command(self):
        result = runner.invoke(app, ["match"])
        assert result.exit_code == 1
        assert "alethia match" in result.output

    def test_unsupported_file_type_lists_supported_ones(self, tmp_path, reference):
        source = tmp_path / "data.pdf"
        source.write_text("nonsense")
        result = runner.invoke(app, ["match", str(source), str(reference)])
        assert result.exit_code == 1
        assert ".csv" in result.output

    def test_empty_file_is_rejected(self, tmp_path, reference):
        source = tmp_path / "empty.txt"
        source.write_text("\n\n")
        result = runner.invoke(app, ["match", str(source), str(reference)])
        assert result.exit_code == 1

    def test_directory_instead_of_file_is_rejected(self, tmp_path, reference):
        result = runner.invoke(app, ["match", str(tmp_path), str(reference)])
        assert result.exit_code == 1

    def test_install_extras_survive_rich_markup(self, tmp_path, reference):
        source = tmp_path / "wide.csv"
        source.write_text("a,b\n1,2\n")
        result = runner.invoke(
            app, ["assess", str(source), str(reference), "-m", "only-one"]
        )
        assert result.exit_code == 1
        assert "pip install 'alethia'" not in result.output

    def test_assess_rejects_a_single_model_with_an_explanation(self, messy, reference):
        result = runner.invoke(
            app, ["assess", str(messy), str(reference), "--models", "only-one"]
        )
        assert result.exit_code == 1
        assert "--models" in result.output

    def test_same_file_twice_is_caught(self, messy):
        result = runner.invoke(app, ["match", str(messy), str(messy)])
        assert result.exit_code == 1
        assert "same file" in result.output

    def test_cluster_without_a_file_explains_what_is_needed(self):
        result = runner.invoke(app, ["cluster"])
        assert result.exit_code == 1
        assert "alethia cluster" in result.output


class TestFallbackHonesty:
    """The CLI must report the backend that ran, not the one that was asked for."""

    @staticmethod
    def _frame(requested, actual):
        frame = pd.DataFrame({"given_entity": ["a"]})
        frame.attrs["backend"] = requested
        frame.attrs["effective_backend"] = actual
        frame.attrs["fallback_from"] = requested if actual != requested else None
        return frame

    def test_no_fallback_when_the_requested_backend_ran(self):
        from alethia.cli import _fallback_backend

        assert _fallback_backend(self._frame("rapidfuzz", "rapidfuzz")) is None
        assert (
            _fallback_backend(
                self._frame("sentence-transformers", "sentence-transformers")
            )
            is None
        )

    def test_fallback_names_the_backend_that_took_over(self):
        from alethia.cli import _fallback_backend

        frame = self._frame("sentence-transformers", "rapidfuzz")
        assert _fallback_backend(frame) == "rapidfuzz"

    def test_fallback_to_an_api_backend_is_not_called_rapidfuzz(self):
        from alethia.cli import _fallback_backend

        assert _fallback_backend(self._frame("fastembed", "openai")) == "openai"
        assert _fallback_backend(self._frame("fastembed", "gemini")) == "gemini"

    def test_missing_metadata_is_treated_as_no_fallback(self):
        from alethia.cli import _fallback_backend

        assert _fallback_backend(pd.DataFrame({"given_entity": ["a"]})) is None

    def test_library_records_the_backend_that_actually_ran(self):
        from alethia import alethia as run_match

        result = run_match(["Bombay"], ["Mumbai", "Delhi"], model="rapidfuzz")
        assert result.attrs["effective_backend"] == "rapidfuzz"
        assert result.attrs["fallback_from"] is None

    def test_the_default_run_is_not_reported_as_a_fallback(self):
        from alethia import alethia as run_match
        from alethia.cli import _fallback_backend

        result = run_match(["Bombay"], ["Mumbai", "Delhi"])
        assert _fallback_backend(result) is None


class TestBackendGating:
    """Each keyword model is gated on its own dependency, not on a shared one."""

    @staticmethod
    def _deps(**overrides):
        base = {
            "SENTENCE_TRANSFORMERS_AVAILABLE": False,
            "FASTEMBED_AVAILABLE": False,
            "RAPIDFUZZ_AVAILABLE": False,
            "OPENAI_AVAILABLE": False,
            "GEMINI_AVAILABLE": False,
        }
        base.update(overrides)
        return base

    def test_openai_is_not_blocked_by_a_missing_local_backend(
        self, monkeypatch, messy, reference
    ):
        import alethia.cli as cli_mod

        monkeypatch.setattr(cli_mod, "_has_embedding_backend", lambda deps: False)
        seen = {}

        def fake_check(*a, **k):
            return self._deps(OPENAI_AVAILABLE=True)

        def fake_match(*args, **kwargs):
            seen["ran"] = True
            raise RuntimeError("stop before any API call")

        import importlib

        import alethia.alethia as _  # noqa: F401  (ensure module is importable)

        monkeypatch.setattr(
            importlib.import_module("alethia.alethia"),
            "check_optional_dependencies",
            fake_check,
        )
        monkeypatch.setattr(
            importlib.import_module("alethia.alethia"), "alethia", fake_match
        )

        result = runner.invoke(
            app, ["match", str(messy), str(reference), "-m", "openai"]
        )
        assert seen.get("ran"), result.output

    def test_gemini_without_its_package_names_that_package(
        self, monkeypatch, messy, reference
    ):
        import importlib

        monkeypatch.setattr(
            importlib.import_module("alethia.alethia"),
            "check_optional_dependencies",
            lambda *a, **k: self._deps(SENTENCE_TRANSFORMERS_AVAILABLE=True),
        )
        result = runner.invoke(
            app, ["match", str(messy), str(reference), "-m", "gemini"]
        )
        assert result.exit_code == 1
        assert "google-generativeai" in result.output


class TestMarkupSafety:
    """Rich eats [text] as a markup tag, so status markers must avoid brackets."""

    def test_status_markers_survive_rendering(self):
        result = runner.invoke(app, ["check"])
        assert "ready" in result.output
        assert "[ok]" not in result.output
        assert "+ ready" in result.output or "- missing" in result.output

    def test_saved_confirmation_keeps_its_marker(self, messy, reference, tmp_path):
        out = tmp_path / "saved.csv"
        result = runner.invoke(
            app, ["match", str(messy), str(reference), "-o", str(out)]
        )
        assert "Saved" in result.output
        assert "[ok]" not in result.output

    def test_source_uses_no_bare_bracket_markers_in_rich_strings(self):
        import pathlib

        import alethia.cli as cli_mod

        source = pathlib.Path(cli_mod.__file__).read_text()
        for marker in ("[ok]", "[x]", "[!]"):
            assert marker not in source, f"{marker} would be swallowed by Rich markup"
