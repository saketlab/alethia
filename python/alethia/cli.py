"""Command-line interface for alethia.

Four rules hold across the commands:

1. Never dead-end: a bare command opens a guided prompt or a worked example.
2. Every error prints the command, column name, or ``pip install`` line that fixes it.
3. Infer file type and columns; turn ambiguity into a menu.
4. Keep imports local. Nothing heavier than typer and rich loads at module scope, which
   keeps ``alethia --help`` at ~0.1s instead of ~4s.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="alethia",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

TEXT_SUFFIXES = {".txt", ".text", ""}
TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls"}
DATA_SUFFIXES = TABLE_SUFFIXES | {".txt"}

LIKELY_COLUMNS = (
    "entity",
    "name",
    "value",
    "text",
    "term",
    "label",
    "entry",
    "string",
    "item",
)

EXTRA_RECOMMENDED = escape("alethia[recommended]")
EXTRA_FULL = escape("alethia[full]")
EXTRA_ONNX = escape("alethia[onnx]")  # CPU-only: FastEmbed on ONNX Runtime, no torch

NON_EMBEDDING_BACKENDS = {"rapidfuzz", "openai", "gemini"}

DEFAULT_MATCH_MODEL = "rapidfuzz"  # needs no download, so a first run works offline
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~90 MB on first use


def _interactive() -> bool:
    """True when we can safely prompt (a real terminal on both ends)."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _fail(message: str, *hints: str) -> typer.Exit:
    """Print an error with actionable next steps and exit non-zero."""
    body = f"[bold red]{message}[/bold red]"
    if hints:
        body += "\n\n" + "\n".join(f"[dim]->[/dim] {h}" for h in hints)
    err_console.print(Panel(body, title="[red]Problem[/red]", border_style="red"))
    return typer.Exit(1)


def _note(message: str) -> None:
    console.print(f"[dim]*[/dim] {message}")


def _version() -> str:
    """Version of the installed distribution, not of the imported source tree."""
    try:
        from importlib.metadata import version

        return version("alethia")
    except Exception:  # pragma: no cover - only if run from a non-installed tree
        return "unknown"


def _nearby_data_files() -> list[Path]:
    """Data files in the working directory, to suggest when a path is needed."""
    found = [
        p
        for p in sorted(Path.cwd().iterdir())
        if p.is_file() and p.suffix.lower() in DATA_SUFFIXES
    ]
    return found[:8]


def _choose_from(prompt: str, options: Sequence[str]) -> str:
    """Show a numbered menu and return the chosen option."""
    for i, opt in enumerate(options, 1):
        console.print(f"  [cyan]{i}[/cyan]. {opt}")
    while True:
        raw = Prompt.ask(prompt, default="1").strip()
        if raw in options:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        console.print(f"[yellow]Enter a number from 1 to {len(options)}.[/yellow]")


def _ask_for_path(label: str, exclude: Path | None = None) -> Path:
    """Prompt for a file path, showing what is available in the current folder."""
    nearby = [p for p in _nearby_data_files() if p != exclude]
    if nearby:
        console.print(f"\n[bold]{label}[/bold] - data files in this folder:")
        names = [p.name for p in nearby]
        names.append("(type a different path)")
        picked = _choose_from("Which file?", names)
        if picked != "(type a different path)":
            return Path(picked)
    return Path(Prompt.ask(f"[bold]{label}[/bold] - path to the file").strip())


def _resolve_column(
    columns: Sequence[str], requested: str | None, path: Path, flag: str
) -> str:
    """Pick which column holds the entity text."""
    if requested:
        for col in columns:  # exact match first, then case-insensitive
            if col == requested:
                return col
        for col in columns:
            if col.lower() == requested.lower():
                return col
        raise _fail(
            f"'{path.name}' has no column named '{requested}'.",
            f"Columns in this file: {', '.join(columns)}",
            f"Try: [cyan]{flag} '{_suggest_column(columns)}'[/cyan]",
        )

    if len(columns) == 1:
        return columns[0]

    lowered = {c.lower(): c for c in columns}
    for candidate in LIKELY_COLUMNS:
        if candidate in lowered:
            chosen = lowered[candidate]
            _note(f"Using column [cyan]{chosen}[/cyan] from {path.name}.")
            return chosen

    if _interactive():
        console.print(f"\n[bold]{path.name}[/bold] has several columns:")
        return _choose_from("Which column holds the entities?", list(columns))

    raise _fail(
        f"'{path.name}' has {len(columns)} columns, so it is unclear which to read.",
        f"Columns: {', '.join(columns)}",
        f"Try: [cyan]{flag} '{_suggest_column(columns)}'[/cyan]",
    )


def _suggest_column(columns: Sequence[str]) -> str:
    """Pick the column most worth suggesting in an error hint."""
    skip = {"id", "index", "idx", "key", "row", "no", "num", "count", ""}
    for col in columns:
        stripped = col.strip().lower()
        if stripped in skip or stripped.startswith("unnamed"):
            continue
        return col
    return columns[0]


def _read_entities(path: Path, column: str | None, label: str, flag: str) -> list[str]:
    """Read entity strings from a text or tabular file."""
    if not path.exists():
        hints = [f"Check the path and spelling of [cyan]{path}[/cyan]."]
        nearby = _nearby_data_files()
        if nearby:
            hints.append(
                "Data files in this folder: " + ", ".join(p.name for p in nearby)
            )
        raise _fail(f"{label} file not found: {path}", *hints)

    if path.is_dir():
        raise _fail(f"{label} path is a folder, not a file: {path}")

    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        values = [
            line.strip() for line in path.read_text(errors="replace").splitlines()
        ]
    elif suffix in TABLE_SUFFIXES:
        import pandas as pd

        try:
            if suffix in {".xlsx", ".xls"}:
                frame = pd.read_excel(path)
            else:
                frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        except ImportError as exc:  # missing openpyxl for Excel
            raise _fail(
                f"Cannot read {path.name}: {exc}",
                "Try: [cyan]pip install openpyxl[/cyan]",
                "Or save the sheet as .csv and use that instead.",
            ) from None
        except Exception as exc:
            raise _fail(
                f"Could not read {path.name}: {exc}",
                "Make sure the file is a valid CSV/TSV/Excel file.",
            ) from None
        if frame.empty or not len(frame.columns):
            raise _fail(f"{path.name} has no data in it.")
        chosen = _resolve_column([str(c) for c in frame.columns], column, path, flag)
        values = [str(v).strip() for v in frame[chosen].tolist()]
    else:
        raise _fail(
            f"Don't know how to read '{path.suffix}' files.",
            "Supported: .csv, .tsv, .xlsx, .txt (one entity per line).",
        )

    # the library's null vocabulary, so both drop the same cells
    from .alethia import _filter_nan_entries

    entities = _filter_nan_entries(values)
    if not entities:
        raise _fail(f"No usable entries found in {path.name}.")
    return entities


def _ask_for_file_pair(
    first: Path | None,
    second: Path | None,
    *,
    title: str,
    blurb: str,
    hints: Sequence[str],
) -> tuple:
    """Fill in whichever of the two input files was not given on the command line."""
    if first is not None and second is not None:
        return first, second
    if not _interactive():
        raise _fail(*hints)
    console.print(Panel(blurb, title=f"[cyan]{title}[/cyan]", border_style="cyan"))
    if first is None:
        first = _ask_for_path("Messy entries")
    if second is None:
        second = _ask_for_path("Correct entries (the reference list)", exclude=first)
    return first, second


def _require_embedding_backend(message: str, *hints: str) -> dict:
    """Return the dependency map, failing with ``message`` if no embedding backend is installed."""
    from .alethia import check_optional_dependencies

    deps = check_optional_dependencies()
    if not _has_embedding_backend(deps):
        raise _fail(message, *hints)
    return deps


def _read_pair(
    messy_file: Path,
    column: str | None,
    reference_file: Path,
    reference_column: str | None,
):
    """Read the (messy, reference) file pair that ``match`` and ``assess`` both take."""
    return (
        _read_entities(messy_file, column, "Messy entries", "--column"),
        _read_entities(
            reference_file, reference_column, "Reference", "--reference-column"
        ),
    )


def _write_output(frame, path: Path) -> None:
    """Save results, choosing the format from the file extension."""
    suffix = path.suffix.lower()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".tsv":
            frame.to_csv(path, sep="\t", index=False)
        elif suffix == ".json":
            frame.to_json(path, orient="records", indent=2)
        elif suffix in {".xlsx", ".xls"}:
            frame.to_excel(path, index=False)
        else:
            frame.to_csv(path, index=False)
    except ImportError as exc:
        raise _fail(
            f"Cannot write {path.suffix} files: {exc}",
            "Try: [cyan]pip install openpyxl[/cyan], or save as .csv instead.",
        ) from None
    except OSError as exc:
        raise _fail(f"Could not write {path}: {exc}") from None
    console.print(f"\n[green]OK[/green] Saved {len(frame)} rows to [bold]{path}[/bold]")


def _has_embedding_backend(deps: dict) -> bool:
    """True when some backend can turn text into embeddings."""
    return deps["SENTENCE_TRANSFORMERS_AVAILABLE"] or deps["FASTEMBED_AVAILABLE"]


def _score_style(score: float) -> str:
    """Colour a similarity score so a scan of the table shows what needs review."""
    if score >= 0.9:
        return "green"
    if score >= 0.75:
        return "yellow"
    return "red"


def _actual_backend(frame) -> str:
    """The backend that produced these results, as recorded by ``alethia()``."""
    return str(frame.attrs.get("effective_backend", ""))


def _fallback_backend(frame) -> str | None:
    """The backend that took over, or ``None`` when the selected one ran."""
    if not frame.attrs.get("fallback_from"):
        return None
    return _actual_backend(frame) or None


def _render_matches(frame, limit: int, method_label: str | None = None) -> None:
    """Show matches as a table, changed rows first, with a plain-language summary."""
    import pandas as pd

    total = len(frame)
    predictions = frame["alethia_prediction"]
    missing = predictions.isna()
    unmatched = int(missing.sum())
    changed_mask = ~missing & (frame["given_entity"] != predictions)
    changed = int(changed_mask.sum())
    unchanged = total - changed - unmatched

    # truncated before concatenating, so a large frame is never copied whole
    shown = frame[changed_mask].head(limit)
    if changed < limit:
        shown = pd.concat([shown, frame[~changed_mask].head(limit - changed)])

    table = Table(
        title=f"Matches ({min(limit, total)} of {total} rows)",
        header_style="bold",
        border_style="dim",
    )
    table.add_column("Your entry", overflow="fold")
    table.add_column("Matched to", overflow="fold")
    table.add_column("Score", justify="right")
    table.add_column("How", style="dim")

    for index, row in shown.iterrows():
        prediction = row["alethia_prediction"]
        score = row.get("alethia_score", float("nan"))
        if pd.isna(score):
            rendered_score = "[dim]-[/dim]"
        else:
            style = _score_style(score)
            rendered_score = f"[{style}]{score:.3f}[/{style}]"
        table.add_row(
            str(row["given_entity"]),
            "[dim italic]no match[/dim italic]" if missing[index] else str(prediction),
            rendered_score,
            method_label or str(row.get("alethia_method", "")),
        )

    console.print()
    console.print(table)

    summary = [
        f"[bold]{total}[/bold] entries processed",
        f"[green]{unchanged}[/green] already correct",
        f"[yellow]{changed}[/yellow] corrected",
    ]
    if unmatched:
        summary.append(f"[red]{unmatched}[/red] with no confident match")
    console.print("  " + "  [dim]|[/dim]  ".join(summary))

    if changed and total > limit:
        console.print(
            "[dim]Showing corrected entries first. "
            "Use --limit to see more, or --output to save every row.[/dim]"
        )
    if unmatched:
        console.print(
            "[dim]No confident match means nothing scored above the threshold. "
            "Lower it with --threshold 0.5 to be more permissive.[/dim]"
        )


def _welcome() -> None:
    """Landing screen: what this tool does and three commands to copy."""
    console.print(
        Panel(
            "[bold]alethia[/bold] cleans up messy lists of names.\n\n"
            "You have a column of entries with typos, abbreviations, and\n"
            "inconsistent spellings. alethia matches them to the correct versions.",
            title=f"[bold cyan]alethia[/bold cyan] [dim]v{_version()}[/dim]",
            border_style="cyan",
        )
    )
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("If you want to...")
    table.add_column("Run")
    table.add_row(
        "Fix messy names against a correct list",
        "[cyan]alethia match[/cyan] messy.csv correct.csv",
    )
    table.add_row(
        "Group duplicates with no correct list",
        "[cyan]alethia cluster[/cyan] messy.csv",
    )
    table.add_row(
        "Find which AI model works best on your data",
        "[cyan]alethia assess[/cyan] messy.csv correct.csv",
    )
    table.add_row("See what's installed and working", "[cyan]alethia check[/cyan]")
    console.print(table)
    console.print(
        "\n[dim]New here? Run [cyan]alethia match[/cyan] with no files and it will "
        "walk you through it.\nAdd [cyan]--help[/cyan] to any command for its "
        "options.[/dim]\n"
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Show the version and exit."
    ),
) -> None:
    """[bold]alethia[/bold] - clean up messy lists of names using AI embeddings.

    Match misspelled entries against a correct list, group duplicates, or work out
    which embedding model performs best on your own data.
    """
    if version:
        console.print(f"alethia {_version()}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _welcome()


@app.command()
def match(
    messy_file: Path | None = typer.Argument(
        None,
        help="Messy entries to fix (.csv, .tsv, .xlsx, .txt).",
    ),
    reference_file: Path | None = typer.Argument(
        None,
        help="Correct entries to match against.",
    ),
    column: str | None = typer.Option(
        None, "--column", "-c", help="Column holding the messy entries."
    ),
    reference_column: str | None = typer.Option(
        None, "--reference-column", "-r", help="Column in the reference file."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Save all results here (.csv/.tsv/.json/.xlsx).",
    ),
    model: str = typer.Option(
        DEFAULT_MATCH_MODEL,
        "--model",
        "-m",
        help="'rapidfuzz' to match on spelling, or a model name to match on meaning.",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        "-t",
        min=0.0,
        max=1.0,
        help="Minimum score to accept. Unset, models use 0.7 and rapidfuzz takes its "
        "best guess.",
    ),
    limit: int = typer.Option(
        20, "--limit", "-n", min=1, help="Rows to show on screen."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed progress."
    ),
) -> None:
    """Fix messy entries by matching them to a list of correct ones.

    [bold]Examples[/bold]

      [dim]# Fix city names against a reference list[/dim]
      [cyan]alethia match cities_messy.csv cities_correct.csv[/cyan]

      [dim]# Pick the column when the file has several[/dim]
      [cyan]alethia match data.csv reference.csv --column city_raw[/cyan]

      [dim]# Match by meaning rather than spelling, and save the result[/dim]
      [cyan]alethia match data.csv ref.csv -m all-MiniLM-L6-v2 -o fixed.csv[/cyan]
    """
    messy_file, reference_file = _ask_for_file_pair(
        messy_file,
        reference_file,
        title="alethia match",
        blurb="Let's fix a messy list of names.\n"
        "You need [bold]two files[/bold]: the messy entries, and the correct "
        "ones to match them against.",
        hints=(
            "Two files are needed: the messy entries, and the correct ones.",
            "Try: [cyan]alethia match messy.csv correct.csv[/cyan]",
            "Run [cyan]alethia match --help[/cyan] for the full list of options.",
        ),
    )

    if messy_file == reference_file:
        raise _fail(
            "The messy file and the reference file are the same file.",
            "Every entry would match itself, so nothing would be corrected.",
            "Pass the correct entries as the second file: "
            "[cyan]alethia match messy.csv correct.csv[/cyan]",
        )

    messy, reference = _read_pair(messy_file, column, reference_file, reference_column)
    _note(f"Read {len(messy)} messy entries and {len(reference)} reference entries.")

    from .alethia import alethia as run_match
    from .alethia import check_optional_dependencies

    deps = check_optional_dependencies()
    required = {
        "rapidfuzz": ("RAPIDFUZZ_AVAILABLE", "rapidfuzz"),
        "openai": ("OPENAI_AVAILABLE", "openai"),
        "gemini": ("GEMINI_AVAILABLE", "google-generativeai"),
    }.get(model)
    if required:
        flag, package = required
        if not deps[flag]:
            raise _fail(
                f"{package} is not installed, so [cyan]--model {model}[/cyan] "
                "cannot run.",
                f"Try: [cyan]pip install {package}[/cyan]",
                "Or see what is available: [cyan]alethia check[/cyan]",
            )
    elif not _has_embedding_backend(deps):
        raise _fail(
            f"Matching with '{model}' needs an embedding backend.",
            f"Try: [cyan]pip install '{EXTRA_RECOMMENDED}'[/cyan]",
            f"On a CPU-only machine prefer: [cyan]pip install '{EXTRA_ONNX}'[/cyan]",
            "Or use fast spelling-based matching instead: [cyan]--model rapidfuzz[/cyan]",
        )
    else:
        _note(
            f"Using model [cyan]{model}[/cyan] (first run downloads it, then it's cached)."
        )

    with console.status(f"[cyan]Matching {len(messy)} entries...", spinner="dots"):
        try:
            results = run_match(
                messy, reference, model=model, threshold=threshold, verbose=verbose
            )
        except ImportError as exc:
            raise _fail(
                str(exc), f"Try: [cyan]pip install '{EXTRA_RECOMMENDED}'[/cyan]"
            ) from None
        except Exception as exc:
            raise _fail(
                f"Matching failed: {exc}",
                "Check that both files hold plain text entries.",
                "Run again with [cyan]--verbose[/cyan] to see more detail.",
            ) from None

    fallback = _fallback_backend(results)
    if fallback:
        how = (
            "spelling-based matching"
            if fallback == "rapidfuzz"
            else f"the {fallback} backend"
        )
        err_console.print(
            Panel(
                f"[bold]'{model}' could not be used, so these results come from "
                f"{how} instead[/bold], not from that model.",
                title="[yellow]Model did not run[/yellow]",
                border_style="yellow",
            )
        )
        _note("Check the model name at [cyan]https://huggingface.co/models[/cyan].")
        _note("Or list installed options with [cyan]alethia models[/cyan].")

    _render_matches(results, limit, method_label=fallback)
    if output:
        _write_output(results, output)
    elif _interactive() and Confirm.ask("\nSave all results to a file?", default=False):
        _write_output(
            results, Path(Prompt.ask("Save as", default="alethia_results.csv"))
        )


@app.command()
def cluster(
    entities_file: Path | None = typer.Argument(
        None, help="Entries to group (.csv, .tsv, .xlsx, .txt)."
    ),
    column: str | None = typer.Option(
        None, "--column", "-c", help="Column holding the entries."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save the groups to this file."
    ),
    model: str = typer.Option(
        DEFAULT_EMBEDDING_MODEL, "--model", "-m", help="Embedding model to group by."
    ),
    floor: float = typer.Option(
        0.80,
        "--similarity",
        "-s",
        min=0.0,
        max=1.0,
        help="Similarity needed to group. Higher groups less.",
    ),
    limit: int = typer.Option(
        20, "--limit", "-n", min=1, help="Groups to show on screen."
    ),
) -> None:
    """Group duplicate entries together when you have no correct list.

    Entries are grouped only when each is the other's nearest match, so groups stay
    tight rather than chaining together through one loosely-related entry.

    [bold]Examples[/bold]

      [dim]# Group similar company names[/dim]
      [cyan]alethia cluster companies.csv[/cyan]

      [dim]# Require a closer match, and save the groups[/dim]
      [cyan]alethia cluster companies.csv --similarity 0.9 -o groups.csv[/cyan]
    """
    if entities_file is None:
        if not _interactive():
            raise _fail(
                "A file of entries to group is needed.",
                "Try: [cyan]alethia cluster entries.csv[/cyan]",
            )
        console.print(
            Panel(
                "Let's group duplicate entries together.\n"
                "You need [bold]one file[/bold] of messy entries - no correct list "
                "required.",
                title="[cyan]alethia cluster[/cyan]",
                border_style="cyan",
            )
        )
        entities_file = _ask_for_path("Entries to group")

    entities = _read_entities(entities_file, column, "Entries", "--column")
    _note(f"Read {len(entities)} entries.")

    _require_embedding_backend(
        "Grouping needs an embedding model, which is not installed.",
        f"Try: [cyan]pip install '{EXTRA_RECOMMENDED}'[/cyan]",
    )
    _note(
        f"Using model [cyan]{model}[/cyan] (first run downloads it, then it's cached)."
    )

    from .cluster import cluster_entities

    with console.status(f"[cyan]Grouping {len(entities)} entries...", spinner="dots"):
        try:
            result = cluster_entities(entities, model, floor=floor)
        except Exception as exc:
            raise _fail(
                f"Grouping failed: {exc}",
                "Check the model name, or try [cyan]--model all-MiniLM-L6-v2[/cyan].",
            ) from None

    all_clusters = result.clusters()  # rebuilt on every call, so hold onto it
    groups = {k: v for k, v in all_clusters.items() if len(v) > 1}
    table = Table(
        title=f"Groups of duplicates ({len(groups)} found)",
        header_style="bold",
        border_style="dim",
    )
    table.add_column("Suggested name", overflow="fold")
    table.add_column("Grouped entries", overflow="fold")
    table.add_column("Size", justify="right")

    for cid, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:limit]:
        table.add_row(result.canonical[cid], ", ".join(members), str(len(members)))

    console.print()
    if groups:
        console.print(table)
    else:
        console.print("[yellow]No duplicate groups found.[/yellow]")
        console.print(
            "[dim]Every entry looks distinct at this similarity. Try a lower "
            "value, e.g. [cyan]--similarity 0.7[/cyan].[/dim]"
        )
    singles = len(all_clusters) - len(groups)
    console.print(
        f"  [bold]{len(entities)}[/bold] entries  [dim]|[/dim]  "
        f"[yellow]{len(groups)}[/yellow] groups of duplicates  [dim]|[/dim]  "
        f"[green]{singles}[/green] with no duplicate"
    )

    if output:
        import pandas as pd

        _write_output(pd.DataFrame(result.to_records()), output)


@app.command()
def assess(
    messy_file: Path | None = typer.Argument(
        None, help="File with your messy entries."
    ),
    reference_file: Path | None = typer.Argument(
        None, help="File with your correct entries."
    ),
    models: str | None = typer.Option(
        None,
        "--models",
        "-m",
        help="Comma-separated models to compare. Defaults to two small ones.",
    ),
    column: str | None = typer.Option(
        None, "--column", "-c", help="Column in the messy file."
    ),
    reference_column: str | None = typer.Option(
        None, "--reference-column", "-r", help="Column in the reference file."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save the scoring table (.csv or .html)."
    ),
) -> None:
    """Work out which embedding model performs best on [bold]your[/bold] data.

    This needs no correct answers. Models are scored on the shape of the embeddings
    they produce for your entries, so a leaderboard-topping model that happens to
    suit your data poorly will still rank low here.

    [bold]Examples[/bold]

      [dim]# Compare the two default models[/dim]
      [cyan]alethia assess messy.csv correct.csv[/cyan]

      [dim]# Compare specific models and save a report[/dim]
      [cyan]alethia assess messy.csv correct.csv -m all-MiniLM-L6-v2,all-mpnet-base-v2 -o report.html[/cyan]
    """
    messy_file, reference_file = _ask_for_file_pair(
        messy_file,
        reference_file,
        title="alethia assess",
        blurb="Let's find the best model for your data.\n"
        "You need [bold]two files[/bold]: your messy entries and your "
        "correct ones. No correct answers required - models are scored on "
        "your data directly.",
        hints=(
            "Two files are needed: your messy entries, and your correct ones.",
            "Try: [cyan]alethia assess messy.csv correct.csv[/cyan]",
        ),
    )

    queries, references = _read_pair(
        messy_file, column, reference_file, reference_column
    )

    names = (
        [m.strip() for m in models.split(",") if m.strip()]
        if models
        else [DEFAULT_EMBEDDING_MODEL, "all-mpnet-base-v2"]
    )
    if len(names) < 2:
        raise _fail(
            "Comparing models needs at least two of them.",
            "The score ranks models against each other, so a single model has "
            "nothing to be ranked against.",
            f"Try: [cyan]--models {names[0] if names else 'all-MiniLM-L6-v2'},"
            "all-mpnet-base-v2[/cyan]",
        )

    _require_embedding_backend(
        "Scoring models needs an embedding backend, which is not installed.",
        f"Try: [cyan]pip install '{EXTRA_RECOMMENDED}'[/cyan]",
    )

    _note(f"Comparing {len(names)} models: {', '.join(names)}")
    _note("Each model downloads on first use, then is cached.")

    from .assess import assess_models

    with console.status("[cyan]Scoring models on your data...", spinner="dots"):
        try:
            report = assess_models(queries, references, {n: n for n in names})
        except Exception as exc:
            raise _fail(
                f"Scoring failed: {exc}",
                "Check the model names are correct on huggingface.co.",
            ) from None

    frame = report.to_table()
    table = Table(
        title="Model comparison (higher score is better)",
        header_style="bold",
        border_style="dim",
    )
    table.add_column("Rank", justify="right")
    table.add_column("Model", overflow="fold")
    table.add_column("Score", justify="right")
    table.add_column("Notes", style="dim", overflow="fold")

    for rank, (_, row) in enumerate(frame.iterrows(), 1):
        score = row.get("score", float("nan"))
        table.add_row(
            str(rank),
            str(row["model"]),
            "[dim]-[/dim]" if score != score else f"{score:.3f}",
            str(row.get("error", "") or ""),
        )

    console.print()
    console.print(table)

    best = report.best
    if best is not None:
        console.print(
            f"\n[green]OK[/green] Best for your data: [bold]{best.name}[/bold]"
        )
        console.print(
            f"[dim]Use it with: [cyan]alethia match <messy> <reference> "
            f"-m {best.name}[/cyan][/dim]"
        )

    if output:
        if output.suffix.lower() in {".html", ".htm"}:
            report.to_html(path=str(output))
            console.print(f"\n[green]OK[/green] Saved report to [bold]{output}[/bold]")
        else:
            _write_output(frame, output)


@app.command()
def check() -> None:
    """Show what's installed and what each part lets you do.

    Run this first if a command fails, or if you are not sure the install worked.
    """
    from .alethia import check_optional_dependencies

    deps = check_optional_dependencies()

    features = [
        (
            "Spelling-based matching",
            deps["RAPIDFUZZ_AVAILABLE"],
            "alethia match (the default, fast, no download)",
        ),
        (
            "Meaning-based matching (ONNX, CPU)",
            deps["FASTEMBED_AVAILABLE"],
            "fast local embeddings, no torch needed",
        ),
        (
            "Meaning-based matching (PyTorch)",
            deps["SENTENCE_TRANSFORMERS_AVAILABLE"],
            "any HuggingFace model, larger download",
        ),
        ("OpenAI embeddings", deps["OPENAI_AVAILABLE"], "alethia match -m openai"),
        (
            "Google embeddings",
            deps["GEMINI_AVAILABLE"],
            "alethia match -m gemini",
        ),
    ]

    table = Table(
        title=f"alethia {_version()}", header_style="bold", border_style="dim"
    )
    table.add_column("Feature")
    table.add_column("Status", justify="center")
    table.add_column("Lets you", style="dim", overflow="fold")

    for name, ok, purpose in features:
        table.add_row(
            name,
            "[green]+ ready[/green]" if ok else "[yellow]- missing[/yellow]",
            purpose,
        )

    console.print()
    console.print(table)

    core_ok = deps["RAPIDFUZZ_AVAILABLE"] or deps["SENTENCE_TRANSFORMERS_AVAILABLE"]
    if not core_ok:
        console.print(
            Panel(
                "No matching backend is installed, so [cyan]alethia match[/cyan] "
                "cannot run yet.\n\n"
                f"Fix it with:  [cyan]pip install '{EXTRA_RECOMMENDED}'[/cyan]",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    if not all(ok for _, ok, _ in features):
        console.print(
            f"[dim]Everything essential works. To add the rest: "
            f"[cyan]pip install '{EXTRA_FULL}'[/cyan][/dim]"
        )
    else:
        console.print("[green]OK[/green] Everything is installed.")
    console.print(
        "\n[dim]Next: [cyan]alethia match messy.csv correct.csv[/cyan], or "
        "[cyan]alethia[/cyan] to see all commands.[/dim]\n"
    )


@app.command()
def models(
    limit: int = typer.Option(15, "--limit", "-n", min=1, help="How many to list."),
) -> None:
    """List embedding models you can pass to [cyan]--model[/cyan].

    Smaller models are faster and download quicker; larger ones are usually more
    accurate. If you are unsure, start with the first one, then run
    [cyan]alethia assess[/cyan] to see which actually works best on your data.
    """
    from .alethia import get_available_models

    table = Table(title="Embedding models", header_style="bold", border_style="dim")
    table.add_column("Model name (pass to --model)", overflow="fold")
    table.add_column("Backend", style="dim")

    try:
        available = get_available_models(include_details=False, verbose=False)
    except Exception:
        available = {}

    rows = [
        (str(name), backend)
        for backend, entries in available.items()
        if backend not in NON_EMBEDDING_BACKENDS
        for name in (entries if isinstance(entries, list) else [])
    ][:limit]
    for name, backend in rows:
        table.add_row(name, backend)

    console.print()
    if rows:
        console.print(table)
    else:
        console.print(
            Panel(
                "No embedding backend is installed, so there are no models to list.\n\n"
                f"Fix it with:  [cyan]pip install '{EXTRA_RECOMMENDED}'[/cyan]\n\n"
                "Good starting models once installed:\n"
                "  [cyan]all-MiniLM-L6-v2[/cyan]     small and fast\n"
                "  [cyan]all-mpnet-base-v2[/cyan]    larger and more accurate",
                border_style="yellow",
            )
        )
    console.print(
        "\n[dim]Not sure which to pick? "
        "[cyan]alethia assess messy.csv correct.csv[/cyan] scores them on your "
        "own data.[/dim]\n"
    )


if __name__ == "__main__":
    app()
