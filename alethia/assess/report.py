"""Self-contained HTML reporting for model assessments (figures embedded as base64)."""

import base64
import html
import io
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .assessor import AssessmentReport


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _render_figure(draw) -> Optional[str]:
    """Run ``draw(plt)`` to build a figure and return it as a base64 PNG data URI.

    ``draw`` returns the matplotlib figure, or ``None`` to skip (e.g. no data). Returns
    ``None`` if matplotlib is unavailable or ``draw`` opts out. Handles encoding and cleanup.
    """
    plt = _matplotlib()
    if plt is None:
        return None
    fig = draw(plt)
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _ranking_figure(report: "AssessmentReport") -> Optional[str]:
    def draw(plt):
        valid = [a for a in report.assessments if not a.error and np.isfinite(a.score)]
        if not valid:
            return None
        valid = sorted(valid, key=lambda a: a.score)
        names = [a.name for a in valid]
        scores = [a.score for a in valid]
        fig, ax = plt.subplots(figsize=(7, max(1.6, 0.5 * len(names) + 0.6)))
        colors = ["#2563eb"] * len(names)
        colors[-1] = "#16a34a"  # best
        ax.barh(names, scores, color=colors)
        ax.axvline(0, color="#9ca3af", lw=0.8)
        ax.set_xlabel("Composite quality score (z-blend, higher = better)")
        ax.set_title("Model ranking for your data")
        ax.spines[["top", "right"]].set_visible(False)
        return fig

    return _render_figure(draw)


def _alignment_uniformity_figure(report: "AssessmentReport") -> Optional[str]:
    def draw(plt):
        pts: List[Tuple[str, float, float]] = []
        for a in report.assessments:
            if a.error:
                continue
            al = a.metrics.get("alignment_loss")
            un = a.metrics.get("uniformity_loss")
            if (
                al is not None
                and un is not None
                and np.isfinite(al)
                and np.isfinite(un)
            ):
                pts.append((a.name, al, un))
        if not pts:
            return None
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for name, al, un in pts:
            ax.scatter(al, un, s=70, color="#2563eb")
            ax.annotate(
                name, (al, un), textcoords="offset points", xytext=(6, 4), fontsize=8
            )
        ax.set_xlabel("Alignment loss (lower = positives stay close)")
        ax.set_ylabel("Uniformity loss (lower = more spread)")
        ax.set_title("Alignment vs uniformity (bottom-left is best)")
        ax.spines[["top", "right"]].set_visible(False)
        return fig

    return _render_figure(draw)


_STYLE = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
         Arial, sans-serif; color: #111827; max-width: 980px; margin: 2rem auto;
         padding: 0 1.25rem; line-height: 1.5; }
  h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
  .sub { color: #6b7280; margin-top: 0; }
  .verdict { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px;
             padding: 1rem 1.25rem; margin: 1.5rem 0; }
  .verdict b { color: #166534; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin: 1rem 0; }
  th, td { text-align: right; padding: 0.4rem 0.6rem; border-bottom: 1px solid #e5e7eb; }
  th:first-child, td:first-child { text-align: left; }
  thead th { background: #f9fafb; border-bottom: 2px solid #d1d5db; position: sticky; top: 0; }
  tbody tr:first-child { background: #f0fdf4; font-weight: 600; }
  .fig { margin: 1.5rem 0; text-align: center; }
  .fig img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; }
  .meta { color: #6b7280; font-size: 0.8rem; }
  details { margin: 1rem 0; }
  code { background: #f3f4f6; padding: 0.1rem 0.3rem; border-radius: 4px; }
</style>
"""

_METHOD_NOTE = """
<details>
<summary>How these metrics are computed (label-free methodology)</summary>
<ul>
<li><b>Separability</b>: nearest-other-reference cosine and confusability rate (can the
    model tell distinct canonical entries apart?).</li>
<li><b>Alignment / uniformity</b>: alignment uses synthesized dirty variants of your
    references as label-free positive pairs; uniformity measures hypersphere spread. After
    Wang &amp; Isola (2020),
    <a href="https://doi.org/10.48550/arXiv.2005.10242">doi:10.48550/arXiv.2005.10242</a>.</li>
<li><b>Retrieval margin</b>: top-1 vs top-2 cosine gap (confidence).</li>
<li><b>Hubness</b>: skew of how often references are top-k neighbours; high values indicate
    a retrieval pathology. After Radovanović et al. (2010),
    <a href="https://www.jmlr.org/papers/v11/radovanovic10a.html">JMLR 11:2487-2531</a>.</li>
<li><b>Mutual-NN rate</b>: reciprocity of query and reference nearest neighbours.</li>
</ul>
<p class="meta">No ground-truth labels are used. The composite score is a documented,
oriented z-score blend across the assessed models.</p>
</details>
"""


def _table_html(report: "AssessmentReport") -> str:
    df = report.to_table()
    # Round floats for display.
    disp = df.copy()
    for col in disp.columns:
        if disp[col].dtype.kind in "fc":
            disp[col] = disp[col].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "n/a")
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in disp.columns)
    rows = []
    for _, r in disp.iterrows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in r)
        rows.append(f"<tr>{cells}</tr>")
    return (
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def render_html(
    report: "AssessmentReport",
    path: Optional[str] = None,
    title: str = "Alethia Model Assessment",
) -> str:
    """Render the assessment as a self-contained HTML string; optionally write to ``path``."""
    best = report.best
    if best is None:
        verdict = (
            '<div class="verdict">No model could be assessed successfully. '
            "Check the error column in the table below.</div>"
        )
    elif report.is_comparative:
        verdict = (
            f'<div class="verdict">Recommended model for your data: '
            f"<b>{html.escape(best.name)}</b> "
            f"(composite score {best.score:.3f}, across {len(report.assessments)} "
            f"candidate models).</div>"
        )
    else:
        verdict = (
            f'<div class="verdict">Assessed <b>{html.escape(best.name)}</b>. '
            "The composite score ranks models against each other, so assess two or more "
            "models to get a recommendation. The metrics below still characterize this "
            "model on your data.</div>"
        )

    figs = []
    rank = _ranking_figure(report)
    if rank:
        figs.append(f'<div class="fig"><img src="{rank}" alt="ranking"/></div>')
    au = _alignment_uniformity_figure(report)
    if au:
        figs.append(
            f'<div class="fig"><img src="{au}" alt="alignment-uniformity"/></div>'
        )
    if not figs:
        figs.append(
            '<p class="meta">Install matplotlib to include charts '
            "(<code>pip install matplotlib</code>).</p>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>{_STYLE}</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="sub">{report.n_queries} queries &middot; {report.n_references} references
&middot; label-free assessment</p>
{verdict}
{"".join(figs)}
<h2>Assessment table</h2>
{_table_html(report)}
{_METHOD_NOTE}
</body></html>"""

    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
    return doc
