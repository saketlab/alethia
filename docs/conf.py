"""Sphinx configuration for Alethia documentation."""

import importlib
import os
import sys
from datetime import datetime

# Add parent directory to path for autodoc
sys.path.insert(0, os.path.abspath(".."))

import alethia

# -- Project information -----------------------------------------------------

project = "Alethia"
copyright = f"{datetime.now().year}, Saket Choudhary"
author = "Saket Choudhary"
version = alethia.__version__
release = alethia.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_nb",
]

# Templates
templates_path = ["_templates"]

# Source files. myst-nb handles both Markdown and notebooks.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Master document
master_doc = "index"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Pygments styles: neutral light background (the "sphinx" style tints code blocks green),
# monokai for the dark default theme.
pygments_style = "friendly"
pygments_dark_style = "monokai"

# -- Autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Autosummary
autosummary_generate = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"

html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#1b1b1f",
        "color-brand-content": "#1b1b1f",
        "font-stack": '"SF Pro Display", "SF Pro Text", -apple-system, '
        'BlinkMacSystemFont, "Segoe UI", sans-serif',
        "font-stack--monospace": '"SF Mono", "JetBrains Mono", Menlo, monospace',
    },
    "dark_css_variables": {
        "color-brand-primary": "#f5f5f7",
        "color-brand-content": "#f5f5f7",
        "background-color": "#050505",
        "background-color-secondary": "#121214",
    },
    "source_repository": "https://github.com/saketlab/alethia",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_title = f"{project} {version}"
html_logo = "logo.svg"
html_favicon = "logo.svg"

html_static_path = ["_static"]

# Custom CSS
html_css_files = [
    "css/custom.css",
]

# Social links in footer
html_context = {
    "display_github": True,
    "github_user": "saketlab",
    "github_repo": "alethia",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Options for copybutton --------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True

# -- Options for MyST / MyST-NB ----------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Notebook vignettes ship with pre-computed outputs; do not re-execute at build
# time (avoids downloading models / heavy deps in CI). Matches the varunayan setup.
nb_execution_mode = "off"
nb_merge_streams = True

# Mock heavy optional dependencies so autodoc can import alethia without them.
autodoc_mock_imports = []
for _module in ("torch", "sentence_transformers", "fastembed", "faiss", "umap"):
    try:
        importlib.import_module(_module)
    except Exception:
        autodoc_mock_imports.append(_module)

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

latex_documents = [
    (master_doc, "alethia.tex", "Alethia Documentation", "Saket Choudhary", "manual"),
]

# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "alethia", "Alethia Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "alethia",
        "Alethia Documentation",
        author,
        "alethia",
        "Entity matching and standardization using language model embeddings.",
        "Miscellaneous",
    ),
]

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "alethiadoc"
