"""Sphinx configuration for Alethia documentation."""

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
    "myst_parser",
]

# Templates
templates_path = ["_templates"]

# Source files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Master document
master_doc = "index"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Pygments style
pygments_style = "sphinx"
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
    "light_css_variables": {
        "color-brand-primary": "#2962FF",
        "color-brand-content": "#2962FF",
        "color-code-background": "#f6f8fa",
        "color-code-foreground": "#24292e",
    },
    "dark_css_variables": {
        "color-brand-primary": "#82B1FF",
        "color-brand-content": "#82B1FF",
        "color-code-background": "#1d2128",
        "color-code-foreground": "#e6edf3",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/saketkc/alethia",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_title = f"{project} {version}"
html_logo = "logo.svg"
html_favicon = "logo.svg"

html_static_path = ["_static"]

# Custom CSS
html_css_files = [
    "custom.css",
]

# Social links in footer
html_context = {
    "display_github": True,
    "github_user": "saketkc",
    "github_repo": "alethia",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Options for copybutton --------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True

# -- Options for myst-parser -------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

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
