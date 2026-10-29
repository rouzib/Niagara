"""Sphinx configuration for the Niagara documentation site."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

project = "Niagara"
author = "Nicolas Payot and Niagara contributors"
copyright = "2026, Nicolas Payot and Niagara contributors"

# Read a literal version without importing the GPU runtime while configuring
# Sphinx, or accidentally using another checkout's installed distribution.
tree = ast.parse((ROOT / "src/niagara/__init__.py").read_text(encoding="utf-8"))
release = next(
    ast.literal_eval(node.value)
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
)
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
source_encoding = "utf-8"
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", ".ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# Hosted builders do not have the locally qualified pair of CUDA devices.
# Notebook execution is a separate, explicit validation step.
nb_execution_mode = "off"
nb_execution_raise_on_error = True
nb_merge_streams = True
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

autodoc_class_signature = "separated"
autodoc_default_options = {"show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
autosummary_generate = False

# Keep a strict HTML build useful without access to documentation inventories.
# Explicit external links remain available in either mode.
intersphinx_mapping = {}
if os.environ.get("NIAGARA_DOCS_NETWORK") == "1":
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "torch": ("https://docs.pytorch.org/docs/2.12/", None),
    }
linkcheck_report_timeouts_as_broken = False

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
html_theme = "pydata_sphinx_theme"
html_logo = "_static/niagara-logo.svg"
html_favicon = "_static/niagara-mark.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = [
    ("mermaid-pan-zoom.js", {"defer": "defer"}),
    ("sidebar-state.js", {"defer": "defer"}),
]
html_sidebars = {"**": ["sidebar-master.html"]}
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "sidebar_includehidden": True,
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "Caskade",
            "url": "https://github.com/Ciela-Institute/caskade",
            "icon": "fa-brands fa-github",
        },
    ],
}
html_title = f"Niagara {release} documentation"
html_show_sourcelink = True
html_last_updated_fmt = "%Y-%m-%d"

# Vendored browser assets retain their upstream licenses. Math and Mermaid do
# not require a CDN when serving the generated site locally.
mathjax_path = "vendor/mathjax/tex-mml-chtml.js"
mathjax3_config = {
    "options": {
        "ignoreHtmlClass": "mathjax_ignore_disabled",
        "processHtmlClass": "tex2jax_process|mathjax_process|math|output_area",
    },
}
mermaid_use_local = "vendor/mermaid/mermaid-module.js"
mermaid_output_format = "raw"
mermaid_init_config = {
    "startOnLoad": False,
    "securityLevel": "strict",
    "themeVariables": {"lineColor": "#5e7f7a"},
}
mermaid_light_theme = "neutral"
mermaid_dark_theme = "dark"
mermaid_fullscreen = True
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
