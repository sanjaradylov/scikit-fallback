"""Configuration file for the Sphinx documentation builder."""

import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

# region Project information
project = "scikit-fallback"
copyright = "2024-2026, Sanjar Ad[yi]lov"
author = "Sanjar Ad[yi]lov"

release = "0.2.0"
version = "0.2"
# endregion

# region General configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "numpydoc",
]
autosummary_generate = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

pygments_style = None
# endregion

# Options for HTML output
html_theme = "sphinx_rtd_theme"

# Options for EPUB output
epub_show_urls = "footnote"
