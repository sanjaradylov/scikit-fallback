"""Configuration file for the Sphinx documentation builder."""

# region Project information
project = "scikit-fallback"
copyright = "2024, Sanjar Ad[yi]lov"  # pylint: disable=redefined-builtin
author = "Sanjar Ad[yi]lov"

release = "0.1"
version = "0.1.0"
# endregion

# region General configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
# endregion

# Options for HTML output
html_theme = "sphinx_rtd_theme"

# Options for EPUB output
epub_show_urls = "footnote"