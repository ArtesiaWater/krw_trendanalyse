# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "krw_trendanalyse_kwantiteit"
copyright = "2025, D.A. Brakenhoff"
author = "D.A. Brakenhoff"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autosectionlabel",
    "myst_nb",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for myst-nb ------------------------------------------------------

nb_execution_mode = "off"


# -- Autodoc, autosummary, and autosectionlabel settings ------------------------------

autodoc_typehints = "description"
autodoc_typehints_format = "short"
# autosummary_generate = True
# autoclass_content = "class"
autosectionlabel_prefix_document = True

# -- AutoAPI settings -----------------------------------------------------------------

autoapi_dirs = ["../krw_trendanalyse"]
autoapi_root = "api/"
