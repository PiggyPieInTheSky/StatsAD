# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys
import os

main_folder = pathlib.Path(__file__).parents[2].resolve().as_posix()
sys.path.insert(0, main_folder)
# load the __version__ attribute
exec(open(os.path.join(main_folder, 'anomdetect', 'version.py')).read())


project = 'AnomDetect'
copyright = '2023, PiggyPieInTheSky'
author = 'PiggyPieInTheSky'
release = __version__
version = __version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'# 'furo', 'sphinx_rtd_theme', 'alabaster'
#html_static_path = ['_static']


# -- Options for EPUB output
epub_show_urls = 'footnote'
