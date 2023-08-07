# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from sphinx.application import Sphinx

sys.path.insert(0, os.path.abspath('..'))

project = 'MOOSE'
copyright = '2023, Lalith Kumar Shiyam Sundar | Sebastian Gutschmayer, QIMP'
author = 'Lalith Kumar Shiyam Sundar | Sebastian Gutschmayer, QIMP'
release = '2.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx', 
    'sphinx.ext.viewcode',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# GitHub linking
html_context = {
    "display_github": True,  # Add 'Edit on Github' link instead of 'View page source'
    "github_user": "LalithShiyam",
    "github_repo": "MOOSE",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Fixing the GitHub source link
def fix_github_source_link(app: Sphinx, pagename: str, templatename: str, context: dict, doctree):
    if "viewcode_target" in context:
        context["viewcode_target"] = context["viewcode_target"].replace("/moosez/moosez/", "/moosez/")

def setup(app: Sphinx):
    app.connect("html-page-context", fix_github_source_link)
