# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'MOOSE'
copyright = '2023, Lalith Kumar Shiyam Sundar | Sebastian Gutschmayer, QIMP'
author = 'Lalith Kumar Shiyam Sundar | Sebastian Gutschmayer, QIMP'
release = '2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',  # Added intersphinx extension
    'sphinx.ext.viewcode',    # Added viewcode extension
    # ... any other extensions you might be using
]

# intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'moosez': ('https://moosez.readthedocs.io/en/latest/', None)
}

# Linking to source code on GitHub
def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return f"https://github.com/LalithShiyam/MOOSE/blob/main/{filename}.py"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
