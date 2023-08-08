# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Insert the parent directory (where your code is) into the system path
sys.path.insert(0, os.path.abspath('..'))

project = 'MOOSE'
copyright = '2023, Lalith Kumar Shiyam Sundar | Sebastian Gutschmayer, QIMP'
author = 'Lalith Kumar Shiyam Sundar | Sebastian Gutschmayer, QIMP'
release = '2.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx', 
    'sphinx.ext.viewcode',
    'sphinx.ext.linkcode', 
    'sphinx_rtd_dark_mode', # Add this line for direct linking to GitHub source
]

# Intersphinx mapping for external libraries
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Theme and static files
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# GitHub linking for "Edit on Github" feature
html_context = {
    "display_github": True,
    "github_user": "LalithShiyam",
    "github_repo": "MOOSE",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Function to resolve direct linking to GitHub source
def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return f"https://github.com/LalithShiyam/MOOSE/blob/main/{filename}.py"


html_theme_options = {
    "style_nav_header_background": "#343131",  # Optional: Change the navbar header color
    "dark_mode_theme": "darkly",  # Optional: Set the dark mode theme to "darkly"
}

html_css_files = [
    'custom.css',
]

html_logo = '_static/Moose-logo-new.png'



