# docs/source/conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('../../meshai'))

# -- Project information -----------------------------------------------------

project = 'MeshAI SDK'
author = 'Robbie Tiwari'
release = '0.1.7'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',             # Automatically document modules
    'sphinx.ext.napoleon',            # Support for Google and NumPy docstrings
    'sphinx.ext.viewcode',            # Add links to highlighted source code
    'sphinx_autodoc_typehints',       # Include type hints in documentation
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

