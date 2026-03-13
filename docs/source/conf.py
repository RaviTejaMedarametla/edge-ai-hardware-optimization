import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

project = 'Edge AI Hardware Optimization'
copyright = '2026, RaviTejaMedarametla'
author = 'RaviTejaMedarametla'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
