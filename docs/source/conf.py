project = 'SymQuPS'
copyright = '2026, Hendry M. Lim'
author = 'Hendry M. Lim'
release = '0.0.1'
html_theme = 'sphinx_rtd_theme'

import os
import sys
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../../src/symqups"))

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.autosummary",
              "sphinx.ext.napoleon",
              "sphinx_rtd_theme"]

autodoc_default_options = {
    "show-inheritance": True,
}

autodoc_member_order = "bysource"