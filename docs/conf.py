"""Sphinx configuration for the Elómë documentation.

The site is static: the tutorial notebooks are committed with their
outputs (executed locally, on Apple Silicon) and the API reference is
generated from the package by ``docs/make_api.py``, so building the docs
needs neither MLX nor a GPU.
"""

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
_meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

project = "Elómë"
author = "David Kipping"
copyright = "2026, David Kipping"
release = _meta["version"]
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",               # Markdown (MyST) pages and the tutorial notebooks
    "sphinx_copybutton",
    "sphinx_design",         # cards and grids on the landing page
]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist",
                          "attrs_inline", "substitution"]
myst_heading_anchors = 3
nb_execution_mode = "off"    # notebooks are executed before they are committed
nb_merge_streams = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints"]

html_theme = "sphinx_book_theme"
html_title = "Elómë"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_extra_path = ["_extra"]  # robots.txt at the site root: keep crawlers out
html_theme_options = {
    "repository_url": "https://github.com/davidkipping/elome",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": False,
    "home_page_in_toc": True,
    "show_toc_level": 2,
    "logo": {
        "text": "Elómë",
        "image_light": "assets/elome_tengwar_light.svg",
        "image_dark": "assets/elome_tengwar_dark.svg",
    },
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
