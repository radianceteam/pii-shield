"""The version is written in three places and they have to agree.

It is not decoration. This package is installed from git, and pip decides whether to
replace an installed copy by comparing version strings — a version left behind means a
user runs old code after doing everything right, and hears nothing about it.
"""

from __future__ import annotations

import pathlib
import re
import tomllib

import pii_shield

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_package_agrees_with_its_metadata():
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert pii_shield.__version__ == declared


def test_the_changelog_names_this_version():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    newest = re.search(r"^## (\S+)", changelog, re.M)
    assert newest, "CHANGELOG.md has no version heading"
    assert newest.group(1) == pii_shield.__version__, (
        f"the newest entry is {newest.group(1)}, the package says {pii_shield.__version__}"
    )
