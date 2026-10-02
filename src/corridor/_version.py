"""Corridor's version: written here and nowhere else.

Everything that states a version derives from this line: ``app_meta``,
``pyproject.toml`` (setuptools reads it statically through ``attr``), and the
Windows version resource and Inno Setup define that ``scripts/sync_version.py``
generates. ``tests/test_version.py`` fails if any of them disagrees.

Three plain integers, nothing else. ``core.updates.is_newer`` compares exactly
three components, so a suffix such as ``2.0.0rc1`` would compare equal to the
final release, and the Windows resource cannot hold one at all.

2.0.0 because schema v2 removes the axis columns from ``tracks.csv`` and
``track_summary.csv``: a breaking change to the output format.

Kept to a single literal assignment so that setuptools and
``sync_version.py`` can read it without importing the package.
"""

__version__ = "2.0.0"
