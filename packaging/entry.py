"""Frozen-application entry point.

The package's own ``__main__.py`` uses a relative import, which is correct for
``python -m corridor`` but fails when PyInstaller runs a file as the top-level
script: there is no parent package, so ``from .cli import main`` raises
ImportError. This launcher uses an absolute import instead.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    # Torch and Cellpose may start worker processes; without this a frozen
    # Windows build can re-launch the whole application instead of a worker.
    multiprocessing.freeze_support()

    from corridor.cli import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
