"""Alias entry point for student training (Phase 9) - same driver, mode=student."""
from __future__ import annotations

import sys

from .train_teacher import main as _main


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = ["--mode", "student"] + [a for a in argv if a not in ("--mode", "student")]
    return _main(argv)


if __name__ == "__main__":
    sys.exit(main())
