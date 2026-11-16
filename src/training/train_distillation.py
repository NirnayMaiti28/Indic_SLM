"""Alias entry point for knowledge-distillation training (Phase 10)."""
from __future__ import annotations

import sys

from .train_teacher import main as _main


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = ["--mode", "distill"] + [a for a in argv if a not in ("--mode", "distill")]
    return _main(argv)


if __name__ == "__main__":
    sys.exit(main())
