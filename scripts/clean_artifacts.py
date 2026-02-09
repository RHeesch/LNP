#!/usr/bin/env python3
"""Remove generated log and ANML artifacts.

Default: remove exp/log/*.log and exp/log/*.anml (including loop files).
Use --all to also remove domain subdir artifacts under exp/log/<domain>/.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean generated artifacts in exp/log.")
    parser.add_argument("--all", action="store_true", help="Also remove exp/log/<domain>/ subdir artifacts.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    log_dir = repo_root / "exp" / "log"
    if not log_dir.exists():
        print(f"No log directory at {log_dir}")
        return 0

    removed = []
    for pattern in ("*.log", "*.anml"):
        for path in log_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed.append(path)

    if args.all:
        for sub in log_dir.iterdir():
            if sub.is_dir():
                shutil.rmtree(sub)
                removed.append(sub)

    print("Removed:")
    for path in removed:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
