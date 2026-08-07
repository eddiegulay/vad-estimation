#!/usr/bin/env python3
"""Fail loudly, up front, if the external research-vault data isn't reachable.

Run this before anything else in a new session/environment — every other
script assumes `configs/data/paths.yaml` resolves to real files.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vad.config import load_yaml  # noqa: E402


def main() -> int:
    paths_cfg = load_yaml(REPO_ROOT / "configs" / "data" / "paths.yaml")
    vault_root = Path(paths_cfg["vault_data_root"])

    print(f"Checking vault_data_root: {vault_root}")
    if not vault_root.is_dir():
        print(f"FAIL: vault_data_root does not exist or is not a directory: {vault_root}")
        print("The external research vault is not reachable from this machine/session.")
        return 1

    ok = True
    for name, rel_path in paths_cfg.get("corpora", {}).items():
        full_path = vault_root / rel_path
        exists = full_path.exists()
        status = "OK  " if exists else "FAIL"
        print(f"  [{status}] {name:16s} -> {full_path}")
        ok = ok and exists

    if not ok:
        print("\nOne or more corpora paths are missing. Check configs/data/paths.yaml.")
        return 1

    print("\nAll data paths OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
