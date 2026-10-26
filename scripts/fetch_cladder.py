"""Download the pinned CLadder v1.5 data and generator files into data/raw and check them by SHA-256.

    python scripts/fetch_cladder.py

The files and their hashes are listed in inputs.json; a download that differs from its hash is refused.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    spec = json.loads((ROOT / "inputs.json").read_text(encoding="utf-8"))
    for source in spec.values():
        for name, sha in source["files"].items():
            target = ROOT / "data" / "raw" / source["dir"] / name
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sha:
                continue
            data = urllib.request.urlopen(source["url"].format(rev=source["revision"], file=name), timeout=120).read()
            if hashlib.sha256(data).hexdigest() != sha:
                raise SystemExit(f"{name}: SHA-256 differs from inputs.json; nothing written")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        print(source["dir"], len(source["files"]), "files ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
