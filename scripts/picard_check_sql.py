# scripts/picard_check_sql.py

#!/usr/bin/env python3
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path


# Minimal SQL sanity regex:
# - must have SELECT <something> FROM <something>
# - <something> must include at least one non-space char
_SQL_MIN_RE = re.compile(r"(?is)\bselect\b\s+(.+?)\s+\bfrom\b\s+(.+?)(;|\s|$)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql", required=True)
    ap.add_argument("--picard_root", required=True)
    args = ap.parse_args()

    picard_root = Path(args.picard_root)
    if not picard_root.exists():
        print(f"[ERR] picard_root not found: {picard_root}", file=sys.stderr)
        return 2

    sql = args.sql.strip()
    if not sql:
        return 1

    m = _SQL_MIN_RE.search(sql)
    if not m:
        return 1

    select_part = m.group(1).strip()
    from_part = m.group(2).strip()

    # reject empty select/from payloads
    if not select_part or not from_part:
        return 1

    # Optional: reject "SELECT FROM" style where select_part is just punctuation
    if all(ch in ",;" for ch in select_part):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
