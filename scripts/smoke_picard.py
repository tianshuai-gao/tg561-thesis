# scripts/smoke_picard.py

#!/usr/bin/env python3
from pathlib import Path
from src.constraints.picard_wrapper import PicardConstraint, PicardConfig


def main():
    repo_root = Path(__file__).resolve().parents[1]
    picard_root = repo_root / "external" / "picard"
    c = PicardConstraint(PicardConfig(picard_root=picard_root))
    c.reset()

    good = "SELECT * FROM author;"
    bad = "SELECT FROM ;"

    print("GOOD:", c.step(good).ok)
    c.reset()
    print("BAD :", c.step(bad).ok)


if __name__ == "__main__":
    main()
