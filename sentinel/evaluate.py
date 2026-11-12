"""Run the monitoring sweep and save it.

    python -m sentinel.evaluate --out results/sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel.pipeline import run_sweep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quantiles", type=float, nargs="+", default=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--min-history", type=int, default=5)
    ap.add_argument("--out", default="results/sweep.json")
    args = ap.parse_args(argv)

    rows = run_sweep(args.quantiles, min_history=args.min_history)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"[evaluate] wrote {out}")


if __name__ == "__main__":
    main()
