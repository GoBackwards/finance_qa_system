"""Explain lexical hints and explicit corpus scope without loading any models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from report_search.query_policy import plan_query


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language analyst question")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"),
                        help="Inventory of the reports being searched (default: data/manifest.json)")
    parser.add_argument("--output", type=Path,
                        help="Save UTF-8 JSON to a new file instead of printing; existing files are preserved")
    args = parser.parse_args(argv)
    try:
        inventory = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        plan = plan_query(args.query, inventory)
        if args.output is None:
            # Escaped Unicode is portable even on legacy Windows stdout codecs.
            print(json.dumps(plan, indent=2, ensure_ascii=True, allow_nan=False))
        else:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
