"""Run without installing the local package: python scripts/parse_reports.py."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from report_parser.cli import main

if __name__=='__main__':
    raise SystemExit(main())
