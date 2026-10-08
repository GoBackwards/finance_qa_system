"""Run from the project root after installing the experiments extra."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from report_chunker.experiment import main

if __name__=='__main__':
    raise SystemExit(main())
