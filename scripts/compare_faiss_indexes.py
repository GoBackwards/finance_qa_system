"""Run isolated Flat, IVF and HNSW construction/search measurements."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from report_index.comparison_experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
