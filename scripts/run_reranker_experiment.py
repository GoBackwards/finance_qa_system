"""Run the fixed BGE-reranker comparison over source-grounded vector candidates."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from report_search.rerank_experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
