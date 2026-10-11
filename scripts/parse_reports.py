"""Run without installing the local package: python scripts/parse_reports.py --output data/processed_v2.

``--output`` is required here: the parser's own default, data/processed, is
corpus v1, which this parser can no longer reproduce (Table A14), so writing
there would overwrite it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from report_parser.cli import main  # noqa: E402

FROZEN = (ROOT / 'data' / 'processed').resolve()  # corpus v1


def refusal(argv):
    """Why these arguments must not run, or None."""
    if any(arg in ('-h', '--help') for arg in argv):
        return None
    outputs = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == '--output']
    outputs += [arg.split('=', 1)[1] for arg in argv if arg.startswith('--output=')]
    if not outputs:
        return ('pass --output (data/processed_v2 for the demo corpus); the default, data/processed, '
                'is corpus v1 and is read-only')
    if any((ROOT / output).resolve() == FROZEN or Path(output).resolve() == FROZEN for output in outputs):
        return 'data/processed is corpus v1 and is read-only; write to data/processed_v2 or a scratch folder'
    return None


if __name__ == '__main__':
    problem = refusal(sys.argv[1:])
    if problem:
        raise SystemExit(f'parse_reports.py: {problem}')
    raise SystemExit(main())
