"""Single application entry point; implementation lives in src/prosus_search."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from prosus_search.cli import main


if __name__ == '__main__':
    raise SystemExit(main(default_config=ROOT / 'config.toml'))
