"""
Trade Logger — structured JSONL log of every trade for failure pattern analysis.
"""
import json
import time
from pathlib import Path
from loguru import logger

LOG_PATH = Path('data/trades.jsonl')


def log_trade(event: dict):
    """Append a trade event to the JSONL log."""
    record = {**event, '_logged_at': time.time()}
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, 'a') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        logger.warning(f'[LOGGER] Could not log trade: {e}')


def load_recent(n: int = 200) -> list[dict]:
    """Load last N trade records."""
    if not LOG_PATH.exists():
        return []
    records = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records[-n:]


def win_rate_by_regime(records: list[dict]) -> dict:
    """Compute win rate and avg PnL grouped by regime."""
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in records:
        groups[r.get('regime', 'NA')].append(r.get('realized_pnl', 0))

    return {
        regime: {
            'trades': len(pnls),
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
            'avg_pnl': round(sum(pnls) / len(pnls), 4),
        }
        for regime, pnls in groups.items()
    }
