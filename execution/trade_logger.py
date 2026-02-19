import csv
import os
import pandas as pd
from loguru import logger
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────
# This path is relative to where the bot runs (usually root)
TRADE_LOG_FILE = "data/trade_features.csv"

# Columns to track - MUST match keys in adaptive_weights.DEFAULT_WEIGHTS
# Plus outcome columns
LOG_COLUMNS = [
    "trade_id",          # Primary Key
    "timestamp",
    "symbol",
    "direction",
    # Factors (continuous 0-1 range)
    "mtf_alignment",
    "ofi",
    "cvd",
    "sweep",
    "liquidity",
    "vol_expansion",
    "correlation",
    "signal_strength",
    "execution_quality",
    # Metadata
    "regime",
    "session",
    "spread_at_entry",
    "entry_price",
    # Outcome (filled on exit)
    "exit_price",
    "exit_reason",
    "hold_time",
    "result_r",          # Target variable (R-multiple)
]

def _ensure_log_exists():
    """Create CSV if missing."""
    if not os.path.exists(TRADE_LOG_FILE):
        os.makedirs(os.path.dirname(TRADE_LOG_FILE), exist_ok=True)
        try:
            with open(TRADE_LOG_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(LOG_COLUMNS)
        except Exception as e:
            logger.error(f"[LOGGER] Failed to init log file: {e}")

def log_trade_entry(pos_id: str, features: dict):
    """
    Log factor values at the moment of trade entry.
    Args:
        pos_id: The unique trade ID from position_manager
        features: Dictionary containing factor values and metadata
    """
    _ensure_log_exists()
    
    # Prepare row
    row = {
        "trade_id": pos_id,
        "timestamp": datetime.now().isoformat(),
        # Fill factors from features dict, default to 0 if missing
        **{col: features.get(col, 0) for col in LOG_COLUMNS if col not in ["trade_id", "timestamp", "result_r", "exit_reason", "hold_time", "exit_price"]}
    }
    
    # Write append
    try:
        with open(TRADE_LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writerow(row)
        logger.info(f"[LOGGER] Entries logged for trade {pos_id}")
    except Exception as e:
        logger.error(f"[LOGGER] Failed to log entry: {e}")

def log_trade_exit(pos_id: str, result_r: float, exit_reason: str, exit_price: float, hold_time: float):
    """
    Update the trade record with outcome data.
    Uses pandas for ease of update (read-modify-write).
    """
    if not os.path.exists(TRADE_LOG_FILE):
        return

    try:
        df = pd.read_csv(TRADE_LOG_FILE)
        
        # Check if trade exists
        mask = df["trade_id"] == pos_id
        if not mask.any():
            logger.warning(f"[LOGGER] Trade ID {pos_id} not found in log for update")
            return

        # Update row
        idx = df.index[mask][0]
        df.at[idx, "result_r"] = result_r
        df.at[idx, "exit_reason"] = exit_reason
        df.at[idx, "exit_price"] = exit_price
        df.at[idx, "hold_time"] = hold_time
        
        # Save back
        df.to_csv(TRADE_LOG_FILE, index=False)
        logger.info(f"[LOGGER] Outcome logged for {pos_id}: R={result_r:.2f}")
        
    except Exception as e:
        logger.error(f"[LOGGER] Failed to log exit: {e}")
