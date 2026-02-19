"""
Smoke test — verifies all modules import cleanly and key interfaces match.
Run: python3 smoke_test.py
"""
import sys

errors = []

def assert_(cond):
    if not cond:
        raise AssertionError("assertion failed")

def check(label, fn):
    try:
        fn()
        print(f"  ✅ {label}")
    except Exception as e:
        errors.append((label, e))
        print(f"  ❌ {label}: {e}")

print("=== Smoke Test ===\n")

# 1. Config loads
print("[1] Config")
check("config imports", lambda: __import__('config'))

# 2. Buffer
print("[2] Buffer")
from data.buffer import buffer
check("buffer singleton exists", lambda: assert_(buffer is not None))
check("buffer.states has symbols", lambda: assert_(len(buffer.states) > 0))
check("get_mark_price('ETH')", lambda: buffer.get_mark_price('ETH'))
check("get_trades('ETH')", lambda: buffer.get_trades('ETH'))
check("get_funding('ETH')", lambda: buffer.get_funding('ETH'))
check("get_depth('ETH')", lambda: buffer.get_depth('ETH'))
check("get_book_snapshots('ETH')", lambda: buffer.get_book_snapshots('ETH'))

# 3. perp_signals registration
print("[3] perp_signals registration")
import perp_signals
perp_signals.set_buffer(buffer)
check("set_buffer OK", lambda: None)

# 4. All signal layers import
print("[4] Signal layer imports")
check("layer1_mtf", lambda: __import__('perp_signals.layer1_mtf'))
check("layer2_orderflow", lambda: __import__('perp_signals.layer2_orderflow'))
check("layer3_correlation", lambda: __import__('perp_signals.layer3_correlation'))
check("layer4_funding", lambda: __import__('perp_signals.layer4_funding'))
check("layer5_microstructure", lambda: __import__('perp_signals.layer5_microstructure'))
check("layer5_vp", lambda: __import__('perp_signals.layer5_vp'))
check("layer6_cvd", lambda: __import__('perp_signals.layer6_cvd'))
check("session_gate", lambda: __import__('perp_signals.session_gate'))
check("setups", lambda: __import__('perp_signals.setups'))

# 5. Execution modules
print("[5] Execution modules")
check("risk", lambda: __import__('execution.risk'))
check("position_manager", lambda: __import__('execution.position_manager'))
check("venue_client", lambda: __import__('execution.venue_client'))
check("fee_model", lambda: __import__('execution.fee_model'))

# 6. Risk manager record_outcome signature
print("[6] Interface checks")
from execution.risk import risk_manager
import inspect
sig = inspect.signature(risk_manager.record_outcome)
params = list(sig.parameters.keys())
check("record_outcome accepts (symbol, pnl)", lambda: assert_(params == ['self', 'symbol', 'pnl'] or params == ['symbol', 'pnl']))

# 7. Dashboard
print("[7] Dashboard")
check("dashboard.server imports", lambda: __import__('dashboard.server'))

# 8. Monitoring
print("[8] Monitoring")
check("telegram", lambda: __import__('monitoring.telegram'))
check("trade_logger", lambda: __import__('monitoring.trade_logger'))

# 9. VP_TICK robustness
print("[9] VP_TICK type safety")
from perp_signals.constants import VP_TICK
check(f"VP_TICK type={type(VP_TICK).__name__}", lambda: None)
from perp_signals.layer5_microstructure import get_microstructure_signals
check("get_microstructure_signals('ETH') doesn't crash", lambda: get_microstructure_signals('ETH'))

print(f"\n{'='*40}")
if errors:
    print(f"❌ {len(errors)} FAILURES:")
    for label, e in errors:
        print(f"   - {label}: {e}")
    sys.exit(1)
else:
    print("✅ All checks passed!")
    sys.exit(0)
