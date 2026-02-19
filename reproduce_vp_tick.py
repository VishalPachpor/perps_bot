
import sys
import os

# Add relevant paths to sys.path
sys.path.append('/Users/vishalpatil/perp-signals/src')
sys.path.append('/Users/vishalpatil/perps-bot')

try:
    import perp_signals.constants
    print(f"Constants file: {perp_signals.constants.__file__}")
    
    from perp_signals.constants import VP_TICK
    print(f"Direct import VP_TICK: {type(VP_TICK)} = {VP_TICK}")

    from perp_signals.layer5_microstructure import VP_TICK as VP_TICK_L5
    print(f"Layer 5 import VP_TICK: {type(VP_TICK_L5)} = {VP_TICK_L5}")

except Exception as e:
    print(f"Error: {e}")
