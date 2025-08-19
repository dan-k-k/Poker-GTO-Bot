# app/fixL1/fix_hyper_aggression.py
# !/usr/bin/env python3
"""
Fix hyper-aggression by adjusting reward weights in data_collector.py
"""

import sys
import os

def fix_reward_weights():
    """Fix the reward weighting to prioritize actual chip wins over equity changes."""
    print("🔧 Fixing Hyper-Aggression: Adjusting Reward Weights")
    print("=" * 55)
    
    data_collector_path = "trainingL1/data_collector.py"
    
    if not os.path.exists(data_collector_path):
        print("❌ data_collector.py not found")
        return False
    
    # Read the file
    with open(data_collector_path, 'r') as f:
        content = f.read()
    
    # Current problematic pattern
    old_pattern = "end_reward = hand_profit / 200.0 * 0.4  # Reduced from 1.0 to 0.4\n                \n                # Equity-based immediate reward (higher weight)\n                equity_reward = exp.get('equity_reward', 0.0) * 0.6"
    
    # New balanced pattern  
    new_pattern = "end_reward = hand_profit / 200.0 * 0.8  # Increased: prioritize actual wins\n                \n                # Equity-based immediate reward (reduced weight)\n                equity_reward = exp.get('equity_reward', 0.0) * 0.2"
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        print("✅ Found and fixed AS reward weights")
    else:
        print("⚠️  AS reward pattern not found - may already be fixed")
    
    # Also fix BR version
    old_pattern_br = "end_reward = hand_profit / 200.0 * 0.4  # Reduced from 1.0 to 0.4\n                \n                # Equity-based immediate reward (higher weight)\n                equity_reward = exp.get('equity_reward', 0.0) * 0.6"
    
    if old_pattern_br in content:
        content = content.replace(old_pattern_br, new_pattern)
        print("✅ Found and fixed BR reward weights")
    
    # Write back
    with open(data_collector_path, 'w') as f:
        f.write(content)
    
    print("\n📊 NEW REWARD STRUCTURE:")
    print("  End-of-hand reward (actual chip wins): 80%")  
    print("  Equity-based reward (PBRS): 20%")
    print("\n💡 This should encourage agents to:")
    print("  - Care more about actually winning chips")
    print("  - Fold when they're likely to lose")
    print("  - Develop more strategic diversity")
    
    return True

def suggest_additional_fixes():
    """Suggest other potential fixes."""
    print("\n🔧 OTHER POTENTIAL FIXES TO TRY:")
    print("-" * 35)
    print("1. If problem persists, try:")
    print("   - Temporarily disable equity rewards (set to 0.0)")
    print("   - Clear training buffers and restart training")
    print("   - Reduce equity simulation count (faster but less accurate)")
    
    print("\n2. Range adjustments in equity_calculator.py:")
    print("   - Make opponent ranges tighter/stronger")
    print("   - Reduce calling frequencies in FLOP_CALL_WEIGHTED")
    print("   - Increase 3-bet frequencies in defensive ranges")
    
    print("\n3. Monitor these files:")
    print("   - training_output/hand_histories.log (see actual play)")
    print("   - Console output for [REWARD DEBUG] messages")
    print("   - Action frequency diagnostics in training output")

if __name__ == "__main__":
    success = fix_reward_weights()
    if success:
        suggest_additional_fixes()
        print("\n🚀 Ready to resume training with fixed reward weights!")
    else:
        print("❌ Could not apply fixes automatically")

