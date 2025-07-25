# fixL1/analyze_reservoir.py
# !/usr/bin/env python3
"""
Analyze the reservoir buffer to understand what actions the agents are learning.
"""

import pickle
import os
from collections import Counter

def analyze_reservoir_buffer():
    """Analyze the training buffers to see action distribution."""
    print("🔍 Analyzing Reservoir Buffer Contents")
    print("=" * 50)
    
    buffer_path = "training_output/training_buffers.pkl"
    
    if not os.path.exists(buffer_path):
        print("❌ No training buffers found. Run some training first.")
        return
    
    try:
        with open(buffer_path, 'rb') as f:
            buffers = pickle.load(f)
        
        reservoir = buffers.get('reservoir_buffer', [])
        br_buffer = buffers.get('br_buffer', [])
        
        if not reservoir:
            print("❌ Reservoir buffer is empty.")
            return
        
        print(f"📊 Reservoir Buffer Analysis ({len(reservoir)} experiences)")
        print("-" * 30)
        
        # Action distribution
        action_counts = Counter(exp['action'] for exp in reservoir)
        action_names = {0: "FOLD", 1: "CALL/CHECK", 2: "BET/RAISE"}
        
        total_actions = sum(action_counts.values())
        print("Action Distribution:")
        for action_id, count in sorted(action_counts.items()):
            action_name = action_names.get(action_id, f"UNKNOWN({action_id})")
            percentage = (count / total_actions) * 100
            print(f"  {action_name}: {count:,} ({percentage:.1f}%)")
        
        # Bet sizing analysis
        bet_amounts = [exp['bet_amount'] for exp in reservoir if exp['action'] == 2 and exp['bet_amount'] > 0]
        if bet_amounts:
            avg_bet = sum(bet_amounts) / len(bet_amounts)
            min_bet = min(bet_amounts)
            max_bet = max(bet_amounts)
            print(f"\nBet Sizing Analysis:")
            print(f"  Average bet: {avg_bet:.1f}")
            print(f"  Min bet: {min_bet}")
            print(f"  Max bet: {max_bet}")
        
        # Recent vs older experiences
        if len(reservoir) > 1000:
            recent_actions = Counter(exp['action'] for exp in list(reservoir)[-1000:])
            old_actions = Counter(exp['action'] for exp in list(reservoir)[:1000])
            
            print(f"\nRecent vs Historical Comparison:")
            print("Recent 1000 experiences:")
            for action_id, count in sorted(recent_actions.items()):
                action_name = action_names.get(action_id, f"UNKNOWN({action_id})")
                percentage = (count / 1000) * 100
                print(f"  {action_name}: {percentage:.1f}%")
            
            print("Oldest 1000 experiences:")
            for action_id, count in sorted(old_actions.items()):
                action_name = action_names.get(action_id, f"UNKNOWN({action_id})")
                percentage = (count / 1000) * 100
                print(f"  {action_name}: {percentage:.1f}%")
        
        # Episode analysis
        if reservoir and 'episode' in reservoir[0]:
            episodes = [exp.get('episode', 0) for exp in reservoir]
            if episodes:
                print(f"\nEpisode Range: {min(episodes)} - {max(episodes)}")
        
        print("\n" + "=" * 50)
        
        # Prediction
        fold_percentage = (action_counts.get(0, 0) / total_actions) * 100
        bet_percentage = (action_counts.get(2, 0) / total_actions) * 100
        
        if fold_percentage < 5:
            print("⚠️  VERY LOW FOLD RATE: Agents may be hyper-aggressive!")
        if bet_percentage > 70:
            print("⚠️  VERY HIGH BET RATE: Agents may lack strategic diversity!")
        if fold_percentage < 5 and bet_percentage > 70:
            print("🚨 LIKELY PROBLEM: 'Bet-call, bet-call' degenerate strategy detected!")
            
        return action_counts
            
    except Exception as e:
        print(f"❌ Error analyzing buffers: {e}")
        return None

def suggest_fixes(action_counts):
    """Suggest fixes based on buffer analysis."""
    if not action_counts:
        return
        
    total = sum(action_counts.values())
    fold_rate = (action_counts.get(0, 0) / total) * 100
    bet_rate = (action_counts.get(2, 0) / total) * 100
    
    print("\n🔧 SUGGESTED FIXES:")
    print("-" * 20)
    
    if fold_rate < 10:
        print("1. INCREASE END REWARD WEIGHT:")
        print("   Current: end_reward * 0.4 + equity_reward * 0.6")
        print("   Try: end_reward * 0.8 + equity_reward * 0.2")
        print("   This makes agents care more about actually winning chips.")
    
    if bet_rate > 75:
        print("2. STRENGTHEN OPPONENT RANGES:")
        print("   Make FLOP_CALL_WEIGHTED and BB_DEFEND_WEIGHTED tighter.")
        print("   This teaches agents to respect strong opponents.")
        
    if fold_rate < 5 and bet_rate > 80:
        print("3. EMERGENCY MEASURES:")
        print("   - Temporarily disable equity rewards (set weight to 0.0)")
        print("   - Clear reservoir buffer and retrain from scratch")
        print("   - Add explicit folding incentives")

if __name__ == "__main__":
    action_counts = analyze_reservoir_buffer()
    suggest_fixes(action_counts)

