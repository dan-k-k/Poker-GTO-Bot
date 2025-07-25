# Debug Output Guide

## Overview
Both AS and BR training now show detailed debug output every 5th episode to help monitor training progress and loss component balance.

## Debug Frequency
- **Trigger:** Every 5th episode (episodes 0, 5, 10, 15, 20, ...)
- **Indicator:** Training phase info shows "(debug)" when debug output will be displayed
- **Control:** Controlled by `show_debug = (episode % 5 == 0)` in both training paths

## AS (Average Strategy) Debug Output

```
=== AS Training Debug (episode 5) ===
   AS DEBUG: action=1.100692*1.0=1.100692
   AS DEBUG: bet=0.034649*0.3=0.010395
   AS DEBUG: entropy=1.096760*0.0498=0.054619
   AS DEBUG: total_loss=1.165705
```

**Components:**
- **action**: Cross-entropy classification loss for action selection × weight
- **bet**: MSE regression loss for bet sizing × weight  
- **entropy**: Exploration bonus (decays over time) × current weight
- **total_loss**: Sum of all weighted components

## BR (Best Response) Debug Output

```
=== BR Training Debug (episode 5) ===
   BR DEBUG: policy=0.058379*1.0=0.058379
   BR DEBUG: value=0.013621*0.5=0.006811
   BR DEBUG: entropy=-1.091768*0.01=-0.010918
   BR DEBUG: bet=0.036654*0.5=0.018327
   BR DEBUG: total_loss=0.072598
   BR DEBUG: avg_advantage=0.053986, avg_value=-0.038342, avg_reward=0.015644
```

**Components:**
- **policy**: PPO policy loss (actor) × weight
- **value**: Value function MSE loss (critic) × weight
- **entropy**: Exploration bonus (negative entropy) × weight
- **bet**: Reward-supervised bet sizing loss × weight
- **total_loss**: Sum of all weighted components

**Additional PPO Metrics:**
- **avg_advantage**: Average advantage (reward - predicted value)
- **avg_value**: Average predicted state value
- **avg_reward**: Average actual reward from experiences

## Interpreting Debug Output

### AS Training (Supervised Learning)
- **High action loss**: Network struggling to imitate BR actions
- **High bet loss**: Network struggling to match BR bet sizes
- **High entropy**: Network predictions are uncertain/random
- **Total loss trend**: Should generally decrease over time

### BR Training (Reinforcement Learning)  
- **High policy loss**: Network not learning profitable actions
- **High value loss**: Value function not accurately predicting rewards
- **Negative entropy**: Good - means network is becoming more decisive
- **High bet loss**: Bet sizing not aligned with profitable outcomes
- **Positive avg_advantage**: Network underestimating value (learning opportunity)
- **Negative avg_advantage**: Network overestimating value (potential overfitting)

## Loss Weight Impact

The debug output shows both raw losses and weighted contributions:
```
bet=0.034649*0.3=0.010395
```
- Raw bet loss: 0.034649
- Weight: 0.3
- Weighted contribution: 0.010395

This helps you understand:
1. Which components are naturally largest
2. How weights affect final loss composition
3. Whether weight adjustments are needed

## Adjusting Based on Debug Output

If you see:
- **AS action loss >> bet loss**: Consider increasing `as_bet_weight`
- **BR policy loss >> other losses**: Consider increasing other component weights
- **Entropy too high/low**: Adjust entropy weights for desired exploration
- **Large advantage values**: May need to adjust value function weight

## Example Usage

```python
# Monitor debug every 5th episode automatically
trainer = NFSPTrainer()
trainer.train_gto(episodes=100, hands_per_episode=200)

# Manually trigger debug for testing
trainer.network_trainer.train_average_strategy(show_debug=True)
trainer.network_trainer.train_best_response(show_debug=True)
```