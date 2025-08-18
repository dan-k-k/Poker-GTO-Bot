## Advanced GTO Poker AI 🤖
An implementation of a Game Theory Optimal (GTO) poker AI using Neural Fictitious Self-Play (NFSP). The agent learns a near-unexploitable strategy for two-player No-Limit Texas Hold'em.

### Key Features
- NFSP Architecture: A Best Response (BR) agent learns to exploit, while an Average Strategy (AS) agent learns to be unexploitable.

- Advanced Feature Engineering: Uses a comprehensive feature vector including raw game state, action sequences, and strategic features.

- Equity-Based Rewards: The BR agent learns from a dense, equity-based reward signal for faster learning.

- Opponent and Self Modeling: Leverages statistical models of both players to inform strategy.

- Curriculum Learning: Follows a phased training schedule for stable convergence.

### ⚙️ Installation
```Bash
# 1. Clone the repository
git clone https://github.com/dan-k-k/Poker-GTO-Bot
cd Poker-GTO-Bot

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the required packages
pip install -r requirements.txt
```
### 🚀 Training Workflow
Training follows a curriculum to ensure stable convergence. The process is to first bootstrap a basic agent, then train a model to predict opponent ranges, and finally, alternate between the two to iteratively improve.

#### 1. Bootstrap Agent & Generate Initial Data

Run the main training for ~100 episodes. This uses a heuristic to teach the agent basics and create the first dataset for the range predictor.

*Note: You are able to pause training at any time with Ctrl + C; resume with the usual command.*

```Bash
python -m trainingL1.train_L1
```
#### 2. Train Initial Range Predictor

Use the data from Step 1 to train the first version of the RangeNetwork.

```Bash
python -m range_predictor.train_range_predictor
```
#### 3. Iterative Refinement

Now, alternate between running the main agent training (which will automatically load and use the range model) and re-training the range predictor with the new, higher-quality data.

```Bash
# Run for another 200-300 episodes to generate better data
python -m trainingL1.train_L1

# Re-train the range predictor with the new data
python -m range_predictor.train_range_predictor

# Repeat this cycle
```