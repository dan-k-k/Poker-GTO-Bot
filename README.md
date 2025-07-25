# Texas Hold'em Poker AI Training System

A complete Neural Fictitious Self-Play (NFSP) implementation for training Game Theory Optimal (GTO) poker agents.

## Quick Start

```bash
# Train the GTO agent
python train_NN.py

# Play against the AI
python playgame.py
```

## System Architecture

### Core Components

- **TexasHoldem.py**: Complete poker game engine (2-9 players)
- **train_NN.py**: NFSP training system with GTO convergence
- **agents_NN.py**: Neural network poker agent
- **exploiter.py**: Layer 2 adaptive learning agent
- **playgame.py**: PyQt GUI for human gameplay

### Key Features

- **Neural Fictitious Self-Play**: Theoretically sound GTO training
- **65-dimensional feature extraction**: Comprehensive poker state representation
- **Bet sizing supervision**: Learns both action selection and bet amounts
- **Validation with early stopping**: Prevents overfitting
- **Multi-strategy evaluation**: Tests against tight-aggressive, loose-passive opponents
- **Graceful interruption**: Ctrl+C saves progress and allows resumption

## Training Parameters

- **Episodes**: 1000 (default)
- **Hands per episode**: 200
- **Learning rates**: 0.0005 (average), 0.0003 (best response)
- **Early stopping patience**: 50 episodes

## Git Workflow

```bash
# Before making changes
git checkout -b feature/your-feature-name

# After training improvements
git add train_NN.py
git commit -m "Improve NFSP convergence: adjust anticipatory parameter"

# Major milestones
git tag -a v1.1 -m "Version 1.1: Improved convergence"
```

## Development Guidelines

1. **Always commit before major training runs**
2. **Tag versions before significant changes**
3. **Use descriptive commit messages**
4. **Keep model files in .gitignore**

## Version History

- **v1.0**: Complete GTO training system ready for first training run