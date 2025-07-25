# Texas Hold'em GTO Poker AI [INCOMPLETE]

A modular poker AI training system with schema-aligned feature extraction and unified analyzer architecture for Game Theory Optimal (GTO) training.

Current aim is to provide the AI with thorough equity and implied odds calculations by predicting opponent's range. 

Goals include: Debugging -> enhanced GTO convergence -> multi-way action (currently trains for heads-up) -> integration with a greedy exploiter layer that looks closer at opponent stats -> make playable online – anyone can practise and discover their own tendencies.

## Quick Start

```bash
# Train the GTO agent (Layer 1 - current primary training) [currently debugging]
python trainingL1/train_L1.py

# Play against an opponent [currently performs random actions, not GTO]
python playgame.py
```

## System Architecture

### Core Components

- **TexasHoldemEnvNew.py**: Enhanced poker game engine with improved state management
- **trainingL1/train_L1.py**: Layer 1 GTO training system with neural networks
- **poker_agents.py**: Unified neural network architecture and base agent classes
- **playgame.py**: PyQt GUI for human gameplay

### Key Features

- **Schema-Aligned Feature Extraction**: 184-dimensional feature space with poker_feature_schema.py
- **Unified Analyzer Architecture**: Modular analyzers/ directory with specialized components
- **History Tracking**: Comprehensive game state history with HistoryTracker
- **Feature Contexts**: Static and dynamic context separation for efficient processing
- **Layer 1 GTO Training**: Neural network-based training in trainingL1/ directory
- **Modular Design**: Clear separation between core engine, feature extraction, and training
- **Dynamic Opponent Modeling & Range Prediction**: Incorporates historical statistics (VPIP, PFR, etc.) and utilizes a novel auxiliary neural network for real-time opponent range estimation.

## Architecture Overview

### Feature Extraction Pipeline
- **poker_core.py**: Core game state representation
- **poker_feature_schema.py**: Master schema defining all feature categories  
- **feature_extractor.py**: Schema-aligned extraction with unified analyzers
- **feature_contexts.py**: Static/dynamic context management

### Analyzer Components
- **analyzers/hand_analyzer.py**: Hand strength and equity analysis
- **analyzers/board_analyzer.py**: Board texture and draw analysis
- **analyzers/current_street_analyzer.py**: Non-history current state features
- **analyzers/history_analyzer.py**: History-tracked features with proper integration
- **analyzers/history_tracking.py**: Comprehensive game state history tracking
- **analyzers/event_identifier.py**: Monotonic signals on player playstyle (VPIP, PFR, etc.)

### Range Predictor Components

### Training System
- **trainingL1/train_L1.py**: Main Layer 1 training script
- **trainingL1/network_trainer.py**: Neural network training logic
- **trainingL1/data_collector.py**: Training data collection and management
- **trainingL1/evaluator.py**: Model evaluation and performance metrics

## Development Guidelines

1. **Follow schema alignment**: Use poker_feature_schema.py for all feature definitions
2. **Modular development**: Keep analyzers, training, and core engine separate
3. **History tracking**: Use HistoryTracker for all stateful feature extraction
4. **Context separation**: Distinguish between static and dynamic contexts
5. **Layer 1 focus**: Current development centers on trainingL1/ directory

## File Structure

```
Poker/
├── poker_core.py              # Core game state representation
├── poker_feature_schema.py    # Master feature schema
├── feature_extractor.py       # Schema-aligned feature extraction
├── feature_contexts.py        # Context management
├── analyzers/                 # Modular analysis components
│   ├── hand_analyzer.py
│   ├── board_analyzer.py  
│   ├── current_street_analyzer.py
│   ├── history_analyzer.py
│   └── history_tracking.py
├── trainingL1/               # Layer 1 training system
│   ├── train_L1.py
│   ├── network_trainer.py
│   ├── data_collector.py
│   └── evaluator.py
├── TexasHoldemEnvNew.py      # Enhanced game engine
├── poker_agents.py           # Agent architecture
└── playgame.py              # GUI interface
```