# Poker AI Development Notes

## Current Focus
- Feature engineering improvements (poker_utils.py refinement)
- Cross-street betting memory
- Aggressor definition refinement
- Completing monotonic and one-hot features throughout codebase
- files working on/completed so far:
  - TexasHoldem to define game and everything on the ruleset
  - poker_utils (working on) to define the features that will be used for training
  - train_NN (working on) to train via CFR and true counterfactual.

## Long-term Goals
- if you notice long term goals straying from or building on what is listed here, always consult the user and update this file if necessary.

### Layer 1 (Current): GTO Foundation Agent
- Complete heads-up poker AI with rich feature extraction
- Neural Fictitious Self-Play (NFSP) training for GTO convergence
- Robust feature engineering with monotonic + one-hot encoding
- 611 -dimensional feature space with complete board representation
- train via CFR agent and true regret found via true counterfactual branching

### Layer 2 (Future): Exploitative Adaptation Layer
- MLP that learns via reinforcement learning to exploit specific player tendencies
- Wraps around completed Layer 1 as foundation
- Trains against exploitable bots with measurable weaknesses
- Tracks comprehensive player statistics:
  - VPIP, PFR, Limp %, 3-Bet %, Fold to 3-Bet %
  - Check-Raise %, AFq (Flop/Turn/River)
  - C-Bet % (Flop/Turn), Fold to Flop C-Bet %
  - WTSD %, Won at SD %, Avg SD Strength
- Adapts to recent playstyle changes for maximum exploitation

### Nice-to-Have (Post Layer 1+2 Completion)
- Multi-player support (currently just heads up)
- Multi-way support
- Tournament mode
- Advanced multi-way opponent modeling

## Known Issues
- Aggressor should be last raiser across streets, not just current bettor
- Need better cross-street memory for strategic context
- Feature extraction could be more efficient
- Layer 1/Layer 2 integration strategy needs clarification

## Design Philosophy
- Let neural network learn strategy from fundamental features
- Avoid pre-computing strategic conclusions
- Maintain information richness while being computationally efficient
- **Layer 1: Pure GTO foundation** - GTO opponent-specific information such as aggression is still useful.
- **Layer 2: Exploitative adaptation** - wraps Layer 1 with opponent modeling

## Feature Engineering Progress
1. ✅ Monotonic features (implemented in analyze_board_texture, categorize_hand_features)
2. ✅ One-hot board encoding (implemented in analyze_board_texture)  
3. 🔄 Monotonic/one-hot completion throughout poker_utils.py (in progress)
4. ⏳ Better aggressor tracking (pending Layer 2 integration decision)
5. ⏳ Cross-street betting patterns enhancement
6. ⏳ Advanced position awareness

## Architecture Decision Needed

**Layer 1/Layer 2 Integration Question:**
How should Layer 2 influence Layer 1's decision making?

**Option A: Feature Injection**
- Layer 2 adds opponent tendency features into Layer 1's 172-dimensional input
- Train both layers together end-to-end
- Pros: Unified training, seamless integration
- Cons: Complex training, harder to debug

**Option B: Post-Processing**
- Layer 1 outputs GTO action probabilities
- Layer 2 adjusts these probabilities based on opponent stats
- Train layers separately
- Pros: Modular, easier to debug, can swap Layer 2 models
- Cons: May not be optimal, two-stage pipeline

**Option C: Hybrid Features**
- Keep current aggressor features simple in Layer 1
- Add opponent-aware features only when Layer 2 is ready
- Pros: Incremental development, maintains current progress
- Cons: May require feature engineering rework later

## Questions/Decisions Needed
- how will CFR learn via true counterfactual and how will it decide when a branch is too weak to continue from previous experience?
- Should we complete Layer 1 feature engineering before tackling Layer 2 integration?
- How to balance Layer 1 GTO purity vs Layer 2 exploitative features?
- Should aggressor tracking wait for Layer 2, or implement now?
- What's the training strategy: joint or separate layer training?
- how will the agent see its opponent's hand at showdown to make adjustments?

## Current Work Status
- Currently reviewing poker_utils.py (at extract_betting_features)
- Monotonic and one-hot features still being completed throughout codebase
- Layer 2 legacy files exist but need updating for new architecture