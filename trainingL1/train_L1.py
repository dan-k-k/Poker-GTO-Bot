# trainingL1/train_L1.py
# Pure GTO training using Neural Fictitious Self-Play - Refactored with modular components

import random
from collections import deque, defaultdict
from typing import Dict

# Import parent directory modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TexasHoldemEnvNew import TexasHoldemEnv
from feature_extractor import FeatureExtractor
from poker_agents import GTOPokerNet

# Use consistent, package-aware relative imports
from .data_collector import DataCollector
from .training_utils import TrainingUtils, TrainingWatchdog
from .action_selector import ActionSelector
from .network_trainer import NetworkTrainer
from .evaluator import Evaluator
from .stack_depth_simulator import StackDepthSimulator
from stats_tracker import StatsTracker


class NFSPTrainer:
    """
    Neural Fictitious Self-Play trainer for pure GTO convergence.
    Implements the approach used in DeepStack for theoretically sound poker AI.
    
    This version uses modular components for better code organization and maintainability.
    """
    
    def __init__(self, num_players=2, mean_stack_bb=100, big_blind_chips=2, 
                 session_length=20, std_stack_bb=35, min_stack_bb_percent=0.15,
                 enable_stack_depth_simulation=True, loss_weight_config='default',
                 profit_reward_weight=0.3, equity_reward_weight=0.7,
                 bootstrap_episodes=10, br_frequency=2, as_frequency=1):
                
        # 1. Core Parameters (single source of truth)
        self.num_players = num_players
        self.big_blind_chips = big_blind_chips
        self.mean_stack_bb = mean_stack_bb
        
        # 2. Derived Chip and BB Values
        starting_stack_chips = int(self.mean_stack_bb * self.big_blind_chips)
        total_chips = self.num_players * starting_stack_chips
        min_stack_bb = int(self.mean_stack_bb * min_stack_bb_percent)
        small_blind_chips = self.big_blind_chips // 2
        
        print(f"🎯 Centralized Config:")
        print(f"   Mean Stack: {mean_stack_bb} BB = {starting_stack_chips} chips")
        print(f"   Total Chips: {total_chips} chips ({total_chips // big_blind_chips} BB total)")
        print(f"   Blinds: {small_blind_chips}/{big_blind_chips} chips")
        print(f"   Min Stack: {min_stack_bb} BB = {min_stack_bb * big_blind_chips} chips")
        
        # 3. Initialize Environment with derived chip values
        self.env = TexasHoldemEnv(
            num_players=self.num_players,
            starting_stack=starting_stack_chips,
            small_blind=small_blind_chips,
            big_blind=self.big_blind_chips
        )
        self.feature_extractor = FeatureExtractor(num_players=self.num_players)
        
        # 4. Initialize Networks
        self.avg_pytorch_net = GTOPokerNet(input_size=790)
        self.br_pytorch_net = GTOPokerNet(input_size=790)
        
        # 5. Initialize Simulator with derived values
        self.stack_depth_simulator = None
        if enable_stack_depth_simulation:
            self.stack_depth_simulator = StackDepthSimulator(
                total_chips=total_chips,
                session_length=session_length,
                mean_stack_bb=self.mean_stack_bb,
                std_stack_bb=std_stack_bb,
                min_stack_bb=min_stack_bb,
                big_blind=self.big_blind_chips
            )
                
        # 🎯 Reward Shaping Weights - Tunable hyperparameters
        self.profit_reward_weight = profit_reward_weight
        self.equity_reward_weight = equity_reward_weight
        
        # 🎯 Training Schedule Parameters - Configurable cycle settings
        self.bootstrap_episodes = bootstrap_episodes
        self.br_frequency = br_frequency
        self.as_frequency = as_frequency
        
        print(f"🎯 Training Schedule:")
        print(f"   Bootstrap: {bootstrap_episodes} BR episodes")
        print(f"   Cycle: {br_frequency} BR : {as_frequency} AS")
        
        # Initialize comprehensive stats tracker for opponent modeling
        stats_path = os.path.join('training_output', 'nfsp_training_stats.json')
        self.stats_tracker = StatsTracker(stats_path)
        
        # Agent-specific reporting counters for StatsTracker-based diagnostics
        self.as_episodes_since_last_report = 0
        self.br_episodes_since_last_report = 0
        
        # Period-based stats tracking (reset after each diagnostic report)
        self.as_period_hands = 0
        self.br_period_hands = 0
        self.as_baseline_stats = None  # Store stats at start of reporting period
        self.br_baseline_stats = None
        
        # History tracking for plotting agent evolution over time
        self.as_stat_history = []
        self.br_stat_history = []
        
        # --- START REFACTOR ---
        # 1. Create the core, shared components here, ONCE.
        # Use RangeConstructorNN as the main entry point - it will automatically
        # fallback to heuristics when the NN model isn't trained yet.
        from .range_constructors import RangeConstructorNN
        from .equity_calculator import EquityCalculator
        
        self.range_constructor = RangeConstructorNN()  # Smart controller with fallback
        self.equity_calculator = EquityCalculator()

        # 2. Pass these shared components to the modules that need them.
        self.feature_extractor = FeatureExtractor(
            num_players=self.num_players,
            range_constructor=self.range_constructor,
            equity_calculator=self.equity_calculator
        )
        
        # Initialize modular components with shared components
        self.data_collector = DataCollector(
            self.env, 
            self.feature_extractor, 
            self.stack_depth_simulator, 
            self.stats_tracker,
            profit_weight=self.profit_reward_weight, 
            equity_weight=self.equity_reward_weight,
            # Pass the shared components to the DataCollector as well
            range_constructor=self.range_constructor,
            equity_calculator=self.equity_calculator
        )
        # --- END REFACTOR ---
        
        self.action_selector = ActionSelector(self.env, self.avg_pytorch_net, self.br_pytorch_net)
        self.network_trainer = NetworkTrainer(self.avg_pytorch_net, self.br_pytorch_net)
        self.evaluator = Evaluator(
            self.env, 
            self.feature_extractor, 
            self.action_selector,
            bootstrap_episodes=self.bootstrap_episodes,
            br_frequency=self.br_frequency,
            as_frequency=self.as_frequency
        )
        
        # Load existing models if available
        TrainingUtils.load_existing_models(self.avg_pytorch_net, self.br_pytorch_net)
        
        # Apply loss weight configuration
        if loss_weight_config != 'default':
            print(f"🎯 Applying '{loss_weight_config}' loss weight configuration...")
            apply_config(self.network_trainer, loss_weight_config)
        else:
            print(f"🎯 Using default loss weights")
            weights = self.network_trainer.get_current_loss_weights()
            print(f"   AS: action={weights['as']['action']}, bet={weights['as']['bet']}, entropy={weights['as']['entropy_start']}→{weights['as']['entropy_end']}")
            print(f"   BR: policy={weights['br']['policy']}, value={weights['br']['value']}, entropy={weights['br']['entropy']}, bet={weights['br']['bet']}")
        
        # Display reward shaping weights
        print(f"🎯 Reward shaping weights")
        print(f"   Profit weight: {self.profit_reward_weight}; Equity weight: {self.equity_reward_weight}")
        
        # Training state for graceful exit
        self.training_interrupted = False
        self.current_episode = 0
        TrainingUtils.setup_graceful_exit(self)
        
        # Performance tracking (currently using simple schedule, but kept for future complex logic)
        self.as_performance_window = 5  # Episodes to track AS performance
        self.as_training_losses = deque(maxlen=self.as_performance_window)  # Actual training losses
        self.as_validation_losses = deque(maxlen=self.as_performance_window)  # Validation losses for overfitting detection
        self.as_chip_performance = deque(maxlen=self.as_performance_window)  # Chip EV per episode
        self.br_consecutive_training = 0  # Track consecutive BR training episodes
        self.as_struggling_threshold = 0.0  # Positive loss = struggling (losing money)
        self.as_chip_struggling_threshold = -5  # If losing > 5 chips/hand on average
        self.avg_network_update_freq = 5  # Update average network more frequently
        
        # Prioritized sampling for recent experiences
        self.recent_experiences = deque(maxlen=10000)  # Recent high-quality experiences
        
        # Training metrics
        self.performance_history = deque(maxlen=1000)
        self.exploitability_scores = deque(maxlen=100)
        self.avg_strategy_losses = []  # Track average strategy training loss
        self.best_response_losses = []  # Track best response training loss
        self.episode_numbers = []  # Track episode numbers for plotting
        
        # Exploitability-based early stopping
        self.best_exploitability = float('inf')
        self.patience = 100  # Episodes to wait before early stopping (increased for exploitability)
        self.patience_counter = 0
        self.min_episodes_before_stopping = 200  # Don't stop too early
        
    def train_gto(self, episodes: int = 1000, hands_per_episode: int = 200):
        """
        Main NFSP training loop for GTO convergence.
        
        Args:
            episodes: Total number of episodes to train (including already completed)
            hands_per_episode: Number of hands per episode (can be changed when resuming)
        """
        print("  Pure GTO Training with Neural Fictitious Self-Play")
        print("=" * 60)
        
        # Display stack depth simulation info
        if self.stack_depth_simulator:
            print(f"  Stack Depth Simulation Enabled:")
            print(f"   Session Length: {self.stack_depth_simulator.session_length} hands")
            print(f"   Mean Stack: {self.stack_depth_simulator.mean_stack_bb} BB")
            print(f"   Std Dev: {self.stack_depth_simulator.std_stack_bb} BB")
            print(f"   Min Stack: {self.stack_depth_simulator.min_stack_bb} BB")
        else:
            print("  Stack Depth Simulation: Disabled (every hand reset)")
        
        # Load previous training state
        start_episode = TrainingUtils.load_training_state(self)
        
        # Parameter change warnings
        if start_episode > 0:
            print(f"ℹ   Training parameters for resumed session:")
            print(f"   Episodes: {episodes} (continuing from {start_episode})")
            print(f"   Hands per episode: {hands_per_episode}")
            print(f"   Note: Changing hands_per_episode is safe and won't affect model quality")
        
        # Adjust episode range based on resume state
        if start_episode > 0:
            print(f"🔄 Resuming training from episode {start_episode + 1}")
            remaining_episodes = episodes - start_episode
            if remaining_episodes <= 0:
                print(f"✅ Training already completed! Requested {episodes} episodes, already trained {start_episode}")
                return
        
        for episode in range(start_episode, episodes):
            self.current_episode = episode
            
            # Wrap each episode with watchdog to detect freezes/infinite loops
            with TrainingWatchdog(seconds=30):
                #
                # EPISODE RESET LOGIC: Ensure every episode starts with a fresh, playable scenario.
                # This clears the board from the previous episode's results and prevents infinite loops.
                #
                # print(f"\n--- Episode {episode+1}/{episodes}: Resetting Scenario ---")
                if self.stack_depth_simulator:
                    # Tell the simulator to start a new session with new asymmetric stacks
                    new_stacks = self.stack_depth_simulator.reset_session(reason="new_episode")
                    self.env.reset(preserve_stacks=False)
                    self.env.state.stacks = list(new_stacks)
                    # print(f"🔄 Fresh stacks for episode: {new_stacks}")
                else:
                    # If not using the simulator, just do a standard reset.
                    self.env.reset(preserve_stacks=False)
                    print(f"DEBUG: 🔄 Standard reset for episode (no stack depth simulation)")
                #
                
                # Reset session tracking for new episode
                self.data_collector.reset_session_tracking()
                
                # Update episode in all components
                self.action_selector.set_current_episode(episode)
                self.network_trainer.set_current_episode(episode)
                
                # Check for interruption
                if self.training_interrupted:
                    print(f"\n💾 Saving progress and exiting gracefully...")
                    TrainingUtils.save_training_state(self)
                    TrainingUtils.save_models(self.avg_pytorch_net, self.br_pytorch_net, self.current_episode)
                    print(f"🛑 Training stopped at episode {episode}. \nResume (full power) with: python -m trainingL1.train_L1\nResume (efficiency core macOS) with: taskpolicy -b python -m trainingL1.train_L1")
                    return
                
                print(f"\n{'='*60}")
                print(f"Episode {episode+1}/{episodes}")
                
                # Get training decision and phase info
                train_best_response, decision_reason = self.evaluator.should_train_best_response(
                    episode, return_reason=True
                )
                
                # Show current training phase clearly
                self._print_training_phase_info(episode, train_best_response, decision_reason)
                
                if not train_best_response:  # Train Average Strategy
                    
                    # Define agent IDs for stats tracking
                    as_agent_id = "average_strategy_v1"
                    br_agent_id = "best_response_v1"
                    
                    # Create player map: P0=AS (training), P1=BR (opponent)
                    player_map = {0: as_agent_id, 1: br_agent_id}
                    
                    # Train average strategy network (GTO approximation)
                    experiences, win_rate = self.data_collector.collect_average_strategy_data(
                        hands_per_episode, self.action_selector, self.current_episode, player_map
                    )
                    
                    if len(experiences) > 32:  # Reduced threshold for faster startup
                        show_debug = (episode % 5 == 0)  # Show debug every 5th episode
                        avg_loss = self.network_trainer.train_average_strategy(show_debug=show_debug)
                        print(f"✅ AS Training Complete: Win Rate: {win_rate:.3f} | Loss: {avg_loss:.6f}")
                        
                        # Track loss for plotting
                        self.avg_strategy_losses.append(avg_loss)
                        self.best_response_losses.append(None)  # No BR training this episode
                        
                        # Track AS training performance for adaptive decisions
                        self.as_training_losses.append(avg_loss)
                        
                        # Track chip performance (more meaningful than win rate)
                        episode_chip_performance = sum(exp['reward'] for exp in experiences) * 200  # Convert back to chips
                        self.as_chip_performance.append(episode_chip_performance / len(experiences))  # Avg per hand
                    else:
                        self.avg_strategy_losses.append(None)
                        self.best_response_losses.append(None)
                        print(f"⏸️  AS Training Skipped: Not enough experiences ({len(experiences)} < 32)")
                        
                        # No training occurred, track placeholder values
                        if len(experiences) > 0:
                            episode_chip_performance = sum(exp['reward'] for exp in experiences) * 200
                            self.as_chip_performance.append(episode_chip_performance / len(experiences))
                            self.as_training_losses.append(0.0)  # No training loss
                    
                    # Process experiences for NFSP
                    self._process_as_experiences(experiences)
                        
                else:  # Train Best Response
                    # BR Training - show validation metrics
                    
                    # Define agent IDs for stats tracking
                    as_agent_id = "average_strategy_v1"
                    br_agent_id = "best_response_v1"
                    
                    # Create player map: P0=BR (training), P1=AS (opponent)
                    player_map = {0: br_agent_id, 1: as_agent_id}
                    
                    # Train best response network (exploiter)
                    experiences, win_rate = self.data_collector.collect_best_response_data(
                        hands_per_episode, self.action_selector, self.current_episode, player_map
                    )
                    
                    if len(experiences) > 32:  # Reduced threshold for faster startup
                        show_debug = (episode % 5 == 0)  # Show debug every 5th episode
                        br_loss = self.network_trainer.train_best_response(show_debug=show_debug)
                        
                        # Calculate BR validation loss for plateau detection
                        br_val_loss = self.network_trainer.calculate_br_validation_loss()
                        self.network_trainer.add_br_validation_loss(br_val_loss)
                        
                        # Reset baseline when switching to BR training (new AS to exploit)
                        if self.br_consecutive_training == 1:  # First BR episode after AS training
                            self.network_trainer.set_br_baseline_loss(br_val_loss)
                            print(f"   🎯 New BR baseline set: {br_val_loss:.6f} (facing updated AS)")
                        
                        # Check if BR validation loss has plateaued (stuck in local minima)
                        if self.network_trainer.should_mutate_br():
                            self.network_trainer.mutate_br_strategy()
                            self.network_trainer.set_last_mutation_episode(episode)
                            print("🧬 BR strategy mutation applied - validation loss plateaued")
                        
                        print(f"✅ BR Training Complete: Win Rate: {win_rate:.3f} | Loss: {br_loss:.6f} | Val Loss: {br_val_loss:.6f}")
                        
                        # Track loss for plotting
                        self.best_response_losses.append(br_loss)
                        self.avg_strategy_losses.append(None)  # No avg training this episode
                        
                        # Update BR consecutive training counter
                        self.br_consecutive_training += 1
                    else:
                        self.avg_strategy_losses.append(None)
                        self.best_response_losses.append(None)
                        print(f"⏸️  BR Training Skipped: Not enough experiences ({len(experiences)} < 32)")
                    
                    # Process experiences for NFSP
                    self._process_br_experiences(experiences)
                                
                # Always track episode number
                self.episode_numbers.append(episode)
                
                # # Print episode session summary
                # self.data_collector.print_episode_summary()
                
                # Track episodes for StatsTracker-based reporting
                if not train_best_response:
                    self.as_episodes_since_last_report += 1
                    # Set baseline at start of new AS reporting period
                    if self.as_episodes_since_last_report == 1:
                        as_agent_id = "average_strategy_v1"
                        self.as_baseline_stats = self.stats_tracker.get_player_percentages(as_agent_id)
                else:
                    self.br_episodes_since_last_report += 1
                    # Set baseline at start of new BR reporting period
                    if self.br_episodes_since_last_report == 1:
                        br_agent_id = "best_response_v1"
                        self.br_baseline_stats = self.stats_tracker.get_player_percentages(br_agent_id)
                
                # Print AS diagnostics every 12 AS episodes showing period changes
                if self.as_episodes_since_last_report >= 12:
                    print("\n" + "="*50)
                    print(f"  AS AGENT DIAGNOSTICS (Last {self.as_episodes_since_last_report} AS Episodes)")
                    print("="*50)
                    as_agent_id = "average_strategy_v1"
                    current_stats = self.stats_tracker.get_player_percentages(as_agent_id)
                    print(self._format_period_diagnostic(current_stats, self.as_baseline_stats, self.as_episodes_since_last_report))
                    print("="*50)
                    
                    # Save stats for plotting
                    stats_snapshot = current_stats.copy()
                    stats_snapshot['episode'] = episode
                    self.as_stat_history.append(stats_snapshot)
                    
                    # Generate real-time plot update
                    TrainingUtils.plot_agent_stats_evolution(self)
                    
                    self.as_episodes_since_last_report = 0
                    self.as_baseline_stats = None
                
                # Print BR diagnostics every 6 BR episodes showing period changes
                if self.br_episodes_since_last_report >= 6:
                    print("\n" + "="*50)
                    print(f"  BR AGENT DIAGNOSTICS (Last {self.br_episodes_since_last_report} BR Episodes)")
                    print("="*50)
                    br_agent_id = "best_response_v1"
                    current_stats = self.stats_tracker.get_player_percentages(br_agent_id)
                    print(self._format_period_diagnostic(current_stats, self.br_baseline_stats, self.br_episodes_since_last_report))
                    print("="*50)
                    
                    # Save stats for plotting
                    stats_snapshot = current_stats.copy()
                    stats_snapshot['episode'] = episode
                    self.br_stat_history.append(stats_snapshot)
                    
                    # Generate real-time plot update
                    TrainingUtils.plot_agent_stats_evolution(self)
                    
                    self.br_episodes_since_last_report = 0
                    self.br_baseline_stats = None
                    if episode % 50 == 0:  # Additional stats less frequently                    
                        # Print stack depth report if enabled
                        if self.stack_depth_simulator:
                            self.stack_depth_simulator.print_stack_depth_report()
                
                # Periodically update the average network with accumulated strategies (skip episode 0)
                if episode % self.avg_network_update_freq == 0 and episode > 0:
                    self.network_trainer.update_average_network()
                    
                # Measure exploitability and validation loss (skip episode 0)
                if episode % 20 == 0 and episode > 0:
                    exploitability = self.evaluator.measure_gto_exploitability()
                    self.exploitability_scores.append(exploitability)
                    print(f"📊 Exploitability Score: {exploitability:.4f}")
                    
                    # Exploitability-based early stopping
                    if episode >= self.min_episodes_before_stopping:
                        if exploitability < self.best_exploitability:
                            self.best_exploitability = exploitability
                            self.patience_counter = 0
                            print(f"✅ New best exploitability: {exploitability:.6f}")
                        else:
                            self.patience_counter += 1
                            print(f"⏳ Patience counter: {self.patience_counter}/{self.patience} (best: {self.best_exploitability:.6f})")
                            
                            if self.patience_counter >= self.patience:
                                print(f"🛑 Early stopping: No exploitability improvement for {self.patience} episodes")
                                break
                    
                    # Optional: Still calculate AS validation loss for monitoring
                    if len(self.network_trainer.as_validation_buffer) > 100:
                        val_loss = self.network_trainer.calculate_as_validation_loss()
                        print(f"  AS Validation Loss: {val_loss:.6f} (monitoring only)")
                        
        print("\n🏆 GTO Training Completed!")
        TrainingUtils.save_models(self.avg_pytorch_net, self.br_pytorch_net, self.current_episode)
        TrainingUtils.create_playable_agent()
        TrainingUtils.plot_training_progress(self)
        TrainingUtils.plot_agent_stats_evolution(self)  # Add the new plotting function
        TrainingUtils.generate_gto_report(self)
    
    def _print_training_phase_info(self, episode: int, train_best_response: bool, decision_reason: str):
        """Print clear information about current training phase and metrics."""
        
        # Determine phase (matches new evaluator schedule)
        if episode < 10:
            phase = f"  PHASE 1: Bootstrap (BR Training)"
            phase_progress = f"{episode+1}/10"
        else:
            cycle = (episode - 10) // 3 + 1
            cycle_pos = (episode - 10) % 3 + 1
            phase = f"  PHASE 2: Alternating 2BR:1AS (Cycle {cycle})"
            phase_progress = f"{cycle_pos}/3"
        
        training_type = "  BR Training" if train_best_response else "  AS Training"
        debug_indicator = " (debug)" if (episode % 5 == 0) else ""
        
        print(f"{phase} | Progress: {phase_progress}")
        print(f"{training_type}{debug_indicator} | {decision_reason}")
        
        # Show relevant metrics based on what's training
        if train_best_response:
            self._print_br_metrics()
        else:
            self._print_as_metrics()
    
    def _print_as_metrics(self):
        """Print AS-specific training metrics."""
        reservoir_size = len(self.network_trainer.reservoir_buffer)
        validation_size = len(self.network_trainer.as_validation_buffer)
        print(f"  AS Buffers: Reservoir={reservoir_size} | Validation={validation_size}")
        
        if len(self.as_training_losses) > 0:
            recent_losses = list(self.as_training_losses)[-3:]  # Last 3 only
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            print(f"  AS Performance: Avg Loss={avg_loss:.4f} (last {len(recent_losses)} episodes)")
    
    def _print_br_metrics(self):
        """Print BR-specific training metrics."""
        training_size = len(self.network_trainer.br_buffer)
        validation_size = len(self.network_trainer.br_validation_buffer)
        print(f"  BR Buffers: Training={training_size} | Validation={validation_size}")
        
        if len(self.network_trainer.br_validation_losses) > 0:
            recent_val = list(self.network_trainer.br_validation_losses)[-3:]  # Last 3 only
            avg_val = sum(recent_val) / len(recent_val) if recent_val else float('inf')
            print(f"  BR Performance: Avg Val Loss={avg_val:.4f} (last {len(recent_val)} episodes)")
    
    def _process_as_experiences(self, experiences):
        """Process AS experiences for NFSP training."""
        for exp in experiences:
            # Add to recent experiences for monitoring
            self.recent_experiences.append(exp)
            
            # 10% chance to add to validation buffer
            if random.random() < 0.1:
                self.network_trainer.add_to_as_validation_buffer(exp)
            
            # IMPORTANT: For NFSP, AS network experiences are NOT added to reservoir buffer
            # Reservoir buffer is populated by BR network actions (see _process_br_experiences)
    
    def _process_br_experiences(self, experiences):
        """Process BR experiences for NFSP training."""
        for exp in experiences:
            # FIXED: Split experiences between training and validation
            # 10% go to validation buffer (unseen data for meaningful validation)
            if random.random() < 0.1:
                self.network_trainer.add_to_br_validation_buffer(exp)
            else:
                # 90% go to training buffer
                self.network_trainer.add_to_br_buffer(exp)
            
            # CRITICAL: Add to reservoir buffer for AS network to learn from
            # This is the key insight of NFSP - AS learns to imitate BR actions
            self.network_trainer.add_to_reservoir_buffer(exp)
    
    def _collect_period_stats(self, agent_id: str):
        """Collect stats for the current reporting period from StatsTracker."""
        # Get recent stats from the main tracker
        all_stats = self.stats_tracker.get_player_percentages(agent_id)
        
        # Extract key stats for period reporting
        return {
            'hands': all_stats.get('sample_size', 0),
            'vpip': all_stats.get('vpip', 0.0),
            'pfr': all_stats.get('pfr', 0.0),
            'cbet_flop': all_stats.get('cbet_flop', 0.0),
            'fold_to_cbet_flop': all_stats.get('fold_to_cbet_flop', 0.0),
            'wtsd': all_stats.get('wtsd', 0.0),
            'aggression_frequency': all_stats.get('aggression_frequency', 0.0)
        }
    
    def _format_period_diagnostic(self, current_stats: Dict, baseline_stats: Dict, episodes: int) -> str:
        """Format diagnostic showing changes since start of reporting period."""
        if not baseline_stats:
            return f"Period diagnostic unavailable (no baseline recorded)"
        
        # Calculate the difference in hands played during this period
        hands_this_period = current_stats.get('sample_size', 0) - baseline_stats.get('sample_size', 0)
        
        if hands_this_period < 10:
            return f"Insufficient data ({hands_this_period} hands in last {episodes} episodes)"
        
        # Show current stats with changes from baseline
        def format_change(current, baseline, label):
            if baseline == 0 and current == 0:
                return f"{label}: {current:.1%} (no change)"
            elif baseline == 0:
                return f"{label}: {current:.1%} (new)"
            else:
                change = current - baseline
                change_str = f"+{change:.1%}" if change >= 0 else f"{change:.1%}"
                return f"{label}: {current:.1%} ({change_str})"
        
        # Key stats with changes
        vpip_str = format_change(current_stats.get('vpip', 0), baseline_stats.get('vpip', 0), "VPIP")
        pfr_str = format_change(current_stats.get('pfr', 0), baseline_stats.get('pfr', 0), "PFR")
        cbet_str = format_change(current_stats.get('cbet_flop', 0), baseline_stats.get('cbet_flop', 0), "C-bet")
        fold_cbet_str = format_change(current_stats.get('fold_to_cbet_flop', 0), baseline_stats.get('fold_to_cbet_flop', 0), "Fold-to-C-bet")
        wtsd_str = format_change(current_stats.get('wtsd', 0), baseline_stats.get('wtsd', 0), "WTSD")
        agg_str = format_change(current_stats.get('aggression_frequency', 0), baseline_stats.get('aggression_frequency', 0), "Aggression")
        
        return f"""📊 {hands_this_period} hands played during last {episodes} episodes
  Pre-flop: {vpip_str}, {pfr_str}
  Post-flop: {cbet_str}, {fold_cbet_str}
  General: {agg_str}, {wtsd_str}"""


if __name__ == "__main__":
    # Example usage with different loss weight configurations:
    # trainer = NFSPTrainer(loss_weight_config='balanced')     # Equal emphasis on all components
    # trainer = NFSPTrainer(loss_weight_config='action_focused') # Prioritize action selection
    # trainer = NFSPTrainer(loss_weight_config='bet_focused')    # Emphasize bet sizing
    # trainer = NFSPTrainer(loss_weight_config='conservative')   # Low exploration
    # trainer = NFSPTrainer(loss_weight_config='aggressive')     # High exploration
    
    trainer = NFSPTrainer()  # Uses 'default' configuration
    
    # Recommended training parameters:
    # - Start with 1000+ episodes for good convergence
    # - 200 hands per episode balances learning speed vs stability
    # - Can always resume with more episodes later
    trainer.train_gto(episodes=1000, hands_per_episode=200)

