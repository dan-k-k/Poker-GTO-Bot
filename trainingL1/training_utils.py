# trainingL1/training_utils.py
# Utility functions for NFSP training: I/O, plotting, diagnostics, and helpers

import torch
import numpy as np
import os
import json
import signal
import sys
import pickle
from collections import deque
from typing import Dict, List, Tuple, Optional


class TrainingUtils:
    """
    Utility functions for NFSP training.
    Includes:
        - Graceful exit
        - Load existing models
        - Save models
        - Save training state
        - Load training state
        - Create playable agent
        - Print training diagnostics
        - Plot training progress
        - Generate gto report
    """
    
    # Centralized training directory
    TRAINING_DIR = "training_output"
    
    @staticmethod
    def ensure_training_dir():
        """Create training directory if it doesn't exist."""
        if not os.path.exists(TrainingUtils.TRAINING_DIR):
            os.makedirs(TrainingUtils.TRAINING_DIR)
            print(f"📁 Created training directory: {TrainingUtils.TRAINING_DIR}")
        return TrainingUtils.TRAINING_DIR
    
    @staticmethod
    def get_training_path(filename):
        """Get full path for a training file."""
        TrainingUtils.ensure_training_dir()
        return os.path.join(TrainingUtils.TRAINING_DIR, filename)
    
    @staticmethod
    def clean_training_dir():
        """Remove all files in training directory for fresh start."""
        if os.path.exists(TrainingUtils.TRAINING_DIR):
            import shutil
            shutil.rmtree(TrainingUtils.TRAINING_DIR)
            print(f"🗑️  Cleaned training directory: {TrainingUtils.TRAINING_DIR}")
        TrainingUtils.ensure_training_dir()
    
    @staticmethod
    def setup_graceful_exit(trainer_instance):
        """Set up signal handlers for graceful exit on Ctrl+C."""
        def signal_handler(signum, frame):
            print(f"\n🛑 Training interrupted! Saving current progress...")
            trainer_instance.training_interrupted = True
            # Don't exit immediately - let the training loop handle it
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    @staticmethod
    def load_existing_models(avg_pytorch_net, br_pytorch_net):
        """Load existing PyTorch models if they exist to resume training."""
        avg_model_path = TrainingUtils.get_training_path("avg_pytorch_net.pt")
        br_model_path = TrainingUtils.get_training_path("br_pytorch_net.pt")
        
        # Try to load average strategy model
        if os.path.exists(avg_model_path):
            try:
                avg_pytorch_net.load_state_dict(torch.load(avg_model_path))
                print(f"✅ Loaded existing average strategy model")
            except Exception as e:
                print(f"⚠️  Failed to load average strategy model: {e}")
        else:
            print(f"🆕 Starting fresh average strategy model")
        
        # Try to load best response model
        if os.path.exists(br_model_path):
            try:
                br_pytorch_net.load_state_dict(torch.load(br_model_path))
                print(f"✅ Loaded existing best response model")
            except Exception as e:
                print(f"⚠️  Failed to load best response model: {e}")
        else:
            print(f"🆕 Starting fresh best response model")
    
    @staticmethod
    def save_models(avg_pytorch_net, br_pytorch_net, current_episode):
        """Save both networks to training directory."""
        avg_path = TrainingUtils.get_training_path("avg_pytorch_net.pt")
        br_path = TrainingUtils.get_training_path("br_pytorch_net.pt")
        torch.save(avg_pytorch_net.state_dict(), avg_path)
        torch.save(br_pytorch_net.state_dict(), br_path)
        print(f"💾 Saved models to {TrainingUtils.TRAINING_DIR}/ (episode {current_episode})")
    
    @staticmethod
    def save_training_state(trainer_instance):
        """Save current training state for resumption."""
        state = {
            'episode': trainer_instance.current_episode,
            'exploitability_scores': list(trainer_instance.exploitability_scores),
            'reservoir_buffer_size': len(trainer_instance.network_trainer.reservoir_buffer),
            'br_buffer_size': len(trainer_instance.network_trainer.br_buffer),
            'avg_strategy_losses': trainer_instance.avg_strategy_losses,
            'best_response_losses': trainer_instance.best_response_losses,
            'episode_numbers': trainer_instance.episode_numbers
        }
        
        state['best_exploitability'] = trainer_instance.best_exploitability
        state['patience_counter'] = trainer_instance.patience_counter
        state['as_training_losses'] = list(trainer_instance.as_training_losses)
        state['as_chip_performance'] = list(trainer_instance.as_chip_performance)
        state['br_consecutive_training'] = trainer_instance.br_consecutive_training
        
        try:
            # Save main state as JSON
            state_path = TrainingUtils.get_training_path('training_state.json')
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"✅ Saved training state to {TrainingUtils.TRAINING_DIR}/ (episode {trainer_instance.current_episode})")
            
            # Save replay buffers using pickle
            buffers = {
                'reservoir_buffer': trainer_instance.network_trainer.reservoir_buffer,
                'br_buffer': trainer_instance.network_trainer.br_buffer,
                'as_validation_buffer': trainer_instance.network_trainer.as_validation_buffer,
                'br_validation_buffer': trainer_instance.network_trainer.br_validation_buffer
            }
            buffers_path = TrainingUtils.get_training_path('training_buffers.pkl')
            with open(buffers_path, 'wb') as f:
                pickle.dump(buffers, f)
            
            # Calculate total buffer size for user info
            total_experiences = sum(len(buf) for buf in buffers.values())
            print(f"✅ Saved replay buffers ({total_experiences} total experiences)")
            
        except Exception as e:
            print(f"⚠️  Failed to save training state or buffers: {e}")
    
    @staticmethod
    def load_training_state(trainer_instance):
        """Load previous training state and buffers if available."""
        state_path = TrainingUtils.get_training_path('training_state.json')
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                trainer_instance.current_episode = state.get('episode', 0)
                exploitability_list = state.get('exploitability_scores', [])
                trainer_instance.exploitability_scores.extend(exploitability_list)
                
                # Load loss data
                trainer_instance.avg_strategy_losses = state.get('avg_strategy_losses', [])
                trainer_instance.best_response_losses = state.get('best_response_losses', [])
                trainer_instance.episode_numbers = state.get('episode_numbers', [])
                
                # Load early stopping state
                trainer_instance.best_exploitability = state.get('best_exploitability', float('inf'))
                trainer_instance.patience_counter = state.get('patience_counter', 0)
                
                # Load adaptive training state
                saved_losses = state.get('as_training_losses', [])
                saved_chip_perf = state.get('as_chip_performance', [])
                trainer_instance.as_training_losses.extend(saved_losses)
                trainer_instance.as_chip_performance.extend(saved_chip_perf)
                trainer_instance.br_consecutive_training = state.get('br_consecutive_training', 0)
                
                print(f"✅ Resumed from episode {trainer_instance.current_episode}")
                print(f"📊 Loaded {len(trainer_instance.avg_strategy_losses)} loss data points")
                
                # Try to load replay buffers
                buffers_path = TrainingUtils.get_training_path('training_buffers.pkl')
                if os.path.exists(buffers_path):
                    try:
                        with open(buffers_path, 'rb') as f:
                            buffers = pickle.load(f)
                        
                        from collections import deque
                        trainer_instance.network_trainer.reservoir_buffer = buffers.get('reservoir_buffer', deque(maxlen=200000))
                        trainer_instance.network_trainer.br_buffer = buffers.get('br_buffer', deque(maxlen=50000))  
                        trainer_instance.network_trainer.as_validation_buffer = buffers.get('as_validation_buffer', deque(maxlen=5000))
                        trainer_instance.network_trainer.br_validation_buffer = buffers.get('br_validation_buffer', deque(maxlen=5000))
                        
                        total_experiences = sum(len(buf) for buf in buffers.values())
                        print(f"✅ Loaded replay buffers ({total_experiences} total experiences)")
                        print(f"   Reservoir: {len(trainer_instance.network_trainer.reservoir_buffer)}")
                        print(f"   BR Training: {len(trainer_instance.network_trainer.br_buffer)}")
                        print(f"   AS Validation: {len(trainer_instance.network_trainer.as_validation_buffer)}")
                        print(f"   BR Validation: {len(trainer_instance.network_trainer.br_validation_buffer)}")
                        
                    except Exception as e:
                        print(f"⚠️  Failed to load replay buffers: {e}")
                        print("   Training will start with empty buffers")
                else:
                    print("📋 No replay buffers found - starting with empty buffers")
                
                return trainer_instance.current_episode
            except Exception as e:
                print(f"⚠️  Failed to load training state: {e}")
        
        return 0
    
    @staticmethod
    def create_playable_agent():
        """Create a playable agent file for use in playgame.py"""
        print("🎮 Creating playable agent files...")
        
        # Models are ready to use as PyTorch .pt files
        avg_path = TrainingUtils.get_training_path("avg_pytorch_net.pt")
        br_path = TrainingUtils.get_training_path("br_pytorch_net.pt")
        
        print("✅ GTO models ready for use:")
        print(f"   📄 Average strategy: {avg_path}")
        print(f"   📄 Best response: {br_path}")
        print(f"   📝 Usage: GTOAgent(seat_id=0, model_path='{avg_path}')")
    
    @staticmethod
    def print_training_diagnostics(training_stats):
        """Print detailed training diagnostics."""
        # print("\n" + "="*50)
        # print("📈 TRAINING DIAGNOSTICS")
        # print("="*50)
        
        if len(training_stats['all_in_frequency']) > 0:
            all_in_freq = sum(training_stats['all_in_frequency']) / len(training_stats['all_in_frequency'])
            fold_freq = sum(training_stats['fold_frequency']) / len(training_stats['fold_frequency'])
            avg_hand_len = sum(training_stats['avg_hand_length']) / len(training_stats['avg_hand_length'])
            showdown_freq = sum(training_stats['showdown_frequency']) / len(training_stats['showdown_frequency'])
            bet_raise_freq = sum(training_stats['bet_raise_frequency']) / len(training_stats['bet_raise_frequency'])
            vpip = sum(training_stats['vpip']) / len(training_stats['vpip'])
            pfr = sum(training_stats['pfr']) / len(training_stats['pfr'])
            
            print(f"🎯 All-in Frequency: {all_in_freq:.1%}")
            print(f"📉 Fold Frequency: {fold_freq:.1%}")
            print(f"🎲 Avg Actions/Hand: {avg_hand_len:.1f}")
            print(f"🏁 Showdown Frequency: {showdown_freq:.1%}")
            print(f"📈 Bet/Raise Frequency: {bet_raise_freq:.1%}")
            print(f"💰 VPIP (Pre-flop Play): {vpip:.1%}")
            print(f"🔥 PFR (Pre-flop Raise): {pfr:.1%}")
            
            # Provide interpretations
            if all_in_freq > 0.15:
                print("⚠️  High all-in frequency > 0.15 - agents may be too aggressive")
            elif all_in_freq < 0.02:
                print("✅ Low all-in frequency < 0.02 - good strategic balance")
                
            if fold_freq > 0.4:
                print("⚠️  High fold frequency > 0.4 - agents may be too tight")
            elif fold_freq < 0.1:
                print("⚠️  Low fold frequency < 0.1 - agents may not be folding enough")
            else:
                print("✅ Reasonable fold frequency 0.10 < f < 0.40")
                
            if avg_hand_len < 3:
                print("⚠️  Very short hands < 3 - likely many quick folds or all-ins")
            elif avg_hand_len > 8:
                print("✅ Longer hands > 8 - good strategic depth")
            
            # New interpretations for advanced metrics
            if bet_raise_freq > 0.35:
                print("⚠️  Very high aggression > 35% - may be over-aggressive")
            elif bet_raise_freq < 0.15:
                print("⚠️  Low aggression < 15% - may be too passive")
            else:
                print("✅ Good aggression balance 15-35%")
                
            if vpip > 0.85:
                print("⚠️  Very loose VPIP > 85% - playing too many hands")
            elif vpip < 0.50:
                print("⚠️  Too tight VPIP < 50% - folding too much heads-up")
            else:
                print("✅ Good heads-up VPIP range 50-85%")
                
            # PFR/VPIP ratio analysis
            if vpip > 0 and pfr > 0:
                pfr_ratio = pfr / vpip
                if pfr_ratio > 0.8:
                    print("✅ Aggressive pre-flop strategy (PFR/VPIP > 80%)")
                elif pfr_ratio < 0.4:
                    print("⚠️  Passive pre-flop - calling too much (PFR/VPIP < 40%)")
                else:
                    print("✅ Balanced pre-flop aggression")
                
        print("="*50)
        
    @staticmethod
    def plot_training_progress(trainer_instance):
        """Generate loss plots and save training data for external plotting."""
        print("📈 Generating training progress plots...")
        
        # Save raw data for external plotting
        plot_data = {
            'episode_numbers': trainer_instance.episode_numbers,
            'avg_strategy_losses': trainer_instance.avg_strategy_losses,
            'best_response_losses': trainer_instance.best_response_losses,
            'exploitability_scores': list(trainer_instance.exploitability_scores)
        }
        
        try:
            import pickle
            plots_path = TrainingUtils.get_training_path('training_plots_data.pkl')
            with open(plots_path, 'wb') as f:
                pickle.dump(plot_data, f)
            print(f"✅ Saved plot data to {plots_path}")
        except Exception as e:
            print(f"⚠️  Failed to save plot data: {e}")
        
        # Try to create matplotlib plots if available
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Training Losses
            episodes = np.array(trainer_instance.episode_numbers)
            avg_losses = np.array(trainer_instance.avg_strategy_losses)
            br_losses = np.array(trainer_instance.best_response_losses)
            
            # Filter out None values for plotting
            avg_mask = avg_losses != None
            br_mask = br_losses != None
            
            if np.any(avg_mask):
                avg_episodes = episodes[avg_mask]
                avg_clean = avg_losses[avg_mask].astype(float)
                ax1.plot(avg_episodes, avg_clean, 'b-', label='Average Strategy Loss', alpha=0.7)
                
            if np.any(br_mask):
                br_episodes = episodes[br_mask]
                br_clean = br_losses[br_mask].astype(float)
                ax1.plot(br_episodes, br_clean, 'r-', label='Best Response Loss', alpha=0.7)
            
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Training Loss')
            ax1.set_title('Neural Poker Training - Loss Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Exploitability
            if len(trainer_instance.exploitability_scores) > 1:
                exploit_episodes = np.arange(0, len(trainer_instance.exploitability_scores) * 20, 20)[:len(trainer_instance.exploitability_scores)]
                ax2.plot(exploit_episodes, list(trainer_instance.exploitability_scores), 'g-', marker='o', label='Exploitability')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Exploitability Score')
                ax2.set_title('GTO Convergence - Lower is Better')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            progress_path = TrainingUtils.get_training_path('training_progress.png')
            plt.savefig(progress_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved training plots to {progress_path}")
            
            # Also save individual plots
            plt.figure(figsize=(10, 6))
            if np.any(avg_mask):
                plt.plot(avg_episodes, avg_clean, 'b-', label='Average Strategy Loss', linewidth=2)
            if np.any(br_mask):
                plt.plot(br_episodes, br_clean, 'r-', label='Best Response Loss', linewidth=2)
            plt.xlabel('Episode')
            plt.ylabel('Training Loss')
            plt.title('Neural Poker Training - Loss Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            curves_path = TrainingUtils.get_training_path('loss_curves.png')
            plt.savefig(curves_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved loss curves to {curves_path}")
            
        except ImportError:
            print("📊 matplotlib not available - saved data for external plotting")
        except Exception as e:
            print(f"⚠️  Failed to create plots: {e}")
    
    @staticmethod
    def plot_agent_stats_evolution(trainer_instance):
        """Generate plots showing how agent strategies evolved over time."""
        print("📈 Generating agent strategy evolution plots...")
        
        # Check if we have enough data
        if not trainer_instance.as_stat_history and not trainer_instance.br_stat_history:
            print("   ⚠️ No agent stat history available for plotting.")
            return
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create figure with more subplots for comprehensive analysis
            fig, axes = plt.subplots(3, 3, figsize=(20, 15))
            fig.suptitle('Agent Strategy Evolution & GTO Convergence Analysis', fontsize=16, fontweight='bold')
            
            # Helper function to extract data safely
            def extract_data(history, key):
                episodes = []
                values = []
                for entry in history:
                    if key in entry and 'episode' in entry:
                        episodes.append(entry['episode'])
                        values.append(entry[key])
                return np.array(episodes), np.array(values)
            
            # Plot 1: VPIP Evolution
            if trainer_instance.as_stat_history:
                as_episodes, as_vpip = extract_data(trainer_instance.as_stat_history, 'vpip')
                if len(as_vpip) > 0:
                    axes[0, 0].plot(as_episodes, as_vpip * 100, 'b-', marker='o', label='AS VPIP', linewidth=2, markersize=4)
            
            if trainer_instance.br_stat_history:
                br_episodes, br_vpip = extract_data(trainer_instance.br_stat_history, 'vpip')
                if len(br_vpip) > 0:
                    axes[0, 0].plot(br_episodes, br_vpip * 100, 'r-', marker='s', label='BR VPIP', linewidth=2, markersize=4)
            
            axes[0, 0].set_title('Voluntarily Put In Pot % (VPIP)', fontweight='bold')
            axes[0, 0].set_ylabel('VPIP %')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Tight/Loose Boundary')
            
            # Plot 2: PFR Evolution
            if trainer_instance.as_stat_history:
                as_episodes, as_pfr = extract_data(trainer_instance.as_stat_history, 'pfr')
                if len(as_pfr) > 0:
                    axes[0, 1].plot(as_episodes, as_pfr * 100, 'b-', marker='o', label='AS PFR', linewidth=2, markersize=4)
            
            if trainer_instance.br_stat_history:
                br_episodes, br_pfr = extract_data(trainer_instance.br_stat_history, 'pfr')
                if len(br_pfr) > 0:
                    axes[0, 1].plot(br_episodes, br_pfr * 100, 'r-', marker='s', label='BR PFR', linewidth=2, markersize=4)
            
            axes[0, 1].set_title('Pre-Flop Raise % (PFR)', fontweight='bold')
            axes[0, 1].set_ylabel('PFR %')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: C-bet Evolution
            if trainer_instance.as_stat_history:
                as_episodes, as_cbet = extract_data(trainer_instance.as_stat_history, 'cbet_flop')
                if len(as_cbet) > 0:
                    axes[0, 2].plot(as_episodes, as_cbet * 100, 'b-', marker='o', label='AS C-bet', linewidth=2, markersize=4)
            
            if trainer_instance.br_stat_history:
                br_episodes, br_cbet = extract_data(trainer_instance.br_stat_history, 'cbet_flop')
                if len(br_cbet) > 0:
                    axes[0, 2].plot(br_episodes, br_cbet * 100, 'r-', marker='s', label='BR C-bet', linewidth=2, markersize=4)
            
            axes[0, 2].set_title('Continuation Bet % (Flop)', fontweight='bold')
            axes[0, 2].set_ylabel('C-bet %')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # Plot 4: Fold to C-bet Evolution
            if trainer_instance.as_stat_history:
                as_episodes, as_fold_cbet = extract_data(trainer_instance.as_stat_history, 'fold_to_cbet_flop')
                if len(as_fold_cbet) > 0:
                    axes[1, 0].plot(as_episodes, as_fold_cbet * 100, 'b-', marker='o', label='AS Fold to C-bet', linewidth=2, markersize=4)
            
            if trainer_instance.br_stat_history:
                br_episodes, br_fold_cbet = extract_data(trainer_instance.br_stat_history, 'fold_to_cbet_flop')
                if len(br_fold_cbet) > 0:
                    axes[1, 0].plot(br_episodes, br_fold_cbet * 100, 'r-', marker='s', label='BR Fold to C-bet', linewidth=2, markersize=4)
            
            axes[1, 0].set_title('Fold to C-bet % (Flop)', fontweight='bold')
            axes[1, 0].set_ylabel('Fold to C-bet %')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: WTSD Evolution
            if trainer_instance.as_stat_history:
                as_episodes, as_wtsd = extract_data(trainer_instance.as_stat_history, 'wtsd')
                if len(as_wtsd) > 0:
                    axes[1, 1].plot(as_episodes, as_wtsd * 100, 'b-', marker='o', label='AS WTSD', linewidth=2, markersize=4)
            
            if trainer_instance.br_stat_history:
                br_episodes, br_wtsd = extract_data(trainer_instance.br_stat_history, 'wtsd')
                if len(br_wtsd) > 0:
                    axes[1, 1].plot(br_episodes, br_wtsd * 100, 'r-', marker='s', label='BR WTSD', linewidth=2, markersize=4)
            
            axes[1, 1].set_title('Went to Showdown %', fontweight='bold')
            axes[1, 1].set_ylabel('WTSD %')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Aggression Evolution
            if trainer_instance.as_stat_history:
                as_episodes, as_agg = extract_data(trainer_instance.as_stat_history, 'aggression_frequency')
                if len(as_agg) > 0:
                    axes[1, 2].plot(as_episodes, as_agg * 100, 'b-', marker='o', label='AS Aggression', linewidth=2, markersize=4)
            
            if trainer_instance.br_stat_history:
                br_episodes, br_agg = extract_data(trainer_instance.br_stat_history, 'aggression_frequency')
                if len(br_agg) > 0:
                    axes[1, 2].plot(br_episodes, br_agg * 100, 'r-', marker='s', label='BR Aggression', linewidth=2, markersize=4)
            
            axes[1, 2].set_title('Post-Flop Aggression %', fontweight='bold')
            axes[1, 2].set_ylabel('Aggression %')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            # Plot 7: 3-Bet Frequency (GTO Convergence Indicator)
            if trainer_instance.as_stat_history:
                as_episodes, as_3bet = extract_data(trainer_instance.as_stat_history, 'three_bet')
                if len(as_3bet) > 0:
                    axes[2, 0].plot(as_episodes, as_3bet * 100, 'b-', marker='o', label='AS 3-Bet', linewidth=2, markersize=4)
            
            if trainer_instance.br_stat_history:
                br_episodes, br_3bet = extract_data(trainer_instance.br_stat_history, 'three_bet')
                if len(br_3bet) > 0:
                    axes[2, 0].plot(br_episodes, br_3bet * 100, 'r-', marker='s', label='BR 3-Bet', linewidth=2, markersize=4)
            
            axes[2, 0].set_title('3-Bet Frequency % (GTO Balance)', fontweight='bold')
            axes[2, 0].set_ylabel('3-Bet %')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].axhline(y=8, color='green', linestyle='--', alpha=0.7, label='GTO Range ~8%')
            
            # Plot 8: Showdown Win Rate (Hand Strength Indicator)
            if trainer_instance.as_stat_history:
                as_episodes, as_showdown_wr = extract_data(trainer_instance.as_stat_history, 'showdown_win_rate')
                if len(as_showdown_wr) > 0:
                    axes[2, 1].plot(as_episodes, as_showdown_wr * 100, 'b-', marker='o', label='AS Showdown WR', linewidth=2, markersize=4)
            
            if trainer_instance.br_stat_history:
                br_episodes, br_showdown_wr = extract_data(trainer_instance.br_stat_history, 'showdown_win_rate')
                if len(br_showdown_wr) > 0:
                    axes[2, 1].plot(br_episodes, br_showdown_wr * 100, 'r-', marker='s', label='BR Showdown WR', linewidth=2, markersize=4)
            
            axes[2, 1].set_title('Showdown Win Rate % (Hand Selection)', fontweight='bold')
            axes[2, 1].set_ylabel('Showdown Win %')
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random=50%')
            
            # Plot 9: PFR/VPIP Ratio (Aggression Balance)
            def calculate_pfr_vpip_ratio(history):
                episodes = []
                ratios = []
                for entry in history:
                    if 'pfr' in entry and 'vpip' in entry and 'episode' in entry:
                        vpip = entry['vpip']
                        pfr = entry['pfr']
                        if vpip > 0:  # Avoid division by zero
                            episodes.append(entry['episode'])
                            ratios.append(pfr / vpip)
                return np.array(episodes), np.array(ratios)
            
            if trainer_instance.as_stat_history:
                as_episodes, as_ratio = calculate_pfr_vpip_ratio(trainer_instance.as_stat_history)
                if len(as_ratio) > 0:
                    axes[2, 2].plot(as_episodes, as_ratio * 100, 'b-', marker='o', label='AS PFR/VPIP', linewidth=2, markersize=4)
            
            if trainer_instance.br_stat_history:
                br_episodes, br_ratio = calculate_pfr_vpip_ratio(trainer_instance.br_stat_history)
                if len(br_ratio) > 0:
                    axes[2, 2].plot(br_episodes, br_ratio * 100, 'r-', marker='s', label='BR PFR/VPIP', linewidth=2, markersize=4)
            
            axes[2, 2].set_title('PFR/VPIP Ratio % (Aggression Balance)', fontweight='bold')
            axes[2, 2].set_ylabel('PFR/VPIP Ratio %')
            axes[2, 2].set_xlabel('Episode')
            axes[2, 2].legend()
            axes[2, 2].grid(True, alpha=0.3)
            axes[2, 2].axhline(y=60, color='green', linestyle='--', alpha=0.7, label='GTO Range ~60%')
            axes[2, 2].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Very Aggressive')
            axes[2, 2].axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Too Passive')
            
            # Add a subtitle with interpretation
            fig.text(0.5, 0.02, 'Blue = Average Strategy (AS) | Red = Best Response (BR) | Shows how agent strategies evolve during training', 
                    ha='center', va='bottom', fontsize=11, style='italic')
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.96])
            
            # Save the plot
            stats_plot_path = TrainingUtils.get_training_path('agent_stats_evolution.png')
            plt.savefig(stats_plot_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved agent stats evolution plot to {stats_plot_path}")
            
            # Save raw data for external analysis
            stats_data = {
                'as_stat_history': trainer_instance.as_stat_history,
                'br_stat_history': trainer_instance.br_stat_history
            }
            stats_data_path = TrainingUtils.get_training_path('agent_stats_data.pkl')
            with open(stats_data_path, 'wb') as f:
                pickle.dump(stats_data, f)
            print(f"✅ Saved agent stats data to {stats_data_path}")
            
        except ImportError:
            print("📊 matplotlib not available - saved data for external plotting")
        except Exception as e:
            print(f"⚠️  Failed to create agent stats plots: {e}")
    
    @staticmethod
    def generate_gto_report(trainer_instance):
        """Generate training report focused on GTO metrics."""
        print("\n" + "="*60)
        print("GTO TRAINING REPORT")
        print("="*60)
        
        if len(trainer_instance.exploitability_scores) > 0:
            final_exploit = trainer_instance.exploitability_scores[-1]
            best_exploit = min(trainer_instance.exploitability_scores)
            print(f"Final Exploitability: {final_exploit:.4f}")
            print(f"Best Exploitability: {best_exploit:.4f}")
            print(f"Improvement: {(trainer_instance.exploitability_scores[0] - final_exploit):.4f}")
        
        print(f"Reservoir Buffer Size: {len(trainer_instance.network_trainer.reservoir_buffer)}")
        print(f"Best Response Buffer Size: {len(trainer_instance.network_trainer.br_buffer)}")
        
        # Add stack depth simulation report
        if hasattr(trainer_instance, 'stack_depth_simulator') and trainer_instance.stack_depth_simulator:
            print(f"\n📊 Stack Depth Simulation Summary:")
            stats = trainer_instance.stack_depth_simulator.get_stack_depth_statistics()
            if stats:
                print(f"   Total Sessions: {stats['total_sessions']}")
                print(f"   Avg Effective Stack: {stats['avg_effective_stack']:.1f} BB")
                print(f"   Stack Range: {stats['min_effective_stack']:.1f} - {stats['max_effective_stack']:.1f} BB")
                
                # Stack depth distribution
                distribution = stats.get('stack_depth_distribution', {})
                for category, data in distribution.items():
                    print(f"   {category.capitalize()}: {data['percentage']:.1f}%")
        
        print("="*60)

