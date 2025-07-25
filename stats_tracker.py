# stats_tracker.py
# Comprehensive poker statistics tracking for opponent modeling

import json
import os
from collections import defaultdict, deque
from typing import Dict, Optional, List

class StatsTracker:
    """
    Comprehensive poker statistics tracker for both Layer 1 (GTO) and Layer 2 (Exploitative) training.
    
    Tracks detailed opponent tendencies to enable:
    1. Better range construction for equity calculations (Layer 1)
    2. Exploitative strategy development (Layer 2)
    """
    
    def __init__(self, filepath='training_output/player_stats.json', window_size=5000):
        self.filepath = filepath
        self.window_size = window_size
        self.stats = self._load_stats()
        
        # Temporary storage for current hand data (legacy - will be removed)
        self.current_hand_data = {}
    
    def update_from_events(self, player_id: str, events: Dict[str, bool]):
        """
        Updates deques from a dictionary of named boolean events.
        This is the new, simplified way to update stats that avoids the 'opportunity flaw'.
        
        Args:
            player_id: String identifier for the player (e.g., "average_strategy_v1")
            events: Dict of event_name -> bool (e.g., {'vpip_opportunity': True, 'vpip_action': True})
        """
        stats = self.stats[player_id]
        stats['total_hands'] += 1
        stats['last_updated_episode'] = getattr(self, 'current_episode', 0)
        
        # Pre-flop stats - only append when opportunity exists
        if 'vpip_opportunity' in events:
            stats['vpip_history'].append(1 if events.get('vpip_action', False) else 0)
        
        if 'pfr_opportunity' in events:
            stats['pfr_history'].append(1 if events.get('pfr_action', False) else 0)
        
        if 'three_bet_opportunity' in events:
            stats['three_bet_history'].append(1 if events.get('three_bet_action', False) else 0)
        
        if 'fold_to_three_bet_opportunity' in events:
            stats['fold_to_three_bet_history'].append(1 if events.get('fold_to_three_bet_action', False) else 0)
        
        if 'limp_opportunity' in events:
            stats['limp_history'].append(1 if events.get('limp_action', False) else 0)
        
        # Post-flop stats - only append when opportunity exists
        if 'cbet_flop_opportunity' in events:
            stats['cbet_flop_history'].append(1 if events.get('cbet_flop_action', False) else 0)
        
        if 'cbet_turn_opportunity' in events:
            stats['cbet_turn_history'].append(1 if events.get('cbet_turn_action', False) else 0)
        
        if 'cbet_river_opportunity' in events:
            stats['cbet_river_history'].append(1 if events.get('cbet_river_action', False) else 0)
        
        if 'fold_to_cbet_flop_opportunity' in events:
            stats['fold_to_cbet_flop_history'].append(1 if events.get('fold_to_cbet_flop_action', False) else 0)
        
        if 'fold_to_cbet_turn_opportunity' in events:
            stats['fold_to_cbet_turn_history'].append(1 if events.get('fold_to_cbet_turn_action', False) else 0)
        
        if 'fold_to_cbet_river_opportunity' in events:
            stats['fold_to_cbet_river_history'].append(1 if events.get('fold_to_cbet_river_action', False) else 0)
        
        # Advanced stats
        if 'checkraise_flop_opportunity' in events:
            stats['checkraise_flop_history'].append(1 if events.get('checkraise_flop_action', False) else 0)
        
        if 'checkraise_turn_opportunity' in events:
            stats['checkraise_turn_history'].append(1 if events.get('checkraise_turn_action', False) else 0)
        
        if 'float_opportunity' in events:
            stats['float_history'].append(1 if events.get('float_action', False) else 0)
        
        # General tendencies - these are per-hand events
        if 'had_postflop_action' in events:
            stats['aggression_frequency_history'].append(1 if events.get('was_aggressive_postflop', False) else 0)
            stats['fold_frequency_history'].append(1 if events.get('folded_postflop', False) else 0)
        
        if 'had_street_action' in events:
            if events.get('preflop_action', False):
                stats['preflop_fold_history'].append(1 if events.get('folded_preflop', False) else 0)
            if events.get('flop_action', False):
                stats['flop_fold_history'].append(1 if events.get('folded_flop', False) else 0)
            if events.get('turn_action', False):
                stats['turn_fold_history'].append(1 if events.get('folded_turn', False) else 0)
            if events.get('river_action', False):
                stats['river_fold_history'].append(1 if events.get('folded_river', False) else 0)
        
        # Showdown stats
        if 'showdown_opportunity' in events:
            stats['wtsd_history'].append(1 if events.get('went_to_showdown', False) else 0)
        
        if 'went_to_showdown' in events and events['went_to_showdown']:
            stats['showdown_win_history'].append(1 if events.get('won_showdown', False) else 0)
        
        # All-in and bet sizing
        if 'all_in_opportunity' in events:
            stats['all_in_history'].append(1 if events.get('went_all_in', False) else 0)
        
        if 'bet_sizes' in events and events['bet_sizes']:
            for bet_size in events['bet_sizes']:
                stats['bet_sizes'].append(bet_size)
        
        if 'pot_ratios' in events and events['pot_ratios']:
            for pot_ratio in events['pot_ratios']:
                stats['pot_ratios'].append(pot_ratio)
    
    def _load_stats(self) -> defaultdict:
        """Load existing stats and convert lists back to deques."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    loaded_data = json.load(f)
                    # Convert lists back to deques with the correct maxlen
                    for player_id, stats in loaded_data.items():
                        for key, value in stats.items():
                            if isinstance(value, list) and key.endswith('_history'):
                                stats[key] = deque(value, maxlen=self.window_size)
                            elif key in ['bet_sizes', 'pot_ratios'] and isinstance(value, list):
                                stats[key] = deque(value, maxlen=self.window_size)
                    return defaultdict(self._get_default_stats, loaded_data)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return defaultdict(self._get_default_stats)
    
    def _get_default_stats(self) -> dict:
        """Default stats structure using sliding window deques for accurate recent player modeling."""
        return {
            # Meta stats
            "total_hands": 0,
            "last_updated_episode": 0,
            
            # Pre-flop stats (sliding window histories: 1=action taken, 0=no action)
            "vpip_history": deque(maxlen=self.window_size),
            "pfr_history": deque(maxlen=self.window_size),
            "three_bet_history": deque(maxlen=self.window_size),
            "fold_to_three_bet_history": deque(maxlen=self.window_size),
            "limp_history": deque(maxlen=self.window_size),
            
            # Post-flop aggression stats (sliding window)
            "cbet_flop_history": deque(maxlen=self.window_size),
            "cbet_turn_history": deque(maxlen=self.window_size),
            "cbet_river_history": deque(maxlen=self.window_size),
            
            # Post-flop defense stats (sliding window)
            "fold_to_cbet_flop_history": deque(maxlen=self.window_size),
            "fold_to_cbet_turn_history": deque(maxlen=self.window_size),
            "fold_to_cbet_river_history": deque(maxlen=self.window_size),
            
            # Advanced post-flop stats (sliding window)
            "checkraise_flop_history": deque(maxlen=self.window_size),
            "checkraise_turn_history": deque(maxlen=self.window_size),
            "float_history": deque(maxlen=self.window_size),  # Call flop, bet turn
            
            # Overall tendencies (sliding window)
            "aggression_frequency_history": deque(maxlen=self.window_size),  # 1=aggressive action, 0=passive
            "fold_frequency_history": deque(maxlen=self.window_size),  # 1=folded, 0=didn't fold
            "all_in_history": deque(maxlen=self.window_size),  # 1=went all-in, 0=didn't
            
            # Showdown stats (sliding window)
            "wtsd_history": deque(maxlen=self.window_size),  # 1=went to showdown, 0=didn't
            "showdown_win_history": deque(maxlen=self.window_size),  # 1=won at showdown, 0=lost (only when at showdown)
            
            # Bet sizing patterns (sliding window for exploitative layer)
            "bet_sizes": deque(maxlen=self.window_size),  # Store recent bet sizes
            "pot_ratios": deque(maxlen=self.window_size),  # Store bet-to-pot ratios
            
            # Street-specific folding (sliding window)
            "preflop_fold_history": deque(maxlen=self.window_size),  # 1=folded preflop, 0=didn't
            "flop_fold_history": deque(maxlen=self.window_size),
            "turn_fold_history": deque(maxlen=self.window_size),
            "river_fold_history": deque(maxlen=self.window_size),
        }
    
    def start_hand(self, players: List[int], episode: int):
        """Initialize tracking for a new hand."""
        self.current_hand_data = {
            'episode': episode,
            'players': players,
            'preflop_aggressor': None,
            'flop_aggressor': None,
            'turn_aggressor': None,
            'river_aggressor': None,
            'went_to_showdown': [],
            'showdown_winners': [],
            'player_actions': {p: [] for p in players},
            'street_actions': {p: {'preflop': [], 'flop': [], 'turn': [], 'river': []} for p in players}
        }
    
    def record_action(self, player_id: int, action: int, amount: Optional[int], 
                     stage: int, pot_size: int, was_facing_bet: bool = False,
                     position: str = None):
        """
        Record a single action during a hand.
        
        Args:
            player_id: The player taking the action
            action: 0=fold, 1=call/check, 2=bet/raise
            amount: Chips added (None for fold/check)
            stage: 0=preflop, 1=flop, 2=turn, 3=river
            pot_size: Current pot size
            was_facing_bet: True if player was facing a bet/raise
            position: Player's position (for future positional analysis)
        """
        if player_id not in self.current_hand_data['player_actions']:
            return
        
        # Record the action
        action_data = {
            'action': action,
            'amount': amount,
            'stage': stage,
            'pot_size': pot_size,
            'was_facing_bet': was_facing_bet,
            'position': position
        }
        
        self.current_hand_data['player_actions'][player_id].append(action_data)
        
        # Add to street-specific tracking
        street_names = ['preflop', 'flop', 'turn', 'river']
        if stage < len(street_names):
            street = street_names[stage]
            self.current_hand_data['street_actions'][player_id][street].append(action_data)
        
        # Track aggressors by street
        if action == 2:  # Bet/raise
            if stage == 0:
                self.current_hand_data['preflop_aggressor'] = player_id
            elif stage == 1:
                self.current_hand_data['flop_aggressor'] = player_id
            elif stage == 2:
                self.current_hand_data['turn_aggressor'] = player_id
            elif stage == 3:
                self.current_hand_data['river_aggressor'] = player_id
    
    def finish_hand(self, final_stage: int, winners: List[int]):
        """Process the completed hand and update statistics."""
        self.current_hand_data['final_stage'] = final_stage
        self.current_hand_data['showdown_winners'] = winners if final_stage >= 3 else []
        
        # Determine who went to showdown
        if final_stage >= 3:  # Reached river or beyond
            for player_id in self.current_hand_data['players']:
                if len(self.current_hand_data['street_actions'][player_id]['river']) > 0:
                    self.current_hand_data['went_to_showdown'].append(player_id)
        
        # Update stats for each player
        for player_id in self.current_hand_data['players']:
            self._update_player_stats(player_id)
        
        # Clear current hand data
        self.current_hand_data = {}
    
    def _update_player_stats(self, player_id: int):
        """Update all statistics for a single player based on current hand."""
        player_str = str(player_id)
        stats = self.stats[player_str]
        
        # Update meta stats
        stats['total_hands'] += 1
        stats['last_updated_episode'] = self.current_hand_data['episode']
        
        preflop_actions = self.current_hand_data['street_actions'][player_id]['preflop']
        flop_actions = self.current_hand_data['street_actions'][player_id]['flop']
        turn_actions = self.current_hand_data['street_actions'][player_id]['turn']
        river_actions = self.current_hand_data['street_actions'][player_id]['river']
        
        # Update pre-flop stats
        self._update_preflop_stats(stats, preflop_actions, player_id)
        
        # Update post-flop stats
        self._update_postflop_stats(stats, flop_actions, turn_actions, river_actions, player_id)
        
        # Update showdown stats
        self._update_showdown_stats(stats, player_id)
        
        # Update overall tendencies
        self._update_general_stats(stats, player_id)
    
    def _update_preflop_stats(self, stats: dict, preflop_actions: List[dict], player_id: int):
        """Update pre-flop specific statistics using sliding window approach."""
        if not preflop_actions:
            # If no preflop actions, append 0s for all opportunities
            stats['vpip_history'].append(0)
            stats['pfr_history'].append(0)
            stats['three_bet_history'].append(0)
            stats['fold_to_three_bet_history'].append(0)
            stats['limp_history'].append(0)
            stats['preflop_fold_history'].append(0)
            return
        
        # VPIP: Did player voluntarily put money in pot preflop?
        voluntarily_invested = any(action['action'] in [1, 2] and action['amount'] and action['amount'] > 0 
                                 for action in preflop_actions)
        stats['vpip_history'].append(1 if voluntarily_invested else 0)
        
        # PFR: Did player raise preflop?
        did_raise = any(action['action'] == 2 for action in preflop_actions)
        stats['pfr_history'].append(1 if did_raise else 0)
        
        # 3-bet: Did player make a second raise preflop?
        preflop_raises = [action for action in preflop_actions if action['action'] == 2]
        did_3bet = False
        if len(preflop_raises) > 0:
            # Check if this was a 3-bet (player raised after someone else raised)
            if self.current_hand_data['preflop_aggressor'] != player_id:
                did_3bet = len(preflop_raises) > 1 or (len(preflop_raises) == 1 and preflop_raises[0]['was_facing_bet'])
        stats['three_bet_history'].append(1 if did_3bet else 0)
        
        # Fold to 3-bet: Did player fold when facing a 3-bet?
        faced_3bet = any(action['was_facing_bet'] and action['action'] == 2 for action in preflop_actions)
        folded_to_3bet = faced_3bet and any(action['action'] == 0 for action in preflop_actions)
        stats['fold_to_three_bet_history'].append(1 if folded_to_3bet else 0)
        
        # Limp: Did player just call preflop without raising?
        limped = any(action['action'] == 1 and not action['was_facing_bet'] for action in preflop_actions)
        stats['limp_history'].append(1 if limped else 0)
        
        # Preflop fold: Did player fold preflop?
        folded_preflop = any(action['action'] == 0 for action in preflop_actions)
        stats['preflop_fold_history'].append(1 if folded_preflop else 0)
    
    def _update_postflop_stats(self, stats: dict, flop_actions: List[dict], 
                              turn_actions: List[dict], river_actions: List[dict], player_id: int):
        """Update post-flop statistics using sliding window approach."""
        
        # C-bet opportunities and actions
        was_preflop_aggressor = (self.current_hand_data['preflop_aggressor'] == player_id)
        
        # Flop C-bet
        if was_preflop_aggressor and flop_actions:
            did_cbet_flop = any(action['action'] == 2 for action in flop_actions)
            stats['cbet_flop_history'].append(1 if did_cbet_flop else 0)
        else:
            stats['cbet_flop_history'].append(0)  # No opportunity = 0
        
        # Turn C-bet
        was_flop_aggressor = (self.current_hand_data['flop_aggressor'] == player_id)
        if was_flop_aggressor and turn_actions:
            did_cbet_turn = any(action['action'] == 2 for action in turn_actions)
            stats['cbet_turn_history'].append(1 if did_cbet_turn else 0)
        else:
            stats['cbet_turn_history'].append(0)
        
        # River C-bet
        was_turn_aggressor = (self.current_hand_data['turn_aggressor'] == player_id)
        if was_turn_aggressor and river_actions:
            did_cbet_river = any(action['action'] == 2 for action in river_actions)
            stats['cbet_river_history'].append(1 if did_cbet_river else 0)
        else:
            stats['cbet_river_history'].append(0)
        
        # Fold to C-bet stats (simplified)
        self._update_fold_to_cbet_stats(stats, flop_actions, turn_actions, river_actions)
        
        # Overall post-flop aggression
        all_postflop_actions = flop_actions + turn_actions + river_actions
        was_aggressive = any(action['action'] == 2 for action in all_postflop_actions)
        stats['aggression_frequency_history'].append(1 if was_aggressive else 0)
        
        # Overall fold frequency 
        did_fold = any(action['action'] == 0 for action in all_postflop_actions)
        stats['fold_frequency_history'].append(1 if did_fold else 0)
        
        # Street-specific folding
        flop_folded = any(action['action'] == 0 for action in flop_actions)
        stats['flop_fold_history'].append(1 if flop_folded else 0)
        
        turn_folded = any(action['action'] == 0 for action in turn_actions)  
        stats['turn_fold_history'].append(1 if turn_folded else 0)
        
        river_folded = any(action['action'] == 0 for action in river_actions)
        stats['river_fold_history'].append(1 if river_folded else 0)
    
    def _update_fold_to_cbet_stats(self, stats: dict, flop_actions: List[dict], 
                                  turn_actions: List[dict], river_actions: List[dict]):
        """Update fold-to-continuation-bet statistics using sliding window."""
        
        # Simplified fold-to-cbet tracking for sliding window
        # Flop fold to C-bet (when facing a bet)
        faced_flop_bet = flop_actions and any(action['was_facing_bet'] for action in flop_actions)
        folded_to_flop_bet = faced_flop_bet and any(action['action'] == 0 for action in flop_actions)
        stats['fold_to_cbet_flop_history'].append(1 if folded_to_flop_bet else 0)
        
        # Turn fold to C-bet
        faced_turn_bet = turn_actions and any(action['was_facing_bet'] for action in turn_actions)
        folded_to_turn_bet = faced_turn_bet and any(action['action'] == 0 for action in turn_actions)
        stats['fold_to_cbet_turn_history'].append(1 if folded_to_turn_bet else 0)
        
        # River fold to C-bet
        faced_river_bet = river_actions and any(action['was_facing_bet'] for action in river_actions)
        folded_to_river_bet = faced_river_bet and any(action['action'] == 0 for action in river_actions)
        stats['fold_to_cbet_river_history'].append(1 if folded_to_river_bet else 0)
    
    def _update_checkraise_stats(self, stats: dict, flop_actions: List[dict], 
                                turn_actions: List[dict]):
        """Update check-raise statistics using sliding window."""
        
        # Simplified check-raise detection
        # Flop check-raise: check followed by raise
        flop_checkraised = False
        if len(flop_actions) >= 2:
            for i in range(len(flop_actions) - 1):
                if flop_actions[i]['action'] == 1 and flop_actions[i+1]['action'] == 2:
                    flop_checkraised = True
                    break
        stats['checkraise_flop_history'].append(1 if flop_checkraised else 0)
        
        # Turn check-raise
        turn_checkraised = False
        if len(turn_actions) >= 2:
            for i in range(len(turn_actions) - 1):
                if turn_actions[i]['action'] == 1 and turn_actions[i+1]['action'] == 2:
                    turn_checkraised = True
                    break
        stats['checkraise_turn_history'].append(1 if turn_checkraised else 0)
    
    def _update_showdown_stats(self, stats: dict, player_id: int):
        """Update showdown-related statistics using sliding window."""
        
        # WTSD (Went To Showdown) - simplified
        went_to_showdown = player_id in self.current_hand_data['went_to_showdown']
        stats['wtsd_history'].append(1 if went_to_showdown else 0)
        
        # Showdown wins (only append when actually went to showdown)
        if went_to_showdown:
            won_showdown = player_id in self.current_hand_data['showdown_winners']
            stats['showdown_win_history'].append(1 if won_showdown else 0)
    
    def _update_general_stats(self, stats: dict, player_id: int):
        """Update general playing tendencies using sliding window."""
        
        all_actions = self.current_hand_data['player_actions'][player_id]
        
        # All-in frequency (simplified detection)
        went_all_in = any(action['action'] == 2 and action['amount'] and action['amount'] >= action['pot_size'] 
                         for action in all_actions)
        stats['all_in_history'].append(1 if went_all_in else 0)
        
        # Store bet sizing data for deque
        bet_sizes = [action['amount'] for action in all_actions 
                    if action['action'] == 2 and action['amount'] and action['pot_size'] > 0]
        
        if bet_sizes:
            for bet_size in bet_sizes:
                stats['bet_sizes'].append(bet_size)
            
            # Calculate pot ratios
            pot_ratios = [action['amount'] / max(action['pot_size'], 1) 
                         for action in all_actions 
                         if action['action'] == 2 and action['amount'] and action['pot_size'] > 0]
            
            for pot_ratio in pot_ratios:
                stats['pot_ratios'].append(pot_ratio)
    
    def get_player_percentages(self, player_id: str) -> Dict[str, float]:
        """
        Get calculated percentage stats for a player using sliding window data.
        This is the main method used by the RangeConstructor.
        """
        raw = self.stats[player_id]
        
        def safe_percentage_from_history(history_deque):
            """Calculate percentage from a deque of 1s and 0s."""
            if not history_deque or len(history_deque) == 0:
                return 0.0
            return sum(history_deque) / len(history_deque)
        
        def safe_average_from_values(values_deque):
            """Calculate average from a deque of numeric values."""
            if not values_deque or len(values_deque) == 0:
                return 0.0
            return sum(values_deque) / len(values_deque)
        
        return {
            # Meta - use total hands and recent sample size
            'total_hands': raw['total_hands'],
            'sample_size': len(raw['vpip_history']) if raw['vpip_history'] else 0,
            
            # Pre-flop percentages (from sliding window)
            'vpip': safe_percentage_from_history(raw['vpip_history']),
            'pfr': safe_percentage_from_history(raw['pfr_history']),
            'three_bet': safe_percentage_from_history(raw['three_bet_history']),
            'fold_to_three_bet': safe_percentage_from_history(raw['fold_to_three_bet_history']),
            
            # Post-flop aggression (from sliding window)
            'cbet_flop': safe_percentage_from_history(raw['cbet_flop_history']),
            'cbet_turn': safe_percentage_from_history(raw['cbet_turn_history']),
            'cbet_river': safe_percentage_from_history(raw['cbet_river_history']),
            
            # Post-flop defense (from sliding window)
            'fold_to_cbet_flop': safe_percentage_from_history(raw['fold_to_cbet_flop_history']),
            'fold_to_cbet_turn': safe_percentage_from_history(raw['fold_to_cbet_turn_history']),
            'fold_to_cbet_river': safe_percentage_from_history(raw['fold_to_cbet_river_history']),
            
            # Advanced stats (from sliding window)
            'checkraise_flop': safe_percentage_from_history(raw['checkraise_flop_history']),
            'checkraise_turn': safe_percentage_from_history(raw['checkraise_turn_history']),
            'aggression_frequency': safe_percentage_from_history(raw['aggression_frequency_history']),
            
            # Showdown stats (from sliding window)
            'wtsd': safe_percentage_from_history(raw['wtsd_history']),
            'showdown_win_rate': safe_percentage_from_history(raw['showdown_win_history']),
            
            # Overall tendencies (from sliding window)
            'fold_frequency': safe_percentage_from_history(raw['fold_frequency_history']),
            'all_in_frequency': safe_percentage_from_history(raw['all_in_history']),
            
            # Street-specific fold rates (from sliding window)
            'preflop_fold_rate': safe_percentage_from_history(raw['preflop_fold_history']),
            'flop_fold_rate': safe_percentage_from_history(raw['flop_fold_history']),
            'turn_fold_rate': safe_percentage_from_history(raw['turn_fold_history']),
            'river_fold_rate': safe_percentage_from_history(raw['river_fold_history']),
            
            # Bet sizing analysis (for Layer 2)
            'avg_bet_size': safe_average_from_values(raw['bet_sizes']),
            'avg_pot_ratio': safe_average_from_values(raw['pot_ratios']),
        }
    
    def save_stats(self):
        """Save current stats to disk, converting deques to lists."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        # Create a serializable copy of the stats
        stats_to_save = {}
        for player_id, stats in self.stats.items():
            stats_to_save[player_id] = {}
            for key, value in stats.items():
                if isinstance(value, deque):
                    stats_to_save[player_id][key] = list(value)
                else:
                    stats_to_save[player_id][key] = value
        
        with open(self.filepath, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
    
    def get_diagnostic_summary(self, player_id: str) -> str:
        """Generate a human-readable diagnostic summary."""
        stats = self.get_player_percentages(player_id)
        
        if stats['total_hands'] < 10:
            return f"Player {player_id}: Insufficient data ({stats['total_hands']} hands)"
        
        # Classify player type based on key stats
        vpip, pfr = stats['vpip'], stats['pfr']
        
        if vpip < 0.20:
            tight_loose = "Very Tight"
        elif vpip < 0.35:
            tight_loose = "Tight"
        elif vpip < 0.50:
            tight_loose = "Loose"
        else:
            tight_loose = "Very Loose"
        
        if pfr < 0.10:
            aggressive_passive = "Very Passive"
        elif pfr < 0.20:
            aggressive_passive = "Passive"
        elif pfr < 0.30:
            aggressive_passive = "Aggressive"
        else:
            aggressive_passive = "Very Aggressive"
        
        return f"""Player {player_id} ({stats['total_hands']} hands): {tight_loose}-{aggressive_passive}
  Pre-flop: VPIP={stats['vpip']:.1%}, PFR={stats['pfr']:.1%}, 3-bet={stats['three_bet']:.1%}
  Post-flop: C-bet={stats['cbet_flop']:.1%}, Fold-to-C-bet={stats['fold_to_cbet_flop']:.1%}, Agg={stats['aggression_frequency']:.1%}
  Showdown: WTSD={stats['wtsd']:.1%}, W$SD={stats['showdown_win_rate']:.1%}
  Tendencies: Fold={stats['fold_frequency']:.1%}, Avg Bet={stats['avg_pot_ratio']:.2f}x pot"""
    
