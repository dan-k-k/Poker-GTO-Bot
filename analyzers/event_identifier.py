# analyzers/event_identifier.py
"""
HandEventIdentifier: Analyzes complete hand histories to identify statistical events

This module's sole responsibility is to take a raw hand history (who did what, when, for how much)
and identify all the statistical opportunities and actions that the StatsTracker should record.

Clean architecture:
- DataCollector: Simulates hands and records raw actions
- HandEventIdentifier: Analyzes complete hand history to identify statistical events  
- StatsTracker: Stores and retrieves statistics (passive storage)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check" 
    CALL = "call"
    BET = "bet"
    RAISE = "raise"

class Street(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

@dataclass
class RawAction:
    """Raw action data collected during hand simulation."""
    player_id: int
    street: Street
    action_type: ActionType
    amount: int = 0
    pot_size_before: int = 0
    was_facing_bet: bool = False
    position: Optional[str] = None
    stack_size: int = 0

@dataclass 
class HandHistory:
    """Complete hand history with all relevant context."""
    hand_id: str
    players: List[int]
    starting_stacks: Dict[int, int]
    blinds: Tuple[int, int]  # (small_blind, big_blind)
    dealer_position: int
    hole_cards: Dict[int, List[int]]  # Optional: only if known
    community_cards: Dict[Street, List[int]]  # Cards revealed each street
    raw_actions: List[RawAction]
    final_pot: int
    winners: List[int]
    showdown_hands: Dict[int, List[int]]  # Hands shown at showdown

class HandEventIdentifier:
    """
    Analyzes complete hand histories to identify all statistical events.
    
    Takes raw action logs and identifies complex poker concepts like:
    - Basic stats: VPIP, PFR, 3-bet, fold to 3-bet
    - Per-street strategic actions: donk bet, probe bet, check-raise, float bet
    - Advanced patterns: double barrel, triple barrel, steal attempts, limp-reraise
    - Bet sizing analysis: value vs bluff sizing, river overbets
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset for new hand analysis."""
        self.hand_history: Optional[HandHistory] = None
        self.street_aggressors: Dict[Street, Optional[int]] = {}
        self.preflop_raiser: Optional[int] = None
        self.limpers: List[int] = []
        self.street_actions: Dict[int, Dict[Street, List[RawAction]]] = {}
    
    def identify_events(self, hand_history: HandHistory) -> Dict[int, Dict[str, Any]]:
        """
        Main entry point: analyze complete hand and return events for each player.
        
        Args:
            hand_history: Complete hand history with all actions
            
        Returns:
            Dict mapping player_id -> events dict for StatsTracker.update_from_events()
        """
        self.reset()
        self.hand_history = hand_history
        self._organize_actions_by_player_and_street()
        self._identify_street_aggressors()
        
        # Generate events for each player
        player_events = {}
        for player_id in hand_history.players:
            player_events[player_id] = self._identify_player_events(player_id)
        
        return player_events
    
    def _organize_actions_by_player_and_street(self):
        """Organize raw actions by player and street for easier analysis."""
        self.street_actions = {
            player_id: {street: [] for street in Street} 
            for player_id in self.hand_history.players
        }
        
        for action in self.hand_history.raw_actions:
            self.street_actions[action.player_id][action.street].append(action)
    
    def _identify_street_aggressors(self):
        """Identify who was the aggressor on each street."""
        self.street_aggressors = {street: None for street in Street}
        
        for action in self.hand_history.raw_actions:
            if action.action_type in [ActionType.BET, ActionType.RAISE]:
                self.street_aggressors[action.street] = action.player_id
                if action.street == Street.PREFLOP:
                    self.preflop_raiser = action.player_id
    
    def _identify_player_events(self, player_id: int) -> Dict[str, Any]:
        """Identify all statistical events for a single player."""
        events = {}
        
        # Get this player's actions by street
        preflop_actions = self.street_actions[player_id][Street.PREFLOP]
        flop_actions = self.street_actions[player_id][Street.FLOP]
        turn_actions = self.street_actions[player_id][Street.TURN]
        river_actions = self.street_actions[player_id][Street.RIVER]
        
        # === BASIC PREFLOP STATS ===
        events.update(self._identify_preflop_events(player_id, preflop_actions))
        
        # === PER-STREET STRATEGIC ACTIONS ===
        events.update(self._identify_strategic_actions(player_id))
        
        # === ADVANCED BETTING PATTERNS ===
        events.update(self._identify_advanced_patterns(player_id))
        
        # === POST-FLOP STATS ===
        events.update(self._identify_postflop_events(player_id, flop_actions, turn_actions, river_actions))
        
        # === SHOWDOWN STATS ===
        events.update(self._identify_showdown_events(player_id))
        
        # === BET SIZING ANALYSIS ===
        events.update(self._identify_bet_sizing_events(player_id))
        
        # === GENERAL STREET ACTION TRACKING ===
        # Add flags for StatsTracker to track per-street fold rates
        if preflop_actions or flop_actions or turn_actions or river_actions:
            events['had_street_action'] = True
            events['preflop_action'] = bool(preflop_actions)
            events['flop_action'] = bool(flop_actions)
            events['turn_action'] = bool(turn_actions)
            events['river_action'] = bool(river_actions)
            
            # Track folding on each street
            events['folded_preflop'] = any(a.action_type == ActionType.FOLD for a in preflop_actions)
            events['folded_flop'] = any(a.action_type == ActionType.FOLD for a in flop_actions)
            events['folded_turn'] = any(a.action_type == ActionType.FOLD for a in turn_actions)
            events['folded_river'] = any(a.action_type == ActionType.FOLD for a in river_actions)
        
        return events
    
    def _identify_preflop_events(self, player_id: int, preflop_actions: List[RawAction]) -> Dict[str, Any]:
        """Identify basic preflop statistical events."""
        events = {}
        
        if not preflop_actions:
            return events
        
        # VPIP: Voluntarily put money in pot
        vpip_action = any(
            action.action_type in [ActionType.CALL, ActionType.BET, ActionType.RAISE] 
            and action.amount > 0
            for action in preflop_actions
        )
        events['vpip_opportunity'] = True
        events['vpip_action'] = vpip_action
        
        # PFR: Pre-flop raise
        pfr_action = any(
            action.action_type in [ActionType.BET, ActionType.RAISE] 
            for action in preflop_actions
        )
        events['pfr_opportunity'] = True
        events['pfr_action'] = pfr_action
        
        # 3-bet: Re-raise preflop
        raises = [a for a in preflop_actions if a.action_type in [ActionType.BET, ActionType.RAISE]]
        three_bet_action = len(raises) > 0 and any(a.was_facing_bet for a in raises)
        if any(a.was_facing_bet for a in preflop_actions):  # Had opportunity to 3-bet
            events['three_bet_opportunity'] = True
            events['three_bet_action'] = three_bet_action
        
        # Fold to 3-bet: Folded when facing a 3-bet
        faced_3bet = any(a.was_facing_bet and a.action_type in [ActionType.BET, ActionType.RAISE] for a in preflop_actions)
        if faced_3bet:
            folded_to_3bet = any(a.action_type == ActionType.FOLD for a in preflop_actions)
            events['fold_to_three_bet_opportunity'] = True
            events['fold_to_three_bet_action'] = folded_to_3bet
        
        # Limp: Just called without facing a bet
        limp_action = any(
            a.action_type == ActionType.CALL and not a.was_facing_bet 
            for a in preflop_actions
        )
        if any(not a.was_facing_bet for a in preflop_actions):  # Had opportunity to limp
            events['limp_opportunity'] = True  
            events['limp_action'] = limp_action
        
        return events
    
    def _identify_strategic_actions(self, player_id: int) -> Dict[str, Any]:
        """Identify per-street strategic actions (check-raise, donk bet, etc.)."""
        events = {}
        
        for street in [Street.FLOP, Street.TURN, Street.RIVER]:
            street_name = street.name.lower()
            actions = self.street_actions[player_id][street]
            
            if not actions:
                continue
            
            # Check-raise: Check followed by raise on same street
            check_raise = self._detect_check_raise(actions)
            if any(a.was_facing_bet for a in actions):  # Had opportunity
                events[f'{street_name}_checkraise_opportunity'] = True
                events[f'{street_name}_checkraise_action'] = check_raise
            
            # Donk bet: Bet out of position when not previous street aggressor
            donk_bet = self._detect_donk_bet(player_id, street, actions)
            if self._had_donk_bet_opportunity(player_id, street):
                events[f'{street_name}_donk_bet_opportunity'] = True
                events[f'{street_name}_donk_bet_action'] = donk_bet
            
            # Probe bet: Bet OOP after preflop aggressor checked back
            if street in [Street.TURN, Street.RIVER]:
                probe_bet = self._detect_probe_bet(player_id, street, actions)
                if self._had_probe_bet_opportunity(player_id, street):
                    events[f'{street_name}_probe_bet_opportunity'] = True
                    events[f'{street_name}_probe_bet_action'] = probe_bet
        
        # Float bet: Call flop IP, then bet when checked to on turn
        float_bet = self._detect_float_bet(player_id)
        if self._had_float_bet_opportunity(player_id):
            events['float_bet_opportunity'] = True
            events['float_bet_action'] = float_bet
        
        return events
    
    def _identify_advanced_patterns(self, player_id: int) -> Dict[str, Any]:
        """Identify advanced multi-street betting patterns."""
        events = {}
        
        # Double barrel: Bet turn after c-betting flop
        if self._had_double_barrel_opportunity(player_id):
            double_barrel = self._detect_double_barrel(player_id)
            events['double_barrel_opportunity'] = True
            events['double_barrel_action'] = double_barrel
        
        # Triple barrel: Bet river after betting flop and turn
        if self._had_triple_barrel_opportunity(player_id):
            triple_barrel = self._detect_triple_barrel(player_id)
            events['triple_barrel_opportunity'] = True
            events['triple_barrel_action'] = triple_barrel
        
        # Delayed c-bet: Check flop as PF aggressor, then bet turn
        if self._had_delayed_cbet_opportunity(player_id):
            delayed_cbet = self._detect_delayed_cbet(player_id)
            events['delayed_cbet_opportunity'] = True
            events['delayed_cbet_action'] = delayed_cbet
        
        # Steal attempt: Raise from late position when folded to
        if self._had_steal_opportunity(player_id):
            steal_attempt = self._detect_steal_attempt(player_id)
            events['steal_attempt_opportunity'] = True
            events['steal_attempt_action'] = steal_attempt
        
        # Fold to steal: Fold in blinds vs late position raise
        if self._had_fold_to_steal_opportunity(player_id):
            fold_to_steal = self._detect_fold_to_steal(player_id)
            events['fold_to_steal_opportunity'] = True
            events['fold_to_steal_action'] = fold_to_steal
        
        # Limp-reraise: 3-bet after limping (trapping)
        if self._had_limp_reraise_opportunity(player_id):
            limp_reraise = self._detect_limp_reraise(player_id)
            events['limp_reraise_opportunity'] = True
            events['limp_reraise_action'] = limp_reraise
            
        return events
    
    def _identify_postflop_events(self, player_id: int, flop_actions: List[RawAction], 
                                turn_actions: List[RawAction], river_actions: List[RawAction]) -> Dict[str, Any]:
        """Identify post-flop continuation betting and defense stats.""" 
        events = {}
        
        # C-bet opportunities and actions
        was_preflop_aggressor = (self.preflop_raiser == player_id)
        
        # Flop c-bet
        if was_preflop_aggressor and flop_actions:
            cbet_flop = any(a.action_type in [ActionType.BET, ActionType.RAISE] for a in flop_actions)
            events['cbet_flop_opportunity'] = True
            events['cbet_flop_action'] = cbet_flop
        
        # Turn c-bet (if was flop aggressor)
        was_flop_aggressor = (self.street_aggressors[Street.FLOP] == player_id)
        if was_flop_aggressor and turn_actions:
            cbet_turn = any(a.action_type in [ActionType.BET, ActionType.RAISE] for a in turn_actions)
            events['cbet_turn_opportunity'] = True
            events['cbet_turn_action'] = cbet_turn
        
        # River c-bet (if was turn aggressor)
        was_turn_aggressor = (self.street_aggressors[Street.TURN] == player_id)
        if was_turn_aggressor and river_actions:
            cbet_river = any(a.action_type in [ActionType.BET, ActionType.RAISE] for a in river_actions)
            events['cbet_river_opportunity'] = True
            events['cbet_river_action'] = cbet_river
        
        # Fold to c-bet (simplified - when facing any bet)
        for street_actions, street_name in [(flop_actions, 'flop'), (turn_actions, 'turn'), (river_actions, 'river')]:
            if street_actions and any(a.was_facing_bet for a in street_actions):
                folded = any(a.action_type == ActionType.FOLD for a in street_actions)
                events[f'fold_to_cbet_{street_name}_opportunity'] = True
                events[f'fold_to_cbet_{street_name}_action'] = folded
        
        # CRITICAL FIX: Add general postflop aggression tracking
        all_postflop_actions = flop_actions + turn_actions + river_actions
        if all_postflop_actions:
            events['had_postflop_action'] = True
            # Check if player was aggressive on any postflop street
            was_aggressive = any(a.action_type in [ActionType.BET, ActionType.RAISE] for a in all_postflop_actions)
            events['was_aggressive_postflop'] = was_aggressive
            # Check if player folded on any postflop street
            folded_postflop = any(a.action_type == ActionType.FOLD for a in all_postflop_actions)
            events['folded_postflop'] = folded_postflop
        
        return events
    
    def _identify_showdown_events(self, player_id: int) -> Dict[str, Any]:
        """Identify showdown-related events."""
        events = {}
        
        # Went to showdown
        went_to_showdown = player_id in self.hand_history.showdown_hands
        events['showdown_opportunity'] = True
        events['went_to_showdown'] = went_to_showdown
        
        # Won at showdown
        if went_to_showdown:
            won_showdown = player_id in self.hand_history.winners
            events['won_showdown'] = won_showdown
        
        return events
    
    def _identify_bet_sizing_events(self, player_id: int) -> Dict[str, Any]:
        """Identify bet sizing patterns for Layer 2 exploitative analysis."""
        events = {}
        
        # Collect all bet sizes
        bet_sizes = []
        pot_ratios = []
        river_overbets = []
        
        for street in Street:
            for action in self.street_actions[player_id][street]:
                if action.action_type in [ActionType.BET, ActionType.RAISE] and action.amount > 0:
                    bet_sizes.append(action.amount)
                    if action.pot_size_before > 0:
                        pot_ratio = action.amount / action.pot_size_before
                        pot_ratios.append(pot_ratio)
                        
                        # River overbet detection
                        if street == Street.RIVER and pot_ratio > 1.0:
                            river_overbets.append(action.amount)
        
        if bet_sizes:
            events['bet_sizes'] = bet_sizes
        if pot_ratios:
            events['pot_ratios'] = pot_ratios
        if river_overbets:
            events['river_overbet_opportunity'] = True
            events['river_overbet_action'] = True
        
        # All-in detection
        all_in = any(
            action.amount >= action.stack_size 
            for street_actions in self.street_actions[player_id].values()
            for action in street_actions
            if action.action_type in [ActionType.BET, ActionType.RAISE, ActionType.CALL]
        )
        if all_in:
            events['all_in_opportunity'] = True
            events['went_all_in'] = True
        
        return events
    
    # === HELPER METHODS FOR PATTERN DETECTION ===
    
    def _detect_check_raise(self, actions: List[RawAction]) -> bool:
        """Detect check-raise pattern in a street's actions."""
        for i in range(len(actions) - 1):
            if (actions[i].action_type == ActionType.CHECK and 
                actions[i+1].action_type in [ActionType.BET, ActionType.RAISE]):
                return True
        return False
    
    def _detect_donk_bet(self, player_id: int, street: Street, actions: List[RawAction]) -> bool:
        """Detect donk bet: bet OOP when not previous street aggressor."""
        if not actions:
            return False
        
        # Must be first action of the street and a bet
        first_action = actions[0]
        if first_action.action_type not in [ActionType.BET]:
            return False
        
        # Must not be the previous street's aggressor
        prev_street = Street(street.value - 1) if street.value > 0 else None
        if prev_street and self.street_aggressors.get(prev_street) == player_id:
            return False
        
        return True
    
    def _detect_probe_bet(self, player_id: int, street: Street, actions: List[RawAction]) -> bool:
        """Detect probe bet: bet OOP after PF aggressor checked back."""
        if not actions or street == Street.PREFLOP:
            return False
        
        first_action = actions[0] 
        if first_action.action_type not in [ActionType.BET]:
            return False
        
        # PF aggressor must have checked the previous street
        prev_street = Street(street.value - 1)
        return (self.preflop_raiser != player_id and 
                self.street_aggressors.get(prev_street) != self.preflop_raiser)
    
    def _detect_float_bet(self, player_id: int) -> bool:
        """Detect float bet: call flop IP, bet when checked to on turn."""
        flop_actions = self.street_actions[player_id][Street.FLOP]
        turn_actions = self.street_actions[player_id][Street.TURN]
        
        # Must have called on flop
        flop_call = any(a.action_type == ActionType.CALL for a in flop_actions)
        if not flop_call:
            return False
        
        # Must bet on turn when checked to
        turn_bet = any(a.action_type in [ActionType.BET] for a in turn_actions)
        return turn_bet
    
    def _detect_double_barrel(self, player_id: int) -> bool:
        """Detect double barrel: bet turn after c-betting flop."""
        flop_bet = any(a.action_type in [ActionType.BET, ActionType.RAISE] 
                      for a in self.street_actions[player_id][Street.FLOP])
        turn_bet = any(a.action_type in [ActionType.BET, ActionType.RAISE] 
                      for a in self.street_actions[player_id][Street.TURN])
        return flop_bet and turn_bet
    
    def _detect_triple_barrel(self, player_id: int) -> bool:
        """Detect triple barrel: bet river after betting flop and turn.""" 
        flop_bet = any(a.action_type in [ActionType.BET, ActionType.RAISE] 
                      for a in self.street_actions[player_id][Street.FLOP])
        turn_bet = any(a.action_type in [ActionType.BET, ActionType.RAISE] 
                      for a in self.street_actions[player_id][Street.TURN])
        river_bet = any(a.action_type in [ActionType.BET, ActionType.RAISE] 
                       for a in self.street_actions[player_id][Street.RIVER])
        return flop_bet and turn_bet and river_bet
    
    def _detect_delayed_cbet(self, player_id: int) -> bool:
        """Detect delayed c-bet: check flop as PF aggressor, then bet turn."""
        if self.preflop_raiser != player_id:
            return False
        
        flop_check = any(a.action_type == ActionType.CHECK 
                        for a in self.street_actions[player_id][Street.FLOP])
        turn_bet = any(a.action_type in [ActionType.BET, ActionType.RAISE] 
                      for a in self.street_actions[player_id][Street.TURN])
        return flop_check and turn_bet
    
    def _detect_steal_attempt(self, player_id: int) -> bool:
        """Detect steal attempt: raise from late position when folded to."""
        # Simplified: any preflop raise
        return any(a.action_type in [ActionType.BET, ActionType.RAISE] 
                  for a in self.street_actions[player_id][Street.PREFLOP])
    
    def _detect_fold_to_steal(self, player_id: int) -> bool:
        """Detect fold to steal: fold in blinds vs late position raise."""
        return any(a.action_type == ActionType.FOLD 
                  for a in self.street_actions[player_id][Street.PREFLOP])
    
    def _detect_limp_reraise(self, player_id: int) -> bool:
        """Detect limp-reraise: 3-bet after limping."""
        preflop_actions = self.street_actions[player_id][Street.PREFLOP]
        
        # Must have both limped and raised
        limped = any(a.action_type == ActionType.CALL and not a.was_facing_bet for a in preflop_actions)
        raised = any(a.action_type in [ActionType.BET, ActionType.RAISE] for a in preflop_actions)
        
        return limped and raised
    
    # === OPPORTUNITY DETECTION HELPERS ===
    
    def _had_donk_bet_opportunity(self, player_id: int, street: Street) -> bool:
        """Check if player had opportunity to donk bet."""
        if street == Street.PREFLOP:
            return False
        
        actions = self.street_actions[player_id][street]
        return len(actions) > 0 and actions[0].action_type in [ActionType.CHECK, ActionType.BET]
    
    def _had_probe_bet_opportunity(self, player_id: int, street: Street) -> bool:
        """Check if player had opportunity to probe bet."""
        return street in [Street.TURN, Street.RIVER] and len(self.street_actions[player_id][street]) > 0
    
    def _had_float_bet_opportunity(self, player_id: int) -> bool:
        """Check if player had opportunity to float bet."""
        flop_actions = self.street_actions[player_id][Street.FLOP]  
        turn_actions = self.street_actions[player_id][Street.TURN]
        return len(flop_actions) > 0 and len(turn_actions) > 0
    
    def _had_double_barrel_opportunity(self, player_id: int) -> bool:
        """Check if player had opportunity to double barrel."""
        flop_bet = any(a.action_type in [ActionType.BET, ActionType.RAISE] 
                      for a in self.street_actions[player_id][Street.FLOP])
        has_turn_actions = len(self.street_actions[player_id][Street.TURN]) > 0
        return flop_bet and has_turn_actions
    
    def _had_triple_barrel_opportunity(self, player_id: int) -> bool:
        """Check if player had opportunity to triple barrel."""
        flop_bet = any(a.action_type in [ActionType.BET, ActionType.RAISE] 
                      for a in self.street_actions[player_id][Street.FLOP])
        turn_bet = any(a.action_type in [ActionType.BET, ActionType.RAISE] 
                      for a in self.street_actions[player_id][Street.TURN])
        has_river_actions = len(self.street_actions[player_id][Street.RIVER]) > 0
        return flop_bet and turn_bet and has_river_actions
    
    def _had_delayed_cbet_opportunity(self, player_id: int) -> bool:
        """Check if player had opportunity for delayed c-bet."""
        is_pf_aggressor = (self.preflop_raiser == player_id)
        flop_check = any(a.action_type == ActionType.CHECK 
                        for a in self.street_actions[player_id][Street.FLOP])
        has_turn_actions = len(self.street_actions[player_id][Street.TURN]) > 0
        return is_pf_aggressor and flop_check and has_turn_actions
    
    def _had_steal_opportunity(self, player_id: int) -> bool:
        """Check if player had opportunity to steal."""
        # Simplified: any preflop action opportunity
        return len(self.street_actions[player_id][Street.PREFLOP]) > 0
    
    def _had_fold_to_steal_opportunity(self, player_id: int) -> bool:
        """Check if player had opportunity to fold to steal."""
        # Simplified: faced a preflop bet
        preflop_actions = self.street_actions[player_id][Street.PREFLOP]
        return any(a.was_facing_bet for a in preflop_actions)
    
    def _had_limp_reraise_opportunity(self, player_id: int) -> bool:
        """Check if player had opportunity to limp-reraise."""
        preflop_actions = self.street_actions[player_id][Street.PREFLOP]
        limped = any(a.action_type == ActionType.CALL and not a.was_facing_bet for a in preflop_actions)
        faced_raise_after_limp = limped and any(a.was_facing_bet for a in preflop_actions)
        return faced_raise_after_limp
    
