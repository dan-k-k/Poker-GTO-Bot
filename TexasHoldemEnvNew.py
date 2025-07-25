# TexasHoldemEnvNew.py
# Refactored poker environment using GameState and centralized HandEvaluator
# Clean separation of game logic from AI feature extraction

import random
from typing import List, Optional
from poker_core import GameState, Deck, HandEvaluator


class TexasHoldemEnv:
    """
    Clean poker game environment focused purely on game logic.
    AI-specific tracking moved to FeatureExtractor.
    Uses GameState for clean data/logic separation.
    """
    
    def __init__(self, num_players=5, starting_stack=200, small_blind=1, big_blind=2, seed=None):
        assert 2 <= num_players <= 9
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # Use centralized deck and hand evaluator
        self.deck = Deck(seed)
        self.evaluator = HandEvaluator()
        
        # Game state container
        self.state: Optional[GameState] = None
        
        self.reset()
    
    def reset(self, preserve_stacks: bool = False):
        """Reset for new hand."""

        # 3) Determine surviving players and stacks before anything else
        if not preserve_stacks or not hasattr(self, 'state') or self.state is None:
            # Fresh game - all players survive
            surviving_players = list(range(self.num_players))
            stacks = [self.starting_stack] * self.num_players
        else:
            # First, check if the tournament is already over from the previous state.
            # This prevents infinite loops when preserve_stacks=True is called on a terminal state.
            if self.state and self.state.terminal and self.state.win_reason == 'tournament_winner':
                return self.get_state_dict()
            
            # Preserve stacks but eliminate bust players
            surviving_players = []
            stacks = self.state.stacks.copy()
            
            for player_id in range(self.num_players):
                if stacks[player_id] > 0:
                    surviving_players.append(player_id)
                else:
                    # Player is bust - permanently eliminated
                    stacks[player_id] = 0
            
            # CRITICAL FIX: Check for tournament winner and RETURN IMMEDIATELY
            if len(surviving_players) <= 1:
                if len(surviving_players) == 1:
                    winner = surviving_players[0]
                    # Tournament winner (silent)
                    # Reuse existing hole cards to avoid dealing new ones
                    existing_hole_cards = self.state.hole_cards if self.state else [[-1, -1]] * self.num_players
                    self.state = GameState(
                        num_players=self.num_players,
                        starting_stack=self.starting_stack,
                        small_blind=self.small_blind,
                        big_blind=self.big_blind,
                        hole_cards=existing_hole_cards,  # Don't deal new cards for winner state
                        community=[],
                        stacks=stacks,
                        current_bets=[0] * self.num_players,
                        pot=0,
                        starting_pot_this_round=0,
                        active=[False] * self.num_players,
                        all_in=[False] * self.num_players,
                        acted=[True] * self.num_players,  # No actions needed
                        surviving_players=surviving_players,
                        stage=4,  # Beyond normal game stages
                        dealer_pos=winner,
                        sb_pos=winner,
                        bb_pos=winner,
                        to_move=winner,
                        initial_bet=0,
                        last_raise_size=0,
                        last_raiser=None,
                        terminal=True,
                        winners=[winner],
                        win_reason='tournament_winner'
                    )
                    return self.get_state_dict()  # IMMEDIATE RETURN - DO NOT CONTINUE
                else:
                    # Should not happen - no survivors
                    raise ValueError("No surviving players in tournament")
        
        # 1) Reset deck & shuffle
        self.deck.reset()
        
        # 2) Deal hole cards
        hole_cards = []
        for p in range(self.num_players):
            player_cards = self.deck.deal(2)
            hole_cards.append(player_cards)
        
        # 4) Advance dealer button (clockwise to next surviving player)
        if not hasattr(self, 'state') or self.state is None:
            dealer_pos = surviving_players[0]
        else:
            # Find next surviving player clockwise from current dealer
            current_dealer = self.state.dealer_pos
            next_dealer = (current_dealer + 1) % self.num_players
            
            # Advance clockwise until we find a surviving player
            while next_dealer not in surviving_players:
                next_dealer = (next_dealer + 1) % self.num_players
            
            dealer_pos = next_dealer
        
        # 5) Calculate positions (only among surviving players)
        if len(surviving_players) == 2:
            # Heads-up: Dealer is SB, other player is BB
            dealer_idx = surviving_players.index(dealer_pos)
            sb_pos = surviving_players[dealer_idx]
            bb_pos = surviving_players[(dealer_idx + 1) % 2]
        else:
            # 3+ players: SB is left of dealer, BB is left of SB
            dealer_idx = surviving_players.index(dealer_pos)
            sb_idx = (dealer_idx + 1) % len(surviving_players)
            bb_idx = (dealer_idx + 2) % len(surviving_players)
            sb_pos = surviving_players[sb_idx]
            bb_pos = surviving_players[bb_idx]
        
        # 6) Post blinds
        sb_amount = min(self.small_blind, stacks[sb_pos])
        bb_amount = min(self.big_blind, stacks[bb_pos])
        
        stacks[sb_pos] -= sb_amount
        stacks[bb_pos] -= bb_amount
        
        current_bets = [0] * self.num_players
        current_bets[sb_pos] = sb_amount
        current_bets[bb_pos] = bb_amount
        
        pot = sb_amount + bb_amount
        
        # 7) Determine first to act (only among surviving players)
        if len(surviving_players) == 2:
            to_move = sb_pos  # SB acts first in heads-up
        else:
            # UTG is left of BB
            bb_idx = surviving_players.index(bb_pos)
            utg_idx = (bb_idx + 1) % len(surviving_players)
            to_move = surviving_players[utg_idx]
        
        # Set up player states (all players, but only surviving ones matter)
        active = [p in surviving_players for p in range(self.num_players)]
        all_in = [stacks[p] == 0 for p in range(self.num_players)]
        
        # Ensure first player to act can actually act (not all-in)
        while to_move in surviving_players and all_in[to_move]:
            current_idx = surviving_players.index(to_move)
            next_idx = (current_idx + 1) % len(surviving_players)
            to_move = surviving_players[next_idx]
        
        # 8) Create GameState
        self.state = GameState(
            num_players=self.num_players,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            hole_cards=hole_cards,
            community=[],
            stacks=stacks,
            current_bets=current_bets,
            pot=pot,
            starting_pot_this_round=0,
            active=active,
            all_in=all_in,
            acted=[False] * self.num_players,
            surviving_players=surviving_players,
            stage=0,  # Preflop
            dealer_pos=dealer_pos,
            sb_pos=sb_pos,
            bb_pos=bb_pos,
            to_move=to_move,
            initial_bet=self.big_blind,
            last_raise_size=self.big_blind,
            last_raiser=None,
            terminal=False,
            winners=None,
            win_reason=None
        )
        
        return self.get_state_dict()
    
    def get_state_dict(self):
        """Convert GameState to dictionary for compatibility with existing code."""
        if self.state is None:
            raise ValueError("Game state not initialized")
        
        legal_actions = self._compute_legal_actions(self.state.to_move)
        
        return {
            'hole': self.state.hole_cards[self.state.to_move][:],
            'community': self.state.community.copy(),
            'pot': self.state.pot,
            'to_move': self.state.to_move,
            'stage': self.state.stage,
            'legal_actions': legal_actions,
            'current_bets': self.state.current_bets.copy(),
            'stacks': self.state.stacks.copy(),
            'active': self.state.active.copy(),
            'surviving_players': self.state.surviving_players.copy(),
            'dealer_pos': self.state.dealer_pos,
            'sb_pos': self.state.sb_pos,
            'bb_pos': self.state.bb_pos,
            'initial_bet': self.state.initial_bet,
            'terminal': self.state.terminal,
            'last_raiser': self.state.last_raiser,
            'starting_pot_this_round': self.state.starting_pot_this_round,
            'starting_stack': self.state.starting_stack,
            'big_blind': self.state.big_blind,
            'small_blind': self.state.small_blind,
            'num_players': self.state.num_players,
            'winners': self.state.winners.copy() if self.state.winners else [],
            'win_reason': self.state.win_reason
        }
    
    def _compute_legal_actions(self, player):
        """Compute legal actions for given player."""
        if not self.state.active[player] or self.state.stacks[player] == 0:
            return []
        
        current_max = max(self.state.current_bets)
        have = self.state.current_bets[player]
        to_call = current_max - have
        
        legal = []
        
        # Check/Call (always legal if active)
        legal.append(1)
        
        # Fold (only if facing a bet)
        if to_call > 0:
            legal.append(0)
        
        # Raise (if player has chips and can make minimum raise)
        min_raise = self._min_raise_amount(player)
        if min_raise is not None:
            legal.append(2)
            # If max raise is less than min raise amt, 'all in' button appears
        
        return sorted(legal)
    
    def _min_raise_amount(self, player):
        """Calculate minimum raise amount with robust short-stack handling."""
        if not self.state.active[player] or self.state.stacks[player] == 0:
            return None

        current_max = max(self.state.current_bets)
        have = self.state.current_bets[player]
        to_call = current_max - have
        player_stack = self.state.stacks[player]

        # CORE FIX: A player cannot raise if their entire stack is insufficient to even call.
        # In this case, their "all-in" is just a short call, not a raise.
        if player_stack <= to_call:
            return None

        # If we get here, the player can at least call and has chips remaining.
        # Now, calculate what a full, standard minimum raise would be.
        # Pre-flop special handling for the opener (e.g. raise from BB of 2 to 4)
        last_raise = self.state.last_raise_size
        if self.state.stage == 0 and current_max == self.state.big_blind:
            last_raise = self.state.big_blind
            
        full_raise_amount = (current_max - have) + last_raise

        # Can the player afford a full, standard raise?
        if player_stack >= full_raise_amount:
            # Yes. The minimum additional amount is the standard full raise.
            return full_raise_amount
        else:
            # No, they cannot make a full raise.
            # However, since we've established player_stack > to_call,
            # their all-in constitutes a valid "short raise".
            # The only legal raise amount for them is their entire stack.
            return player_stack
    
    def step(self, action, amount=None):
        """Execute an action and advance game state."""
        if self.state is None:
            raise ValueError("Game state not initialized")
        
        player = self.state.to_move
        current_max = max(self.state.current_bets)
        
        # Execute action
        if action == 0:  # FOLD
            self.state.active[player] = False
            
        elif action == 1:  # CHECK/CALL
            to_call = current_max - self.state.current_bets[player]
            if to_call > 0:
                call_amt = min(to_call, self.state.stacks[player])
                self.state.stacks[player] -= call_amt
                self.state.current_bets[player] += call_amt
                self.state.pot += call_amt
                
        else:  # BET/RAISE
            required = self._min_raise_amount(player)
            
            # CRITICAL FIX: Strict action validation to prevent illegal under-raises
            if required is None:
                raise ValueError(f"Player {player} cannot raise (insufficient chips or illegal state)")
            
            if amount < required:
                current_total = self.state.current_bets[player] + amount
                required_total = self.state.current_bets[player] + required
                raise ValueError(f"Illegal under-raise: Player {player} tried to raise by {amount} (to {current_total} total), minimum raise by {required} (to {required_total} total)")
            
            if amount > self.state.stacks[player]:
                raise ValueError(f"Illegal bet size: Player {player} tried to bet {amount}, only has {self.state.stacks[player]} chips")
            
            # Calculate new total bet after this action
            new_total_bet = self.state.current_bets[player] + amount
            
            # Prevent illegal under-raises relative to current max bet
            if new_total_bet < current_max:
                raise ValueError(f"Illegal under-raise: New total bet {new_total_bet} is less than current max bet {current_max}")
            
            is_full_raise = (self.state.stacks[player] == amount) or \
                           (amount >= (current_max - self.state.current_bets[player]) + self.state.last_raise_size)
            
            self.state.stacks[player] -= amount
            self.state.pot += amount
            self.state.current_bets[player] += amount
            
            if is_full_raise:
                self.state.last_raise_size = self.state.current_bets[player] - current_max
                self.state.last_raiser = player
                
                # Reset acted flags for all other players
                for p in range(self.state.num_players):
                    if p != player:
                        self.state.acted[p] = False
        
        self.state.acted[player] = True
        if self.state.stacks[player] == 0:
            self.state.all_in[player] = True
        
        # Check for winner by fold (only among surviving players)
        still_in = [p for p in self.state.surviving_players if self.state.active[p]]
        if len(still_in) == 1:
            winner = still_in[0]
            self.state.stacks[winner] += self.state.pot
            self.state.pot = 0  # Reset pot after awarding chips
            self.state.terminal = True
            self.state.winners = [winner]
            self.state.win_reason = 'fold'
            reward = +1 if winner == 0 else -1
            self._check_tournament_winner()
            return self.get_state_dict(), reward, True
        
        # Check if street is over
        street_is_over = self._is_street_over()
        
        if not street_is_over:
            # Advance to next player (reverse order like old system)
            self.state.to_move = self._next_active_player_reverse(player)
            return self.get_state_dict(), 0, False
        
        # Street is over - handle all-in situations and advance
        if self._should_auto_complete():
            return self._finish_hand_all_in()
        
        # Reset for next street
        self.state.acted = [False] * self.state.num_players
        self.state.last_raiser = None
        self.state.current_bets = [0] * self.state.num_players
        self.state.initial_bet = None
        self.state.last_raise_size = self.state.big_blind
        
        # Check if hand is over
        if self.state.stage == 3:  # River complete
            self.state.terminal = True
            self.state.win_reason = 'showdown'
            winners = self._determine_showdown_winner()
            self.state.winners = winners
            share = self.state.pot // len(winners)
            remainder = self.state.pot % len(winners)
            for i, w in enumerate(winners):
                self.state.stacks[w] += share
                if i < remainder:
                    self.state.stacks[w] += 1
            self.state.pot = 0  # Reset pot after awarding chips
            reward = +1 if 0 in winners else -1
            self._check_tournament_winner()
            return self.get_state_dict(), reward, True
        
        # Deal next street
        self.state.stage += 1
        if self.state.stage == 1:  # Flop
            new_cards = self.deck.deal(3)
            self.state.community.extend(new_cards)
        else:  # Turn or River
            new_card = self.deck.deal(1)
            self.state.community.extend(new_card)
        
        self.state.starting_pot_this_round = self.state.pot
        
        # Set first to act (dealer acts last post-flop, only among surviving players) 
        active_survivors = [p for p in self.state.surviving_players 
                           if self.state.active[p] and not self.state.all_in[p]]
        if active_survivors:
            self.state.to_move = self._next_active_player_reverse(self.state.dealer_pos)
        
        return self.get_state_dict(), 0, False
    
    def _is_street_over(self):
        """
        Check if the current betting street is complete.
        This is true if all active players have either matched the highest bet or are all-in.
        """
        active_players = [p for p in self.state.surviving_players if self.state.active[p]]
        if len(active_players) <= 1:
            return True  # Hand is over if only one or zero players are left.

        # Find the highest bet made by any active player in this round.
        highest_bet = max(self.state.current_bets[p] for p in active_players)

        # CRITICAL FIX: Special preflop Big Blind option handling
        if self.state.stage == 0:  # Preflop only
            # BB gets option if they haven't acted and their bet equals the big blind (no raise)
            bb_pos = self.state.bb_pos
            if (bb_pos in active_players and 
                not self.state.acted[bb_pos] and 
                self.state.current_bets[bb_pos] == self.state.big_blind and
                highest_bet == self.state.big_blind):
                return False  # BB must get their option

        # If the highest bet is zero, the round is only over if everyone has acted (checked).
        if highest_bet == 0:
            players_who_can_act = [p for p in active_players if not self.state.all_in[p]]
            return all(self.state.acted[p] for p in players_who_can_act)

        # If there is a bet, the street is over if every active player has
        # either matched the highest bet or is all-in.
        for p in active_players:
            is_all_in = self.state.all_in[p]
            has_matched = self.state.current_bets[p] == highest_bet
            has_acted = self.state.acted[p]

            if not (has_matched or is_all_in) and not has_acted:
                return False  # This player can and must still act.

            if not (has_matched or is_all_in) and has_acted:
                # This is the key scenario: a player has acted but their bet is less than the highest
                # bet, and they are NOT all-in. This means they must act again.
                return False

        return True
    
    def _should_auto_complete(self):
        """Check if hand should auto-complete due to all-in situation."""
        active_survivors = [p for p in self.state.surviving_players if self.state.active[p]]
        return any(self.state.all_in[p] for p in active_survivors)
    
    def _next_active_player(self, current_player):
        """Find next active player who can act (forward direction, only among surviving players)."""
        if current_player not in self.state.surviving_players:
            return self.state.surviving_players[0]
        
        current_idx = self.state.surviving_players.index(current_player)
        for i in range(1, len(self.state.surviving_players)):
            next_idx = (current_idx + i) % len(self.state.surviving_players)
            next_player = self.state.surviving_players[next_idx]
            if self.state.active[next_player] and not self.state.all_in[next_player]:
                return next_player
        return current_player  # Fallback if no one can act
    
    def _next_active_player_reverse(self, current_player):
        """Find next active player who can act (reverse direction, only among surviving players)."""
        if current_player not in self.state.surviving_players:
            return self.state.surviving_players[-1]
        
        current_idx = self.state.surviving_players.index(current_player)
        for i in range(1, len(self.state.surviving_players)):
            next_idx = (current_idx - i) % len(self.state.surviving_players)
            next_player = self.state.surviving_players[next_idx]
            if self.state.active[next_player] and not self.state.all_in[next_player]:
                return next_player
        return current_player  # Fallback if no one can act
    
    def _finish_hand_all_in(self):
        """Complete hand when all players are all-in or folded."""
        # Deal remaining community cards
        while self.state.stage < 3:
            self.state.stage += 1
            if self.state.stage == 1:
                new_cards = self.deck.deal(3)
                self.state.community.extend(new_cards)
            else:
                new_card = self.deck.deal(1)
                self.state.community.extend(new_card)
        
        # Determine winners
        self.state.terminal = True
        self.state.win_reason = 'all_in_showdown'
        winners = self._determine_showdown_winner()
        self.state.winners = winners
        
        share = self.state.pot // len(winners)
        remainder = self.state.pot % len(winners)
        for i, w in enumerate(winners):
            self.state.stacks[w] += share
            if i < remainder:
                self.state.stacks[w] += 1
        self.state.pot = 0  # Reset pot after awarding chips
        
        reward = +1 if 0 in winners else -1
        self._check_tournament_winner()
        return self.get_state_dict(), reward, True
    
    def _determine_showdown_winner(self):
        """Determine winner(s) at showdown using centralized hand evaluator."""
        best_rank = None
        winners = []
        
        for p in self.state.surviving_players:
            if not self.state.active[p]:
                continue
            
            seven_cards = self.state.hole_cards[p] + self.state.community
            rank = self.evaluator.best_hand_rank(seven_cards)
            
            if best_rank is None or rank > best_rank:
                best_rank = rank
                winners = [p]
            elif rank == best_rank:
                winners.append(p)
        
        return winners
    
    def _check_tournament_winner(self):
        """Check if any players went bust and declare tournament winner if only 1 remains."""
        # Update surviving players list - remove anyone with 0 chips
        original_survivors = self.state.surviving_players.copy()
        self.state.surviving_players = [p for p in self.state.surviving_players if self.state.stacks[p] > 0]
        
        # Check if anyone was eliminated (silent tracking)
        eliminated = [p for p in original_survivors if p not in self.state.surviving_players]
        
        # Check for tournament winner - this should END the game immediately
        if len(self.state.surviving_players) == 1:
            winner = self.state.surviving_players[0]
            # Override any existing terminal state to declare tournament winner
            self.state.terminal = True
            self.state.winners = [winner]
            self.state.win_reason = 'tournament_winner'
        elif len(self.state.surviving_players) == 0:
            # This shouldn't happen, but handle gracefully
            self.state.terminal = True
            self.state.winners = []
            self.state.win_reason = 'no_survivors'
    
    # Legacy compatibility methods
    @property
    def hole(self):
        """Legacy compatibility - access hole cards."""
        return self.state.hole_cards if self.state else []
    
    @property
    def community(self):
        """Legacy compatibility - access community cards."""
        return self.state.community if self.state else []
    
    @property
    def num_players(self):
        """Number of players in game."""
        return self._num_players
    
    @num_players.setter
    def num_players(self, value):
        """Set number of players."""
        self._num_players = value

