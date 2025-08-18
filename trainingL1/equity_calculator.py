# trainingL1/equity_calculator.py
# Pure equity calculation - separated from range construction

import random
from typing import List, Tuple, Dict
import treys
from treys import Card, Evaluator


class EquityCalculator:
    """
    Calculates poker hand equity using a fast, professional-grade Monte Carlo simulation.
    """
    def __init__(self):
        # Professional hand evaluator using the 'deuces' library
        self.evaluator = Evaluator()

    def calculate_equity(self, my_hand: List[str], board: List[str],
                        opponent_range: Dict[Tuple[str, str], float],
                        num_simulations: int = 200) -> float:
        """
        Calculates hand equity using a streamlined Monte Carlo simulation with deuces.
        """
        if not opponent_range:
            return 0.5

        try:
            # 1. Convert all known cards to deuces format ONCE at the start.
            my_hand_deuces = [Card.new(c) for c in my_hand]
            board_deuces = [Card.new(c) for c in board]
        except ValueError:
            return 0.5  # Return default if card strings are invalid

        # 2. Create a full treys deck and remove the known cards.
        # This is much cleaner than managing our own integer deck.
        used_cards = set(my_hand_deuces + board_deuces)
        
        # AFTER (cleaner and correct for treys):
        deck = [c for c in treys.Deck.GetFullDeck() if c not in used_cards]

        # 3. Prepare for weighted sampling from the opponent's range.
        hands_in_range = list(opponent_range.keys())
        weights = list(opponent_range.values())
        
        wins, ties, total_sims = 0, 0, 0

        for _ in range(num_simulations):
            # 4. Sample opponent's hand and convert to deuces format.
            opp_hand_str = random.choices(hands_in_range, weights=weights, k=1)[0]
            try:
                opp_hand_deuces = [Card.new(c) for c in opp_hand_str]
            except ValueError:
                continue

            # 5. Check for card conflicts and prepare the simulation deck.
            if any(c in used_cards for c in opp_hand_deuces):
                continue
            
            opp_hand_set = set(opp_hand_deuces)
            sim_deck = [c for c in deck if c not in opp_hand_set]

            # 6. Deal the runout DIRECTLY from the deuces deck. No more conversions!
            cards_needed = 5 - len(board_deuces)
            if len(sim_deck) < cards_needed:
                continue
            
            runout_deuces = random.sample(sim_deck, cards_needed)
            full_board_deuces = board_deuces + runout_deuces

            # 7. Evaluate with deuces.
            try:
                my_score = self.evaluator.evaluate(my_hand_deuces, full_board_deuces)
                opp_score = self.evaluator.evaluate(opp_hand_deuces, full_board_deuces)

                if my_score < opp_score:
                    wins += 1
                elif my_score == opp_score:
                    ties += 1
                total_sims += 1
            except:
                continue
            
        if total_sims == 0:
            return 0.5
            
        return (wins + ties * 0.5) / total_sims
    
