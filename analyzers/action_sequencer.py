# analyzers/action_sequencer.py
# for current street analyzer

from typing import List

class ActionSequencer:
    """
    Temporarily records the sequence of actions for the CURRENT street only.
    This is a lightweight, external logger, not part of the game state.
    """
    def __init__(self):
        # This list will hold the live action sequence for the current street.
        self.current_street_log: List[tuple] = []

    def record_action(self, seat_id: int, action_type: str, amount: int = 0):
        """
        Records a single action to the in-memory live log.
        This is called right before env.step().
        """
        self.current_street_log.append((seat_id, action_type, amount))

    def new_street(self):
        """Clears the live log for the start of a new street."""
        self.current_street_log.clear()

    def get_live_action_sequence(self) -> list:
        """Provides the current, up-to-the-second action sequence."""
        return self.current_street_log

