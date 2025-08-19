# app/playgame.py

import sys
import os
import time
# Removed math and random as they are not directly used in this file anymore
from collections import deque # Still needed for StatsTracker if it's imported directly

# Import your core game logic and AI agents
from TexasHoldemEnvNew import TexasHoldemEnv
import random

# Import the new visuals_pyqt.py
from Poker.app.visuals import PokerTableWidget

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QSizePolicy
)
from PyQt5.QtGui import QFont # Only Font is needed from QtGui
from PyQt5.QtCore import Qt, QTimer # Only Qt and QTimer from QtCore


# --- Simple Random Bot Agent ---
class RandomBot:
    """Simple bot that makes random legal actions with configurable aggression."""
    
    def __init__(self, player_id, aggression=0.3):
        self.player_id = player_id
        self.aggression = aggression  # Probability of raising when possible
    
    def compute_action(self, state, env):
        legal_actions = state['legal_actions']
        
        if not legal_actions:
            return 1, None  # Default to check if no actions available
        
        # Random action selection with some aggression
        if 2 in legal_actions and random.random() < self.aggression:
            # Raise/bet - choose random amount between min and max
            min_raise = env._min_raise_amount(self.player_id)
            max_raise = state['stacks'][self.player_id]
            if min_raise is not None and min_raise <= max_raise:
                if min_raise == max_raise:
                    amount = max_raise  # All-in
                else:
                    amount = random.randint(min_raise, max_raise)
                return 2, amount
        
        # Otherwise, call/check if possible, fold if must
        if 1 in legal_actions:
            return 1, None
        elif 0 in legal_actions:
            return 0, None
        else:
            return random.choice(legal_actions), None
    
    def new_hand(self):
        """Called at start of each hand."""
        pass
    
    def observe(self, action, player_id, state, env):
        """Called when observing opponent actions (no-op for random bot)."""
        pass
    
    def observe_showdown(self, state, env):
        """Called at showdown (no-op for random bot)."""
        pass
    
    def save_all(self):
        """Called to save agent state (no-op for random bot)."""
        pass
    
    def reset_player_stats(self):
        """Called to reset stats (no-op for random bot)."""
        pass

# --- PokerGameWindow (Main Application Window) ---
class PokerGameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Texas Hold'em AI")
        self.setGeometry(100, 100, 800, 600) # Initial window size

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- Top-right layout for Reset Stats button ---
        self.top_right_layout = QHBoxLayout()
        # Add a stretch to push the button to the right
        self.top_right_layout.addStretch(1) 
        # Add this layout to the main layout at the top
        self.main_layout.addLayout(self.top_right_layout) 

        # Game State and Agents - Tournament setup with multiple players
        self.num_players = 2  # Tournament with 4 players
        self.env = TexasHoldemEnv(num_players=self.num_players, starting_stack=20, small_blind=1, big_blind=2, seed=None)
        
        # Create agents: Human is Player 0, random bots for others
        self.agents = [None]  # Player 0 is human (no agent needed)
        for i in range(1, self.num_players):
            # Create bots with different aggression levels
            aggression = 0.2 + (i * 0.1)  # Bot 1: 0.3, Bot 2: 0.4, Bot 3: 0.5
            self.agents.append(RandomBot(i, aggression))
        
        # Track for stats compatibility (simplified)
        self.hands_played = 0

        # UI Elements
        self.slider_label = QLabel("Total Bet:")
        self.slider_label.setFont(QFont("Arial", 12)) # Bigger font
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.env.starting_stack * 2) # Max possible bet, adjust as needed
        self.slider.setValue(0)
        self.slider.setSingleStep(1) # For +/- buttons
        self.slider.setPageStep(self.env.big_blind) # For clicking on bar
        
        # Label to show slider's current value
        self.slider_value_display = QLabel("0")
        self.slider_value_display.setMinimumWidth(50) # Ensure enough space for numbers
        self.slider_value_display.setAlignment(Qt.AlignCenter)
        self.slider_value_display.setFont(QFont("Arial", 12, QFont.Bold)) # Bigger font, bold
        self.slider.valueChanged.connect(self._update_slider_value_display) # Connect signal

        # +/- Buttons for slider (larger fixed size)
        self.btn_minus = QPushButton("-")
        self.btn_plus = QPushButton("+")
        self.btn_minus.setFixedSize(45, 45) # Larger buttons
        self.btn_plus.setFixedSize(45, 45) # Larger buttons
        self.btn_minus.setFont(QFont("Arial", 14, QFont.Bold)) # Bigger font
        self.btn_plus.setFont(QFont("Arial", 14, QFont.Bold)) # Bigger font
        self.btn_minus.clicked.connect(self._decrement_slider)
        self.btn_plus.clicked.connect(self._increment_slider)

        # Bet/Action Buttons (larger minimum size)
        self.btn_fold = QPushButton("Fold")
        self.btn_check = QPushButton("Check")
        self.btn_bet = QPushButton("Bet")
        self.btn_fold.setMinimumSize(90, 45) # Larger buttons
        self.btn_check.setMinimumSize(90, 45) # Larger buttons
        self.btn_bet.setMinimumSize(90, 45) # Larger buttons
        self.btn_fold.setFont(QFont("Arial", 12, QFont.Bold)) # Bigger font
        self.btn_check.setFont(QFont("Arial", 12, QFont.Bold)) # Bigger font
        self.btn_bet.setFont(QFont("Arial", 12, QFont.Bold)) # Bigger font

        self.btn_fold.clicked.connect(self._on_fold_clicked)
        self.btn_check.clicked.connect(self._on_check_clicked)
        self.btn_bet.clicked.connect(self._on_bet_clicked)

        # Reset Stats Button (placed in top_right_layout)
        self.btn_reset_stats = QPushButton("Reset Player Stats")
        self.btn_reset_stats.clicked.connect(self._on_reset_stats_clicked)
        self.btn_reset_stats.setMinimumSize(120, 40) 
        self.btn_reset_stats.setFont(QFont("Arial", 12, QFont.Bold)) 
        
        # --- NEW: Set stylesheet for red color ---
        self.btn_reset_stats.setStyleSheet("""
            QPushButton {
                background-color: #d9534f; /* A shade of red */
                color: white; /* White text */
                border-radius: 1px; /* Slightly rounded corners */
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #c9302c; /* Darker red on hover */
            }
            QPushButton:pressed {
                background-color: #ac2925; /* Even darker red when pressed */
            }
        """)
        
        self.top_right_layout.addWidget(self.btn_reset_stats) 
        self.top_right_layout.addSpacing(4) 

        # Add the table widget (now imported from visuals_pyqt.py)
        self.table_widget = PokerTableWidget(self)
        self.main_layout.addWidget(self.table_widget, 1) # Give table more space

        # Layout for controls (bottom section)
        self.control_layout = QHBoxLayout() # Define control_layout here
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.btn_minus)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.btn_plus)
        slider_layout.addWidget(self.slider_value_display) # Add slider value display

        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.addWidget(self.btn_fold)
        action_buttons_layout.addWidget(self.btn_check)
        action_buttons_layout.addWidget(self.btn_bet)

        self.control_layout.addLayout(slider_layout)
        self.control_layout.addLayout(action_buttons_layout)
        self.control_layout.addStretch(1) # Push buttons to left (if more space)

        self.main_layout.addLayout(self.control_layout)


        # Game Loop Timer
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self._game_loop_step)
        self.game_running = False
        self.human_action_pending = False
        self.human_choice = (None, None)

        self.start_new_hand()

    def closeEvent(self, event):
        """Handle window closing."""
        print("--- Window closing. ---")
        event.accept()

    # Slider value display update method
    def _update_slider_value_display(self, value):
        self.slider_value_display.setText(str(value))

    def _update_ui_for_state(self):
        state = self.env.get_state_dict()
        self.table_widget.set_game_data(state, self.env, state.get('terminal', False), self.hands_played)

        player_id = 0 # Human player
        current_bet = state['current_bets'][player_id]
        stack = state['stacks'][player_id]
        to_call = max(state['current_bets']) - current_bet

        legal_actions = state['legal_actions']

        # Update slider range and value
        slider_min_val = 0
        slider_max_val = 0
        if 2 in legal_actions: # If bet/raise is legal
            min_raise_additional = self.env._min_raise_amount(player_id) or 0
            slider_min_val = current_bet + min_raise_additional
            slider_max_val = current_bet + stack
            
            # Ensure min_raise is at least the big blind for opens if pre-flop and unopened pot
            if state['stage'] == 0 and to_call == 0:
                 slider_min_val = max(slider_min_val, self.env.big_blind)

            if slider_min_val >= slider_max_val: # All-in only case
                slider_min_val = slider_max_val
                self.btn_bet.setText("All-In")
            else:
                self.btn_bet.setText("Bet" if to_call == 0 else "Raise")

        self.slider.setMinimum(slider_min_val)
        self.slider.setMaximum(slider_max_val)
        # Only set slider value if it's currently within the new valid range, otherwise default to min
        if not (self.slider.value() >= slider_min_val and self.slider.value() <= slider_max_val):
            self.slider.setValue(slider_min_val) # Set to minimum legal bet by default
        self._update_slider_value_display(self.slider.value()) # Update label immediately


        # Update button states and labels
        self.btn_fold.setEnabled(0 in legal_actions)
        self.btn_check.setEnabled(1 in legal_actions)
        self.btn_bet.setEnabled(2 in legal_actions)
        
        # Enable/disable +/- buttons based on legal actions and if not all-in
        can_bet_or_raise = 2 in legal_actions and not (slider_min_val >= slider_max_val)
        self.btn_minus.setEnabled(can_bet_or_raise)
        self.btn_plus.setEnabled(can_bet_or_raise)

        self.btn_check.setText("Check" if to_call == 0 else "Call")
        
        # Enable/disable all action buttons if it's not human's turn or no legal actions
        is_human_turn = (state['to_move'] == player_id) and (not state.get('terminal', False))
        self.btn_fold.setEnabled(self.btn_fold.isEnabled() and is_human_turn)
        self.btn_check.setEnabled(self.btn_check.isEnabled() and is_human_turn)
        self.btn_bet.setEnabled(self.btn_bet.isEnabled() and is_human_turn)
        self.slider.setEnabled(can_bet_or_raise and is_human_turn)
        self.btn_minus.setEnabled(self.btn_minus.isEnabled() and is_human_turn)
        self.btn_plus.setEnabled(self.btn_plus.isEnabled() and is_human_turn)


    def start_new_hand(self):
        print("--- Starting New Hand ---")
        
        # Reset for new hand, preserving stacks after first hand
        is_first_hand = self.hands_played == 0
        state = self.env.reset(preserve_stacks=not is_first_hand)
        
        # Check if tournament is over
        if state.get('terminal') and state.get('win_reason') == 'tournament_winner':
            self._declare_tournament_winner(state)
            return
        
        self.hands_played += 1
        print(f"Hand #{self.hands_played}")
        print(f"Surviving players: {state.get('surviving_players', 'N/A')}")
        print(f"Stacks: {state['stacks']}")
        
        # Notify agents of new hand
        for agent in self.agents:
            if agent:  # Skip None for human player
                agent.new_hand()
        
        self.human_action_pending = False
        self.game_running = True
        self.game_timer.start(100) # Start game loop, update every 100ms
        self._update_ui_for_state()
        
    def _declare_tournament_winner(self, state):
        """Handle tournament completion."""
        print("--- TOURNAMENT OVER ---")
        winner_idx = state['winners'][0]
        winner_name = "🏆 YOU WIN THE TOURNAMENT!" if winner_idx == 0 else f"🏆 Player {winner_idx} wins the tournament!"
        
        print(f"Tournament Winner: Player {winner_idx}")
        print(f"Final Stacks: {state['stacks']}")
        print(f"Total Hands Played: {self.hands_played}")
        print(winner_name)
        
        # Update UI one final time to show the final state
        self._update_ui_for_state()
        self.game_running = False

    def _game_loop_step(self):
        state = self.env.get_state_dict()

        if state.get('terminal', False):
            self.game_timer.stop()
            self.game_running = False
            self._update_ui_for_state() # Final draw for showdown/fold
            print(f"--- Hand over. Stacks: {state['stacks']} ---")
            print(f"Hand result: {state.get('win_reason', 'unknown')}")
            
            # Check if tournament is over
            if state.get('win_reason') == 'tournament_winner':
                QTimer.singleShot(2000, lambda: self._declare_tournament_winner(state))
            else:
                # Continue with next hand
                QTimer.singleShot(3000, self.start_new_hand) # 3 second pause

            return

        curr_player = state['to_move']
        
        if curr_player == 0: # Human player
            if not self.human_action_pending:
                # Enable UI for human input
                self._update_ui_for_state()
                self.human_action_pending = True
                # Pause the game loop until human acts
                self.game_timer.stop() 
        else: # Bot player
            # Ensure UI is disabled during bot's turn
            self.btn_fold.setEnabled(False)
            self.btn_check.setEnabled(False)
            self.btn_bet.setEnabled(False)
            self.slider.setEnabled(False)
            self.btn_minus.setEnabled(False)
            self.btn_plus.setEnabled(False)

            # Simulate bot thinking time
            QTimer.singleShot(int(1.2 * 1000), lambda: self._perform_bot_action(state))
            self.game_timer.stop() # Stop main loop until bot acts

    def _perform_bot_action(self, state):
        curr_player = state['to_move']
        
        # Check if player is still in the game
        if curr_player not in state.get('surviving_players', []):
            print(f"Player {curr_player} is eliminated!")
            self.game_timer.start(100)
            return
        
        agent_to_act = self.agents[curr_player]
        
        action_tuple = (None, None)
        can_act = bool(state['legal_actions'])

        if can_act and agent_to_act:
            action_tuple = agent_to_act.compute_action(state, self.env)
            action_names = {0: "FOLD", 1: "CALL/CHECK", 2: "RAISE"}
            action_name = action_names.get(action_tuple[0], "UNKNOWN")
            if action_tuple[0] == 2 and action_tuple[1]:
                print(f"Player {curr_player}: {action_name} {action_tuple[1]}")
            else:
                print(f"Player {curr_player}: {action_name}")
        else:
            action_tuple = (1, None) # Default to check/pass if no legal actions

        # Step environment
        new_state, reward, done = self.env.step(action_tuple[0], action_tuple[1])

        self.human_action_pending = False
        self._update_ui_for_state() # Update UI after bot action
        self.game_timer.start(100) # Resume game loop

    # --- Human Action Callbacks ---
    def _on_fold_clicked(self):
        self.human_choice = (0, None)
        self._process_human_action()

    def _on_check_clicked(self):
        self.human_choice = (1, None)
        self._process_human_action()

    def _on_bet_clicked(self):
        total_bet = self.slider.value()
        current_bet = self.env.get_state_dict()['current_bets'][0] # Human is P0
        additional_amount = total_bet - current_bet
        
        # Debug output
        state = self.env.get_state_dict()
        print(f"DEBUG RAISE: total_bet={total_bet}, current_bet={current_bet}, additional={additional_amount}")
        print(f"DEBUG: current_bets={state['current_bets']}, max_bet={max(state['current_bets'])}")
        min_raise = self.env._min_raise_amount(0)
        print(f"DEBUG: min_raise_required={min_raise}, legal_actions={state['legal_actions']}")
        
        self.human_choice = (2, additional_amount)
        self._process_human_action()

    def _process_human_action(self):
        if not self.human_action_pending:
            return # Should not happen if buttons are correctly enabled/disabled

        action_code, amount = self.human_choice
        
        # Log human action
        action_names = {0: "FOLD", 1: "CALL/CHECK", 2: "RAISE"}
        action_name = action_names.get(action_code, "UNKNOWN")
        if action_code == 2 and amount:
            print(f"You: {action_name} {amount}")
        else:
            print(f"You: {action_name}")

        new_state, reward, done = self.env.step(action_code, amount)

        self.human_action_pending = False
        self._update_ui_for_state() # Update UI after human action
        self.game_timer.start(100) # Resume game loop

    # --- Slider +/- Callbacks ---
    def _decrement_slider(self):
        self.slider.setValue(self.slider.value() - self.slider.singleStep())

    def _increment_slider(self):
        self.slider.setValue(self.slider.value() + self.slider.singleStep())

    # --- Reset Stats Callback ---
    def _on_reset_stats_clicked(self):
        print("--- GUI: Reset button clicked. ---")
        # Reset game to start fresh tournament
        self.hands_played = 0
        self.start_new_hand()
        self._update_ui_for_state() # Update display

# --- Main Application Entry Point ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PokerGameWindow()
    window.show()
    sys.exit(app.exec_())

