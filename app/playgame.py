# app/playgame.py
import streamlit as st
import time  # <-- Import the time library
from app.TexasHoldemEnv import TexasHoldemEnv
from app.visuals import create_table_image
from app.poker_agents import RandomBot # Or your GTO agent

# --- Initialize Game State ---
if 'env' not in st.session_state:
    st.session_state.env = TexasHoldemEnv(num_players=2)
    st.session_state.agents = [None, RandomBot(seat_id=1)] # Human is P0
    st.session_state.env.reset()

env = st.session_state.env
agents = st.session_state.agents
state = env.get_state_dict()

# --- Main App Layout ---
st.set_page_config(layout="wide")
st.title("Random Poker Bot")

# --- CHANGE 1: Decide IF we should show all cards ---
show_all_cards = state.get('terminal', False)

# --- CHANGE 2: Pass the decision to the visualizer ---
st.image(create_table_image(state, env, show_all_cards))

# --- Game Logic and Controls ---
if state.get('terminal'):
    st.header("Hand Over")
    st.write(f"Winner(s): {state['winners']}")
    st.write(f"Reason: {state['win_reason']}")
    
    # --- ADDED PAUSE at end of hand ---
    time.sleep(3) # Display results for 3 seconds before showing the button
    
    # --- THIS IS THE FIX ---
    # Check if the tournament is over to decide the button text and reset behavior
    if state.get('win_reason') == 'tournament_winner':
        st.header(f"Player {state['winners'][0]} Wins The Tournament!")
        if st.button("Play Again (Reset Stacks)"):
            # A full reset, not preserving stacks
            st.session_state.env.reset(preserve_stacks=False)
            st.rerun()
    else:
        # Normal end of hand, continue the tournament
        if st.button("Next Hand"):
            st.session_state.env.reset(preserve_stacks=True)
            st.rerun()
    # --- END FIX ---
else:
    current_player = state['to_move']
    
    if current_player == 0: # Human's turn
        st.header("Your Action")
        legal_actions = state['legal_actions']
        
        cols = st.columns(5)
        
        # Fold Button
        if 0 in legal_actions:
            if cols[0].button("Fold"):
                env.step(0, None)
                st.rerun()
        
        # Check/Call Button
        if 1 in legal_actions:
            to_call = max(state['current_bets']) - state['current_bets'][0]
            button_text = "Check" if to_call == 0 else f"Call {to_call}"
            if cols[1].button(button_text):
                env.step(1, None)
                st.rerun()

        # Bet/Raise Controls
        if 2 in legal_actions:
            min_raise = env._min_raise_amount(0) or state['stacks'][0]
            max_raise = state['stacks'][0]
            
            if min_raise < max_raise:
                bet_amount = cols[2].slider(
                    "Bet/Raise Amount", 
                    min_value=min_raise, 
                    max_value=max_raise, 
                    value=min_raise
                )
                if cols[3].button("Bet / Raise"):
                    env.step(2, bet_amount)
                    st.rerun()
            else:
                if cols[2].button("All-In"):
                     env.step(2, max_raise)
                     st.rerun()

    else: # Bot's turn
        st.header(f"Waiting for Player {current_player}...")
        
        # --- ADDED PAUSE for bot's turn ---
        time.sleep(1.5) # Simulate bot "thinking" for 1.5 seconds
        
        bot_agent = agents[current_player]
        action, amount = bot_agent.compute_action(state, env)
        env.step(action, amount)
        st.rerun()

