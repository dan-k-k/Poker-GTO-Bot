# app/visuals.py
import math
from PIL import Image, ImageDraw, ImageFont

# Helper to load a font. Add a basic font file or use a path to an existing one.
try:
    # Use the DejaVu font that we installed in the Docker container
    dejavu_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_sm = ImageFont.truetype(dejavu_font_path, 18)
    font_md = ImageFont.truetype(dejavu_font_path, 22)
    font_lg = ImageFont.truetype(dejavu_font_path, 28)
except IOError:
    # Fallback for local testing if the font isn't on your Mac
    print("DejaVu font not found, using default. Symbols may not appear.")
    font_sm = ImageFont.load_default()
    font_md = ImageFont.load_default()
    font_lg = ImageFont.load_default()

def create_table_image(state, env, show_all_cards=False):
    """
    Creates an image of the poker table from the current game state.
    Returns a Pillow Image object.
    """
    W, H = 800, 600  # Image dimensions
    BG_COLOR = "#005500"
    TABLE_COLOR = "#006400"
    CARD_W, CARD_H = 50, 70

    img = Image.new('RGB', (W, H), color=BG_COLOR)
    draw = ImageDraw.Draw(img, 'RGBA')

    # 1. Draw the table oval
    table_rect = [(W * 0.1, H * 0.2), (W * 0.9, H * 0.8)]
    draw.ellipse(table_rect, fill=TABLE_COLOR, outline="black")

    # 2. Draw community cards
    cx, cy = W / 2, H / 2
    card_spacing = CARD_W + 10
    total_card_width = (len(state['community']) * card_spacing) - 10
    start_x = cx - total_card_width / 2
    for i, card_id in enumerate(state['community']):
        _draw_card(draw, (start_x + i * card_spacing, cy - CARD_H / 2), card_id)

    # 3. Draw Pot (now perfectly centered)
    pot_text = f"Pot: {state['pot']}"
    # --- IMPROVED CENTERING ---
    pot_bbox = font_lg.getbbox(pot_text)
    pot_w, pot_h = pot_bbox[2] - pot_bbox[0], pot_bbox[3] - pot_bbox[1]
    draw.text((cx - pot_w / 2, cy + 40), pot_text, font=font_lg, fill="white")
    # --- END IMPROVEMENT ---

    # 4. Draw Players
    num_players = env.num_players
    player_radius = W * 0.3  # Reduced from 0.35 to bring players closer to center
    
    # Shift player center upward while keeping table center the same
    player_cy = cy - 20  # Move players up by 50 pixels
    
    # --- NEW SECTION: Draw Dealer Button ---
    dealer_pos = state['dealer_pos']
    angle = (math.pi / 2) - (dealer_pos * (2 * math.pi / num_players))
    # Position it slightly outside the player circle
    btn_x = cx + (player_radius * 1.1) * math.cos(angle - 0.32) # Offset angle slightly
    btn_y = player_cy - (player_radius * 1.1) * math.sin(angle - 0.32)
    draw.ellipse([(btn_x - 15, btn_y - 15), (btn_x + 15, btn_y + 15)], fill="white", outline="black")
    d_bbox = font_md.getbbox("D")
    d_w, d_h = d_bbox[2] - d_bbox[0], d_bbox[3] - d_bbox[1]
    draw.text((btn_x - d_w/2, btn_y - d_h/2), "D", font=font_md, fill="black")
    # --- END NEW SECTION ---
    
    for i in range(num_players):
        # Simple direct positioning: P0 at bottom, P1 at top
        if i == 0:  # Player 0 (You) - at bottom
            px = cx
            py = player_cy + player_radius
        else:  # Player 1 (Opponent) - at top  
            px = cx
            py = player_cy - player_radius

        player_name = "You (P0)" if i == 0 else f"P{i}"
        stack_text = f"Stack: {state['stacks'][i]}"
        color = (255, 255, 255) if state['active'][i] else (128, 128, 128)
        
        # --- NEW SECTION: Draw Text Background Plane ---
        # Draw a single semi-transparent box behind both name and stack
        name_bbox = font_md.getbbox(player_name)
        stack_bbox = font_sm.getbbox(stack_text)
        box_w = max(name_bbox[2], stack_bbox[2]) + 20 # Width of the box
        box_h = (name_bbox[3] - name_bbox[1]) + (stack_bbox[3] - stack_bbox[1]) + 25 # Height of the box
        box_x = px - box_w / 2
        box_y = py + 15
        draw.rounded_rectangle([(box_x, box_y), (box_x + box_w, box_y + box_h)], radius=5, fill=(255, 255, 255, 100)) # Transparent white
        # --- END NEW SECTION ---
        
        # Draw player name and stack (now on top of the background)
        name_w = name_bbox[2] - name_bbox[0]
        stack_w = stack_bbox[2] - stack_bbox[0]
        draw.text((px - name_w / 2, box_y + 10), player_name, font=font_md, fill=color)
        draw.text((px - stack_w / 2, box_y + 40), stack_text, font=font_sm, fill=color)

        # Draw hole cards
        if 'hole_cards' in state and state['hole_cards'] and i < len(state['hole_cards']):
            hole_cards = state['hole_cards'][i]
            if hole_cards and len(hole_cards) >= 2:
                # --- CHANGE: Decide if cards are face up ---
                face_up = (i == 0) or show_all_cards
                
                # --- CHANGE 1: Determine if the player has folded ---
                is_folded = not state['active'][i]
                
                # --- CHANGE 2: Pass the folded status to the drawing function ---
                _draw_card(draw, (px - CARD_W, py - 20), hole_cards[0], face_up=face_up, is_folded=is_folded)
                _draw_card(draw, (px, py - 20), hole_cards[1], face_up=face_up, is_folded=is_folded)
        
        # --- NEW SECTION: Draw Player's Current Bet ---
        current_bet = state['current_bets'][i]
        if current_bet > 0:
            # Position the bet between the player and the pot
            bet_x = px + (cx - px) * 0.4
            bet_y = py + (cy - py) * 0.4
            
            # Draw chip stack background
            draw.ellipse([(bet_x - 15, bet_y - 10), (bet_x + 15, bet_y + 10)], fill="#EAEAAE")
            # Draw bet amount text
            bet_text = str(current_bet)
            bet_bbox = font_sm.getbbox(bet_text)
            bet_w, bet_h = bet_bbox[2] - bet_bbox[0], bet_bbox[3] - bet_bbox[1]
            draw.text((bet_x - bet_w / 2, bet_y - bet_h / 2), bet_text, font=font_sm, fill="black")
        # --- END NEW SECTION ---

    return img

def _draw_card(draw, pos, card_id, face_up=True, is_folded=False):
    """Helper function to draw a single card onto the image."""
    x, y = pos
    CARD_W, CARD_H = 50, 70
    card_rect = [(x, y), (x + CARD_W, y + CARD_H)]
    
    RANKS = "23456789TJQKA"
    SUITS = "♣♦♥♠"
    rank = RANKS[card_id // 4]
    suit = SUITS[card_id % 4]
    
    # --- CHANGE 4: Add new drawing logic for folded cards ---
    if is_folded:
        # If folded, always draw face-up but greyed out
        draw.rounded_rectangle(card_rect, radius=5, fill="#E5E5E5", outline="grey") # Light grey background
        draw.text((x + 5, y + 5), rank + suit, font=font_lg, fill="grey") # Grey text
        return # Stop here for folded cards
    # --- END CHANGE ---
    
    if face_up:
        color = "red" if suit in "♦♥" else "black"
        draw.rounded_rectangle(card_rect, radius=5, fill="white", outline="black")
        draw.text((x + 5, y + 5), rank + suit, font=font_lg, fill=color)
    else:
        # Draw card back
        draw.rounded_rectangle(card_rect, radius=5, fill="blue", outline="black")

