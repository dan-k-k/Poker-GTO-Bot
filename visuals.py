# visuals_pyqt.py

import math
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QFontMetrics
from PyQt5.QtCore import Qt, QRectF, QPointF

class PokerTableWidget(QWidget):
    """
    A custom QWidget to draw the poker table, cards, chips, and player info.
    This encapsulates all drawing logic for the poker game.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 450) # Maintain aspect ratio roughly 4:3
        self.state = None
        self.env = None
        self.show_all_holes = False
        self.hands_in_data = 0

    def set_game_data(self, state, env, show_all_holes, hands_in_data):
        """
        Updates the data used for drawing and requests a repaint.
        """
        self.state = state
        self.env = env
        self.show_all_holes = show_all_holes
        self.hands_in_data = hands_in_data
        self.update() # Request a repaint

    def paintEvent(self, event):
        """
        This method is called by PyQt whenever the widget needs to be redrawn.
        All drawing operations are performed here.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing) # For smoother graphics

        if not self.state or not self.env:
            # Display a message if game data is not yet available
            painter.drawText(self.rect(), Qt.AlignCenter, "Game Not Started")
            return

        width = self.width()
        height = self.height()

        # Scale factors for drawing based on widget size
        scale_x = width / 800.0
        scale_y = height / 600.0

        # 1) Draw the green poker-table “oval” background
        table_width_ratio = 0.8
        table_height_ratio = 0.5
        table_x = width * (1 - table_width_ratio) / 2
        table_y = height * (1 - table_height_ratio) / 2 + (height * 0.15 - height * (1 - 0.6) / 2)
        table_rect = QRectF(table_x, table_y, width * table_width_ratio, height * table_height_ratio)
        painter.setBrush(QColor("#005500"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(table_rect)

        # Draw hands in data counter (top right)
        text_content = f"Hands in Data: {self.hands_in_data}"
        font = QFont("Arial", int(16 * min(scale_x, scale_y))) # This font size remains the same
        painter.setFont(font) 
        font_metrics = QFontMetrics(font)
        text_bounding_rect = font_metrics.boundingRect(text_content)

        text_x_pos = width - int(20 * scale_x) - text_bounding_rect.width()
        text_y_pos = int(5 * scale_y)
        text_draw_rect = QRectF(text_x_pos, text_y_pos, text_bounding_rect.width(), text_bounding_rect.height())

        painter.setBrush(QColor(255, 255, 255, 150))
        painter.setPen(Qt.NoPen)
        padding_x = 10 * scale_x
        padding_y = 5 * scale_y
        background_rect = text_draw_rect.adjusted(-padding_x, -padding_y, padding_x, padding_y)
        painter.drawRoundedRect(background_rect, 0, 0)

        painter.setPen(QColor(Qt.black))
        painter.drawText(text_draw_rect, Qt.AlignLeft | Qt.AlignTop, text_content) 

        # 2) Precompute “seat” positions
        r = min(table_rect.width(), table_rect.height()) * 0.22
        center_x, center_y = table_rect.center().x(), table_rect.center().y()
        seat_positions = []
        num_players = self.env.num_players
        
        for i in range(num_players):
            angle_i = math.pi/2 - i * (2 * math.pi / num_players)
            x_i = center_x + r * math.cos(angle_i)
            y_i = center_y + r * math.sin(angle_i)
            seat_positions.append(QPointF(x_i, y_i))

        # 2.5) Draw dealer button
        dpos = self.state['dealer_pos']
        angle_d = math.pi/2 - dpos * (2 * math.pi / num_players) + 0.35
        ind_r_d = r * 1.6
        ind_x_d = center_x + ind_r_d * math.cos(angle_d)
        ind_y_d = center_y + ind_r_d * math.sin(angle_d)
        
        dealer_radius = int(0.025 * min(width, height))
        painter.setBrush(QColor(Qt.white))
        painter.setPen(QPen(Qt.black, 1))
        painter.drawEllipse(QPointF(ind_x_d, ind_y_d), dealer_radius, dealer_radius*5/8)
        
        painter.setFont(QFont("Arial", int(18 * min(scale_x, scale_y)), QFont.Bold))
        painter.setPen(QColor(Qt.black))
        painter.drawText(QRectF(ind_x_d - dealer_radius, ind_y_d - dealer_radius, 2*dealer_radius, 2*dealer_radius),
                         Qt.AlignCenter, "D")

        # 3) Helper to draw one card (nested function for convenience)
        def _draw_card(px, py, card_id, face_up=True, is_folded=False):
            RANKS = "23456789TJQKA"
            SUITS = "♣♦♥♠"
            SUIT_COLORS = [QColor("#2A7E3E"), QColor("#1C6BA9"), QColor("#D53535"), QColor(Qt.black)]

            card_w = int(0.06 * width)
            card_h = int(0.13 * height)
            card_rect = QRectF(px - card_w/2, py - card_h/2, card_w, card_h)

            if is_folded:
                if self.show_all_holes:
                    painter.setBrush(QColor("#E5E5E5"))
                    painter.setPen(QPen(QColor(Qt.gray), 1))
                    painter.drawRoundedRect(card_rect, 5, 5)
                    
                    label = f"{RANKS[card_id // 4]}{SUITS[card_id % 4]}"
                    painter.setFont(QFont("Arial", int(24 * min(scale_x, scale_y)), QFont.Bold))
                    painter.setPen(QColor(Qt.gray))
                    painter.drawText(card_rect, Qt.AlignCenter, label)
                return

            if face_up:
                suit_index = card_id % 4
                color = SUIT_COLORS[suit_index]
                label = f"{RANKS[card_id // 4]}{SUITS[card_id % 4]}"
                
                painter.setBrush(QColor(Qt.white))
                painter.setPen(QPen(Qt.black, 1.5))
                painter.drawRoundedRect(card_rect, 5, 5)

                painter.setFont(QFont("Arial", int(24 * min(scale_x, scale_y)), QFont.Bold))
                painter.setPen(color)
                painter.drawText(card_rect, Qt.AlignCenter, label)
            else: # Face down
                painter.setBrush(QColor(Qt.blue))
                painter.setPen(QPen(Qt.black, 1.5))
                painter.drawRoundedRect(card_rect, 5, 5)

        # 4) Loop over each player and draw their info
        to_move = self.state['to_move']
        for i in range(num_players):
            px, py = seat_positions[i].x(), seat_positions[i].y()
            is_active = self.state['active'][i]

            # Draw hole cards (moved further from player circle)
            hole_cards = self.env.hole[i]
            show_face = self.show_all_holes or (i == 0)
            
            card_y_offset = int(0.16 * height) # Distance from seat
            if py < center_y:
                card_y = py - card_y_offset
            else:
                card_y = py + card_y_offset

            card_x_spacing = int(0.033 * width)
            _draw_card(px - card_x_spacing, card_y, hole_cards[0], face_up=show_face, is_folded=(not is_active))
            _draw_card(px + card_x_spacing, card_y, hole_cards[1], face_up=show_face, is_folded=(not is_active))

            # Player Name
            name_text = "You (P0)" if i == 0 else f"P{i}"
            name_color = QColor(Qt.black) if is_active else QColor(Qt.gray)
            
            name_y_offset = int(0.33 * height) # Distance from seat
            if py < center_y: # Player is in top half, text goes further up (above cards)
                text_y = py - name_y_offset
            else: # Player is in bottom half, text goes further down (below cards)
                text_y = py + name_y_offset
            
            name_font = QFont("Arial", int(18 * min(scale_x, scale_y)), QFont.Bold) # Store font for measurement
            painter.setFont(name_font)
            name_font_metrics = QFontMetrics(name_font) # Get font metrics for name
            name_bounding_rect = name_font_metrics.boundingRect(name_text)

            # Calculate name_draw_rect based on measured bounding box
            name_draw_x = px - name_bounding_rect.width() / 2
            name_draw_rect = QRectF(name_draw_x, text_y, name_bounding_rect.width(), name_bounding_rect.height())

            # --- NEW: Draw glassy white plane behind Player Name ---
            painter.setBrush(QColor(255, 255, 255, 150)) # White with transparency
            painter.setPen(Qt.NoPen)
            name_bg_padding_x = 8 * scale_x
            name_bg_padding_y = 4 * scale_y
            name_background_rect = name_draw_rect.adjusted(-name_bg_padding_x, -name_bg_padding_y, name_bg_padding_x, name_bg_padding_y)
            painter.drawRoundedRect(name_background_rect, 5, 5) # Rounded corners for background

            painter.setPen(name_color)
            painter.drawText(name_draw_rect, Qt.AlignLeft | Qt.AlignTop, name_text) # Draw name text

            # Stack Size
            stack_i = self.state['stacks'][i]
            stack_text = str(stack_i)
            stack_font = QFont("Arial", int(25 * min(scale_x, scale_y)), QFont.Bold) # Store font for measurement
            painter.setFont(stack_font)
            stack_font_metrics = QFontMetrics(stack_font) # Get font metrics for stack
            stack_bounding_rect = stack_font_metrics.boundingRect(stack_text)
            
            # --- MODIFIED: Position stack text symmetrically relative to name ---
            stack_offset_from_name_y = int(0.017 * height) # Fixed vertical offset between name and stack
            
            # Calculate the Y position for the stack text
            if py < center_y: # Player is in top half, stack is below name
                stack_y = name_draw_rect.bottom() + stack_offset_from_name_y
            else: # Player is in bottom half, stack is above name
                stack_y = name_draw_rect.top() - stack_offset_from_name_y - stack_bounding_rect.height() # Subtract actual text height
            
            # Calculate stack_draw_rect based on measured bounding box
            stack_draw_x = px - stack_bounding_rect.width() / 2
            stack_draw_rect = QRectF(stack_draw_x, stack_y, stack_bounding_rect.width(), stack_bounding_rect.height())

            # --- NEW: Draw glassy white plane behind Stack Size ---
            painter.setBrush(QColor(255, 255, 255, 150)) # White with transparency
            painter.setPen(Qt.NoPen)
            stack_bg_padding_x = 8 * scale_x
            stack_bg_padding_y = 4 * scale_y
            stack_background_rect = stack_draw_rect.adjusted(-stack_bg_padding_x, -stack_bg_padding_y, stack_bg_padding_x, stack_bg_padding_y)
            painter.drawRoundedRect(stack_background_rect, 5, 5) # Rounded corners for background

            painter.setPen(QColor(Qt.black))
            painter.drawText(stack_draw_rect, Qt.AlignLeft | Qt.AlignTop, stack_text) # Draw stack text

            # Current Bet (chips in front of player)
            bet_i = self.state['current_bets'][i]
            if bet_i > 0:
                chip_x = px
                chip_y_offset = int(-0.05 * height)
                if py > center_y:
                    chip_y = py - chip_y_offset
                else:
                    chip_y = py + chip_y_offset

                chip_radius = int(0.03 * min(width, height))
                
                painter.setBrush(QColor("#B22222"))
                painter.setPen(QPen(Qt.black, 1))
                painter.drawEllipse(QPointF(chip_x, chip_y), chip_radius, chip_radius*5/8)
                
                painter.setFont(QFont("Arial", int(16 * min(scale_x, scale_y)), QFont.Bold))
                painter.setPen(QColor(Qt.white))
                bet_rect = QRectF(chip_x - chip_radius, chip_y - chip_radius, 2*chip_radius, 2*chip_radius)
                painter.drawText(bet_rect, Qt.AlignCenter, str(bet_i))

        # 5) Draw community cards (moved further from center)
        cx, cy = center_x, center_y
        community_card_y = cy + int(-0.05 * height) # Y position remains the same

        # Calculate card width and effective spacing
        card_w = int(0.06 * width)
        effective_card_spacing = int(0.072 * width) # This is the center-to-center distance

        num_community_cards = len(self.state['community'])

        # Determine the starting X for the first card to achieve the desired centering
        community_card_start_x = cx - (2 * effective_card_spacing)

        for idx, card in enumerate(self.state['community']):
            # Draw all cards relative to this calculated start_x
            _draw_card(community_card_start_x + effective_card_spacing * idx, community_card_y, card, face_up=True)

        # 6) Draw the pot total
        display_pot_amount = self.state['starting_pot_this_round']
        if self.state.get('terminal', False):
            display_pot_amount = self.state['pot']

        pot_radius = int(0.07 * min(width, height))
        painter.setBrush(QColor(Qt.white))
        painter.setPen(QPen(Qt.black, 1.2))
        painter.drawEllipse(QPointF(center_x, center_y), pot_radius, pot_radius*5/8)
        
        painter.setFont(QFont("Arial", int(20 * min(scale_x, scale_y)), QFont.Bold))
        painter.setPen(QColor(Qt.black))
        pot_rect = QRectF(center_x - pot_radius, center_y - pot_radius, 2*pot_radius, 2*pot_radius)
        painter.drawText(pot_rect, Qt.AlignCenter, str(display_pot_amount))

        painter.end()

