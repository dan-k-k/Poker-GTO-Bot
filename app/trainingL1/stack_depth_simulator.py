# app/trainingL1/stack_depth_simulator.py
# Stack depth simulation for more realistic poker training

import numpy as np
from typing import Tuple, List, Dict
from collections import deque


class StackDepthSimulator:
    """
    Simulates realistic stack depth variations in poker training.
    
    Instead of resetting stacks every hand, this system:
    1. Plays sessions of multiple hands with preserve_stacks=True
    2. Periodically resets with asymmetric stack distributions (Gaussian prob.)
    3. Tracks stack depth statistics for analysis
    """
    
    def __init__(self, total_chips: int = 200, session_length: int = 20, 
                 mean_stack_bb: float = 100, std_stack_bb: float = 35, 
                 min_stack_bb: float = 15, big_blind: int = 2):
        """
        Initialize stack depth simulator.
        
        Args:
            total_chips: Total chips between both players (in actual chip units)
            session_length: Number of hands per session before reset
            mean_stack_bb: Mean stack size in big blinds for Gaussian distribution
            std_stack_bb: Standard deviation for stack size distribution
            min_stack_bb: Minimum stack size in big blinds
            big_blind: Size of big blind in chips (for BB to chip conversion)
        """
        self.total_chips = total_chips
        self.session_length = session_length
        self.big_blind = big_blind
        
        self.mean_stack_chips = int(mean_stack_bb * self.big_blind)
        self.std_stack_chips = int(std_stack_bb * self.big_blind)
        self.min_stack_chips = int(min_stack_bb * self.big_blind)
        
        # Store original BB values for reporting
        self.mean_stack_bb = mean_stack_bb
        self.std_stack_bb = std_stack_bb
        self.min_stack_bb = min_stack_bb
        
        # Session tracking
        self.current_session_hand = 0
        
        # Statistics tracking
        self.stack_depth_history = deque(maxlen=1000)
        self.session_results = deque(maxlen=100)
        self.asymmetric_resets = 1  # Count initial asymmetric distribution
        self.total_sessions = 1  # Count initial session
        
        # Stack depth categories for analysis
        self.stack_depth_categories = {
            'deep': 0,      # >150 BB
            'medium': 0,    # 50-150 BB
            'shallow': 0,   # 15-50 BB
            'short': 0      # <15 BB (should be impossible)
        }
        
        # Start with asymmetric stacks immediately
        initial_stacks = self.generate_asymmetric_stacks()
        self.session_start_stacks = list(initial_stacks)
        self.current_stacks = list(initial_stacks)
        
        # Track initial stack depth
        self._update_stack_depth_stats(initial_stacks)
    
    def should_reset_session(self, current_stacks=None) -> bool:
        """
        Check if current session should be reset.
        
        Args:
            current_stacks: Optional tuple/list of current stack sizes
            
        Returns:
            True if session should be reset due to:
            - Reaching session length limit
            - Player going bust (stack = 0)
        """
        # Always reset if we've reached the session length
        if self.current_session_hand >= self.session_length:
            return True
        
        # Reset if any player is bust (stack = 0)
        if current_stacks is not None:
            if len(current_stacks) >= 2:
                if current_stacks[0] <= 0 or current_stacks[1] <= 0:
                    return True
        
        return False
    
    def get_stack_reset_mode(self, current_stacks=None) -> bool:
        """
        Determine if stacks should be preserved or reset for next hand.
        
        Args:
            current_stacks: Optional tuple/list of current stack sizes
            
        Returns:
            True if stacks should be preserved, False if they should be reset
        """
        if self.should_reset_session(current_stacks):
            return False  # Reset stacks (preserve_stacks=False)
        else:
            return True   # Preserve stacks (preserve_stacks=True)
    
    def generate_asymmetric_stacks(self) -> Tuple[int, int]:
        """
        Generate asymmetric stack sizes using Gaussian distribution in CHIPS.
        
        Returns:
            Tuple of (player1_stack, player2_stack) ensuring total = total_chips
        """
        # Generate player 1 stack using the pre-converted CHIP values
        player1_stack = np.random.normal(self.mean_stack_chips, self.std_stack_chips)
        
        # Clamp to reasonable bounds using CHIP values
        max_stack = self.total_chips - self.min_stack_chips
        player1_stack = max(self.min_stack_chips, min(max_stack, player1_stack))
        
        # Player 2 gets the remaining chips
        player2_stack = self.total_chips - player1_stack
        
        # Ensure both players have minimum stack in CHIPS
        if player2_stack < self.min_stack_chips:
            player2_stack = self.min_stack_chips
            player1_stack = self.total_chips - player2_stack
        
        # Convert to integers
        player1_stack = int(player1_stack)
        player2_stack = int(player2_stack)
        
        # Final adjustment to ensure exact total
        total_actual = player1_stack + player2_stack
        if total_actual != self.total_chips:
            diff = self.total_chips - total_actual
            player1_stack += diff
        
        return player1_stack, player2_stack
    
    def reset_session(self, reason="length") -> Tuple[int, int]:
        """
        Reset session with new asymmetric stack distribution.
        
        Args:
            reason: Reason for reset ("length", "bust", "manual")
        
        Returns:
            Tuple of (player1_stack, player2_stack) for new session
        """
        # Generate new asymmetric stacks
        new_stacks = self.generate_asymmetric_stacks()
        
        # Update session tracking
        self.session_start_stacks = list(new_stacks)
        self.current_stacks = list(new_stacks)
        self.current_session_hand = 0
        self.asymmetric_resets += 1
        self.total_sessions += 1
        
        # Track reset reason for statistics
        if not hasattr(self, 'reset_reasons'):
            self.reset_reasons = {'length': 0, 'bust': 0, 'manual': 0}
        self.reset_reasons[reason] = self.reset_reasons.get(reason, 0) + 1
        
        # Track stack depth statistics
        self._update_stack_depth_stats(new_stacks)
        
        return new_stacks
    
    def advance_hand(self, new_stacks: List[int]):
        """
        Advance to next hand in session.
        
        Args:
            new_stacks: Updated stack sizes after current hand
        """
        self.current_stacks = new_stacks.copy()
        self.current_session_hand += 1
        
        # Track stack depth for analysis
        self._update_stack_depth_stats(new_stacks)
    
    def _update_stack_depth_stats(self, stacks: Tuple[int, int]):
        """Update stack depth statistics for analysis."""
        # Calculate effective stack (smaller of the two)
        effective_stack = min(stacks[0], stacks[1])
        
        # Track in history
        self.stack_depth_history.append(effective_stack)
        
        # Update categories (convert chips to BB for comparison)
        effective_stack_bb = effective_stack / self.big_blind
        if effective_stack_bb > 150:
            self.stack_depth_categories['deep'] += 1
        elif effective_stack_bb > 50:
            self.stack_depth_categories['medium'] += 1
        elif effective_stack_bb >= 15:
            self.stack_depth_categories['shallow'] += 1
        else:
            self.stack_depth_categories['short'] += 1
    
    def get_current_session_info(self) -> Dict:
        """Get information about current session."""
        return {
            'session_hand': self.current_session_hand,
            'session_length': self.session_length,
            'session_start_stacks': self.session_start_stacks,
            'current_stacks': self.current_stacks,
            'effective_stack': min(self.current_stacks),
            'hands_remaining': self.session_length - self.current_session_hand
        }
    
    def get_stack_depth_statistics(self) -> Dict:
        """Get comprehensive stack depth statistics."""
        if not self.stack_depth_history:
            return {}
        
        history = list(self.stack_depth_history)
        total_hands = sum(self.stack_depth_categories.values())
        
        stats = {
            'total_sessions': self.total_sessions,
            'asymmetric_resets': self.asymmetric_resets,
            'current_session_hand': self.current_session_hand,
            'avg_effective_stack': np.mean(history),
            'median_effective_stack': np.median(history),
            'std_effective_stack': np.std(history),
            'min_effective_stack': np.min(history),
            'max_effective_stack': np.max(history),
            'stack_depth_distribution': {},
            'reset_reasons': getattr(self, 'reset_reasons', {})
        }
        
        # Calculate distribution percentages
        if total_hands > 0:
            for category, count in self.stack_depth_categories.items():
                stats['stack_depth_distribution'][category] = {
                    'count': count,
                    'percentage': (count / total_hands) * 100
                }
        
        return stats
    
    def print_stack_depth_report(self):
        """Print a comprehensive stack depth report."""
        stats = self.get_stack_depth_statistics()
        
        if not stats:
            print("📊 No stack depth data available yet")
            return
        
        print("\n" + "="*60)
        print("📊 STACK DEPTH SIMULATION REPORT")
        print("="*60)
        
        print(f"Total Sessions: {stats['total_sessions']}")
        print(f"Asymmetric Resets: {stats['asymmetric_resets']}")
        print(f"Current Session Hand: {stats['current_session_hand']}/{self.session_length}")
        
        # Show reset reasons
        reset_reasons = stats.get('reset_reasons', {})
        if reset_reasons:
            print(f"Reset Reasons:")
            for reason, count in reset_reasons.items():
                percentage = (count / stats['total_sessions']) * 100 if stats['total_sessions'] > 0 else 0
                print(f"   {reason.capitalize()}: {count} ({percentage:.1f}%)")
        
        print(f"\n📈 Effective Stack Statistics:")
        print(f"   Average: {stats['avg_effective_stack']:.1f} BB")
        print(f"   Median: {stats['median_effective_stack']:.1f} BB")
        print(f"   Std Dev: {stats['std_effective_stack']:.1f} BB")
        print(f"   Range: {stats['min_effective_stack']:.1f} - {stats['max_effective_stack']:.1f} BB")
        
        print(f"\n🎯 Stack Depth Distribution:")
        for category, data in stats['stack_depth_distribution'].items():
            print(f"   {category.capitalize()}: {data['count']} hands ({data['percentage']:.1f}%)")
        
        # Interpretation
        deep_pct = stats['stack_depth_distribution'].get('deep', {}).get('percentage', 0)
        shallow_pct = stats['stack_depth_distribution'].get('shallow', {}).get('percentage', 0)
        
        print(f"\n💡 Analysis:")
        if deep_pct > 40:
            print("   ✅ Good deep stack exposure for complex strategy development")
        elif deep_pct < 20:
            print("   ⚠️  Limited deep stack training - consider increasing mean_stack_bb")
        
        if shallow_pct > 30:
            print("   ✅ Good shallow stack exposure for push/fold scenarios")
        elif shallow_pct < 10:
            print("   ⚠️  Limited shallow stack training - consider increasing std_stack_bb")
        
        print("="*60)
    
    def reset_statistics(self):
        """Reset all statistics tracking."""
        self.stack_depth_history.clear()
        self.session_results.clear()
        self.asymmetric_resets = 0
        self.total_sessions = 0
        self.stack_depth_categories = {
            'deep': 0,
            'medium': 0,
            'shallow': 0,
            'short': 0
        }

