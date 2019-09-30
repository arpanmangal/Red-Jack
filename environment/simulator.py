"""
Simulator for the game of RedJack
"""

import numpy as np
from player import create, next_state, get_full_state, printable_state


class State:
    """
    State of the Agent
    """
    def __init__ (self, player_card, player_suite, dealer_card, dealer_suite):
        self.me = create (player_card, player_suite)
        self.dealer = create (dealer_card, dealer_suite)

    def update_state (self, card, suite, dealer=False):
        """
        Update the player state
        """
        if not dealer:
            self.me = next_state (self.me, card, suite)
        else:
            self.dealer = next_state (self.dealer, card, suite)

    def max_safe_sum (self, dealer=False):
        """
        The max possible safe sum the player can get in the current state
        safe sum is defined as a sum between 0 to 31 (inclusive)
        Returns either a positive number or -1 denoting bust
        """
        if not dealer:
            possibilities = get_full_state (self.me)
        else:
            possibilities = get_full_state (self.dealer)
            
        possibilities = [p for p in possibilities if 0 <= p <= 31]
        
        if len(possibilities) == 0:
            return -1
        else:
            return possibilities[-1]

    def __str__(self):
        """
        Printing the state
        """
        return "Player: " + printable_state(self.me) + "\nDealer: " + printable_state(self.dealer)


class Action:
    """
    The Action class
    """
    def __init__ (self, action):
        assert (action in ['H', 'S'])
        self.action = action


class GameEndError(Exception):
    """
    When game ends at the first draw
    """
    def __init__ (self, message, payload):
        assert (payload in ['draw', 'lose', 'win'])
        self.message = message
        self.payload = payload

    def __str__(self):
        return str(self.message)

    
class Simulator:
    """
    The simulator
    """
    def __init__ (self):
        """
        Initialize
        """
        # Ensure replicability
        np.random.seed(0)

    def reset (self):
        """
        Reset the simulator
        """
        player_card, player_suite = self.draw()
        dealer_card, dealer_suite = self.draw()
        state = State (player_card, player_suite,
            dealer_card, dealer_suite)

        # Raise custom errors if the game finished due to
        # one or more players getting a negative card
        if (player_suite == 'R' and dealer_suite == 'R'):
            raise GameEndError("Game is Draw", 'draw')
        elif (player_suite == 'R'):
            raise GameEndError("Dealer has won", 'lose')
        elif (dealer_suite == 'R'):
            raise GameEndError("Player has won", 'win')
        
        return state

    def draw (self):
        """
        Draw a card
        """
        if np.random.random() < 1 / 3:
            suite = 'R'
        else:
            suite = 'B'

        card = min(int(np.random.random() * 10 + 1), 10)
        return card, suite

    def step (self, state : State, action : str):
        """
        Perform action a
        """
        assert (action in ['H', 'S'])
        if action == 'S':
            dealer_sum = self.__play_delear(state)
            player_sum = state.max_safe_sum()

            if dealer_sum < 0 or dealer_sum > 31:
                # Bust dealer
                return state, 1, True
            else:
                # Dealer safe
                if (player_sum > dealer_sum):
                    return state, 1, True
                elif (player_sum == dealer_sum):
                    return state, 0, True
                else:
                    return state, -1, True
        else:
            # Draw a card and update
            card, suite = self.draw()
            state.update_state (card, suite)
            player_sum = state.max_safe_sum()
            if (player_sum < 0 or player_sum > 31):
                # Bust player
                return state, -1, True
            else:
                # Safe player
                return state, 0, False

    def __play_delear(self, state : State):
        """
        Play as dealer until death
        """
        print ("Playing as dealer")
        dealer_sum = state.max_safe_sum(dealer=True)
        while (0 <= dealer_sum < 25):
            # Keep hitting
            card, suite = self.draw()
            state.update_state (card, suite, dealer=True)
            dealer_sum = state.max_safe_sum(dealer=True)

        return dealer_sum


if __name__ == '__main__':
    """
    Testing the simulator
    """
    sim = Simulator()

    time = 0
    state = sim.reset()
    reward, done = 0, False
    print (time)
    print (state)

    while not done:
        print(' ')
        time += 1
        state, reward, done = sim.step(state, 'H')
        print (time)
        print (state)
        print (reward, done)
