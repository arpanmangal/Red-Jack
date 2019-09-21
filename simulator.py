"""
Simulator for the game of RedJack
"""

import numpy as np

class PlayerState:
    """
    The representation of one of the Players in the game
    """
    def __init__ (self, card, suite):
        PlayerState.__valid_card (card, suite)
        self.sum = card * PlayerState.__suite_score (suite)
        self.softs = PlayerState.__soft_vector (card, suite)

    def update_state (self, card, suite):
        PlayerState.__valid_card (card, suite)
        self.sum += card * PlayerState.__suite_score (suite)
        self.__update_softs (card, suite)

    def __update_softs (self, card, suite):
        if (suite == 'B'):
            v = list(self.softs)
            if (1 <= card <= 3):
                v[card - 1] = True
            self.softs = tuple(v)

    def gen_full_state (self):
        poss = [0] * 7
        for s in range(3):
            if (self.softs[s]):
                poss[s] = 10
        
        poss[3] = poss[0] + poss[1]
        poss[4] = poss[1] + poss[2]
        poss[5] = poss[2] + poss[0]
        poss[6] = poss[0] + poss[1] + poss[2]

        # Need only three dimensional !!
        # Just 10, 20, 30
        return (self.sum, poss)
            
    @staticmethod
    def __valid_card (card, suite):
        """
        The card should belong to 1-10
        with suite either 'B' or 'R'
        """
        assert (1 <= card <= 10)
        assert (suite in ['B', 'R'])

    @staticmethod
    def __suite_score (suite):
        if suite == 'B':
            return 1
        else:
            return -1

    @staticmethod
    def __soft_vector (card, suite):
        if (suite == 'R'):
            return (False, False, False)
        else:
            v = [False, False, False]
            if (1 <= card <= 3):
                v[card - 1] = True
            return tuple (v)

    def __str__(self):
        """
        Printing the state
        """
        full_state = self.gen_full_state()
        return str(self.sum) + " | " + str(self.softs) + " || " + str(full_state[1])

class State:
    """
    State of the Agent
    """
    def __init__ (self, player_card, player_suite, dealer_card, dealer_suite):
        self.me = PlayerState (player_card, player_suite)
        self.dealer = PlayerState (dealer_card, dealer_suite)

    def update_state (self, card, suite):
        """
        Update the player state
        """
        self.me.update_state (card, suite)

    def max_safe_sum (self):
        """
        The max possible safe sum the player can get in the current state
        """
        expanded_me = self.me.gen_full_state ()
        sums = [(expanded_me[0] + p) for p in expanded_me[1]]
        mx = sums[0]
        print ("Sums: ", sums)
        for s in sums[1:]:
            if mx > 31:
                mx = s
            if s > mx and s < 31:
                mx = s
        return mx

    def __str__(self):
        """
        Printing the state
        """
        return "Player: " + self.me.__str__() + "\nDealer: " + self.dealer.__str__()


class Action:
    """
    The Action class
    """
    def __init__ (self, action):
        assert (action in ['H', 'S'])
        self.action = action


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
        return state

    def draw (self):
        """
        Draw a card
        """
        if np.random.random() > 2 / 3:
            suite = 'R'
        else:
            suite = 'B'

        card = min(int(np.random.random() * 10 + 1), 10)
        print ("Got: ", card, suite)
        return card, suite

    def step (self, state : State, action : str):
        """
        Perform action a
        """
        assert (action in ['H', 'S'])
        if action == 'S':
            dealer_sum = self.__play_delear()
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

    def __play_delear(self):
        """
        Play as dealer until death
        """
        return 27


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
