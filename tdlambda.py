"""
Using forward view eligibility trace
or, TD-Lambda algorithm

Implemented in a efficient O(episode length) way, using the fact
that reward is 0 at each step, except the terminal step
"""

from tqdm import tqdm
import time
from environment.simulator import Simulator, GameEndError
from qpitables import *

def tdlambda (sim, l=0.5, alpha=0.5, num_episodes=1001, interval=100, initial_epsilon=0.1, decay=False):
    """
    Learn optimal policy using TD-Lambda
    """
    epsilon = initial_epsilon

    # Creating a q(s,a) table
    QSAtable = create_qsa_table()

    def generate_initial_state ():
        try:
            state = sim.reset()
            return state
        except GameEndError:
            return generate_initial_state ()

    def generate_episode ():
        """
        Generate an episode based on current QSA values
        """
        def epsilon_greedy (state):
            hVal = index_qsa_table('H', QSAtable, state)
            sVal = index_qsa_table('S', QSAtable, state)

            if (np.random.rand() > epsilon):
                return 'H' if hVal >= sVal else 'S'
            else:
                return 'S' if hVal > sVal else 'H'

        states = []
        state = generate_initial_state ()
        done = False

        while not done:
            state_rep = state.state_rep()
            action = epsilon_greedy (state_rep)
            states.append ((state_rep, action))

            # Take the action
            state, reward, done = sim.step(state, action)

        return states, reward
                

    rewards = []
    for e in tqdm(range(num_episodes)):
        if e % interval == 0:
            PItable = derive_pi_table (QSAtable)
            rewards.append(play_game(sim, PItable))
            # plot_QSAtable(QSAtable, show=True)

        # Decay epsilon
        if decay:
            epsilon = initial_epsilon / (e // decay + 1)

        states, final_reward = generate_episode ()
        assert (len(states) > 0)
        G = np.zeros(len(states))

        G[-1] = final_reward
        for i in range(len(states) - 1, 0, -1):
            state, action = states[i]
            qi = index_qsa_table(action, QSAtable, state)
            Gi = G[i]
            G[i-1] = (1 - l) * qi + l * Gi

        # Perform the updates
        for state, Gt in zip(states, G):
            state, action = state
            old_value = index_qsa_table(action, QSAtable, state)
            new_value = old_value + alpha * (Gt - old_value)
            modify_qsa_table (action, QSAtable, state, new_value)

    return np.array(rewards)



def play_game (sim, PItable, num_games=1000):
    """
    Given a policy, play multiple games and return average reward
    """

    def generate_initial_state ():
        try:
            state = sim.reset()
            return state
        except GameEndError:
            return generate_initial_state ()
            
    total_reward = 0
    for g in range(num_games):
        state = generate_initial_state ()
        done = False
    
        while not done:
            s = state.state_rep()
            action = index_table(PItable, s)

            # Take action
            state, reward, done = sim.step(state, action)

        total_reward += reward

    return total_reward / num_games


def tdlambda_rewards (l=0.5, alpha=0.1, epsilon=0.1,
                      num_episodes=10001, interval=1000, decay=None):
    # Create the simulator
    sim = Simulator()

    return tdlambda(sim, l=l, alpha=alpha, initial_epsilon=epsilon,
                     num_episodes=num_episodes, interval=interval, decay=decay)


if __name__ == '__main__':
    print (tdlambda_rewards())