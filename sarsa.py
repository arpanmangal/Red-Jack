"""
Using SARSA for policy control
"""

from tqdm import tqdm
import time
from environment.simulator import Simulator, GameEndError
from qpitables import *

def sarsa (sim, k=1, alpha=0.1, num_episodes=1001, interval=100, initial_epsilon=0.1, decay=False):
    """
    Learns and an optimal policy PI using SARSA
    """
    epsilon = initial_epsilon
    assert (0 <= epsilon <= 1)

    # Empty Q(s, a) table
    QSAtable = create_qsa_table ()

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
        if (e % interval == 0):
            PItable = derive_pi_table (QSAtable)
            rewards.append(play_game(sim, PItable))
            # Visualize
            # name = 'SARSA -- %d EPISODE' % e
            # plot_QSAtable(QSAtable, show=True)

        # Decay epsilon
        if decay:
            epsilon = initial_epsilon / (e // decay + 1)

        states, final_reward = generate_episode()
        assert (len(states)) > 0

        for idx, state in enumerate(states):
            if idx + k < len(states):
                future_s, future_a = states[idx+k]
                G = index_qsa_table(future_a, QSAtable, future_s)
            else:
                G = final_reward

            s, a = state
            old_value = index_qsa_table (a, QSAtable, s)
            new_value = old_value + alpha * (G - old_value)

            # Update the values
            modify_qsa_table (a, QSAtable, s, new_value)

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


def sarsa_rewards(k=1, alpha=0.1, epsilon=0.1,
                  num_episodes=10001, interval=100, decay=None):
    # Create the simulator
    sim = Simulator()

    return sarsa (sim, k=k, alpha=alpha, num_episodes=num_episodes,
                 interval=interval, initial_epsilon=epsilon, decay=decay)

if __name__ == '__main__':
    # print (sarsa_rewards(num_episodes=10001, interval=100, decay=1000))
    num_episodes = 100001
    interval = 10000
    X = []; Y = []; labels = []
    x = np.array([e for e in range(num_episodes) if e % interval == 0])

    for k in [1, 3, 10, 100]:
        Y.append( sarsa_rewards(k=k, num_episodes=num_episodes, interval=interval) )
        X.append(x)
        labels.append('k = %d' % k)

    plot_curves (X, Y, labels, title='SARSA learning with different k',
                xlabel='Number of episodes', ylabel='Average test reward',
                name='K-dist', path='plots/SARSA', show=True)

