"""
Using SARSA for policy control
"""

from tqdm import tqdm
import time
from environment.simulator import Simulator, GameEndError
from qpitables import *

def sarsa (sim, k=1, alpha=0.1, num_episodes=100, interval=100, initial_epsilon=0.1, decay=False):
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

    def epsilon_greedy (state):
        hVal = index_qsa_table('H', QSAtable, state)
        sVal = index_qsa_table('S', QSAtable, state)

        if (np.random.rand() > epsilon):
            return 'H' if hVal >= sVal else 'S'
        else:
            return 'S' if hVal > sVal else 'H'

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

        state_actions = []

        t = 0
        s = generate_initial_state()
        state = s.state_rep()
        a = epsilon_greedy(state)
        done = False
        state_actions.append((state, a))

        while not done:
            # Take action
            s, reward, done = sim.step(s, a)
            t += 1

            # Perform update
            def update(old_idx, G):
                old_state, old_a = state_actions[old_idx]#[t-k]
                old_value = index_qsa_table(old_a, QSAtable, old_state)
                new_value = old_value + alpha * (G - old_value)

                modify_qsa_table(old_a, QSAtable, old_state, new_value)
            
            if done:
                G = reward

                # update
                past = k
                while (past > 0 and t - past >= 0):
                    update (t - past, G)
                    past -= 1
            else:
                # Sample another action
                state = s.state_rep()
                a = epsilon_greedy (state)
                state_actions.append((state, a))
                G = index_qsa_table(a, QSAtable, state)

                # Update
                if (t - k >= 0):
                    update (t - k, G)

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

    # # Running Parameters
    # num_episodes = 100001 # Number of episodes in each run
    # interval = 1000

    return sarsa (sim, k=k, alpha=alpha, num_episodes=num_episodes,
                 interval=interval, initial_epsilon=epsilon, decay=decay)

if __name__ == '__main__':
    # print (sarsa_rewards(num_episodes=10001, interval=100, decay=1000))
    num_episodes = 100001
    interval = 100
    X = []; Y = []; labels = []
    x = np.array([e for e in range(num_episodes) if e % interval == 0])

    for k in [1, 3]:
        Y.append( sarsa_rewards(k=k, num_episodes=num_episodes, interval=interval) )
        X.append(x)
        labels.append('k = %d' % k)

    plot_curves (X, Y, labels, title='SARSA learning with different k',
                xlabel='Number of episodes', ylabel='Average test reward',
                name='K-dist', path='plots/SARSA', show=True)

