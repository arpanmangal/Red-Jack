"""
Using q-learning for policy control
"""

from tqdm import tqdm
from environment.simulator import Simulator, GameEndError
from qpitables import *

def qlearning (sim, alpha=0.1, num_episodes=1000, interval=100, epsilon=0.1):
    """
    Learn the optimal policy using Q-Learning
    """
    # Creating a q(s,a) table
    QSAtable = create_qsa_table()

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

    def greedy (state):
        hVal = index_qsa_table('H', QSAtable, state)
        sVal = index_qsa_table('S', QSAtable, state)
        return 'H' if hVal >= sVal else 'S'

    total_reward = 0
    rewards = []
    for e in tqdm(range(num_episodes)):
        if e % interval == 0:
            rewards.append(total_reward / (e + 1))
            plot_QSAtable(QSAtable, show=True)

        S = generate_initial_state ()
        state = S.state_rep()
        done = False

        while not done:
            A = epsilon_greedy(state)

            # Take action
            new_S, reward, done = sim.step(S, A)
            total_reward += reward

            G = reward
            if not done:
                new_state = new_S.state_rep()
                a = greedy(new_state)
                G +=  index_qsa_table(a, QSAtable, new_state)

            old_value = index_qsa_table(A, QSAtable, state)
            new_value = old_value + alpha * (G - old_value)

            modify_qsa_table (A, QSAtable, state, new_value)

            if not done:
                # Update
                state = new_state; S = new_S

    return np.array(rewards)


if __name__ == '__main__':
    sim = Simulator()
    r = qlearning (sim, num_episodes=1000001, interval=50000)
    # r = qlearning (sim, num_episodes=10001, interval=1000)
    print (r)