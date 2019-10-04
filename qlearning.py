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

    rewards = []
    for e in tqdm(range(num_episodes)):
        if e % interval == 0:
            PItable = derive_pi_table (QSAtable)
            rewards.append(play_game(sim, PItable))
            # plot_QSAtable(QSAtable, show=True)

        S = generate_initial_state ()
        state = S.state_rep()
        done = False

        while not done:
            A = epsilon_greedy(state)

            # Take action
            new_S, reward, done = sim.step(S, A)

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


def qlearning_rewards(alpha=0.1, epsilon=0.1, num_episodes=10001, interval=100):
    # Create the simulator
    sim = Simulator()

    return qlearning (sim, alpha=alpha, num_episodes=num_episodes, interval=interval, epsilon=epsilon)

if __name__ == '__main__':
    print (qlearning_rewards())