"""
Using SARSA for policy control
"""

from tqdm import tqdm
import time
from environment.simulator import Simulator, GameEndError
from qpitables import *

def sarsa (sim, k=1, alpha=0.1, num_episodes=100, epsilon=0.1, decay=False):
    """
    Learns and an optimal policy PI using SARSA
    """
    assert (0 <= epsilon <= 1)

    # Empty Q(s, a) table
    QSAtable = create_qsa_table ()
    # print (QSAtable)

    for e in range(1, num_episodes+1):
        # print ("Episode %d" % e)
        if (e % 100 == 0):
            # Visualize
            # print (QSAtable)
            name = 'SARSA -- %d EPISODE' % e
            plot_QSAtable(QSAtable, title=name, name=name, show=True)

        state_actions = []

        t = 0
        s = generate_initial_state(sim)
        state = s.state_rep()
        a = sample_e_greedy_action(QSAtable, state, epsilon)
        done = False
        state_actions.append((state, a))

        while not done:
            # Take action
            s, reward, done = sim.step(s, a)
            t += 1

            # Perform update
            def update(old_idx):
                old_state, old_a = state_actions[old_idx]#[t-k]
                old_value = index_qsa_table(old_a, QSAtable, old_state)
                new_value = old_value + alpha * (G - old_value)

                modify_qsa_table(old_a, QSAtable, old_state, new_value)
            
            if done:
                # print ('finale: ', reward)
                G = reward

                # update
                past = k
                while (past > 0 and t - past >= 0):
                    update (t - past)
                    past -= 1
            else:
                # Sample another action
                state = s.state_rep()
                a = sample_e_greedy_action(QSAtable, state, epsilon)
                state_actions.append((state, a))
                G = index_qsa_table(a, QSAtable, state)

                # Update
                if (t - k >= 0):
                    update (t - k)


        # print (state_actions)

    return QSAtable


def generate_initial_state (sim):
    try:
        state = sim.reset()
        return state
    except GameEndError:
        return generate_initial_state (sim)


def sample_e_greedy_action (QSAtable, state, epsilon):
    hitValue = index_qsa_table ('H', QSAtable, state)
    stickValue = index_qsa_table ('S', QSAtable, state)
    
    mx = 'S' if stickValue >= hitValue else 'H'
    mi = 'H' if mx == 'S' else 'S'

    # return 'S'
    if np.random.rand() >= epsilon:
        a = mx
    else:
        a = mi
    # print ("Choosing %s | epsilon = %f" % (a, epsilon))
    return a

if __name__ == '__main__':
    # Create the simulator
    sim = Simulator()

    # Running Parameters
    # runs = 100
    num_episodes = 1000 # Number of episodes in each run
    alpha = 0.1
    k = 3

    sarsa (sim, k, alpha, num_episodes, epsilon=0.1)
