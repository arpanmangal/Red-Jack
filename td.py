"""
Using Temporal Difference method for policy evaluation
"""

from tqdm import tqdm
import time
from environment.simulator import Simulator, GameEndError
from qpitables import *

def temporaldifference (sim, PItable, k=1, alpha=0.1, decay=None, num_episodes=100):
    """
    Given a policy pi, runs TD(k) method
    and do policy-evaluation to compute the 
    value of the policy, until the value converges
    """

    # Visualizing the policy
    plot_PItable (PItable, title='Temporal-Difference policy', path=plot, name='TD Policy')

    # Generate an empty q table
    Qtable = create_q_table ()

    for e in range(num_episodes):
        episode = generate_episode (sim, PItable)
        states, final_reward = episode

        for idx, state in enumerate(states):
            if idx + k < len(states):
                G = index_table(Qtable, states[idx+k])
            else:
                G = final_reward

            old_value = index_table(Qtable, state)
            new_value = old_value + alpha * (G - old_value)

            # Update the values
            modify_q_table (Qtable, state, new_value)

    return Qtable


def generate_episode (sim, PItable):
    """
    Generate a series of episodes using the policy from the policy table pi
    """

    states = []

    def generate_initial_state ():
        try:
            state = sim.reset()
            return state
        except GameEndError:
            return generate_initial_state ()

    state = generate_initial_state ()
    done = False
   
    while not done:
        s = state.state_rep()
        action = index_table(PItable, s)
        states.append(s)

        # Take action
        state, reward, done = sim.step(state, action)

    assert (len(states) > 0)

    return states, reward


if __name__ == '__main__':
    # Create the simulator
    sim = Simulator()

    # Running Parameters
    runs = 100
    num_episodes = 1000 # Number of episodes in each run
    alpha = 0.1
    k = 3
    PItable = create_pi_table ()
    plot='plots/TD' # Directory where to save plots

    # Visualizing the policy
    plot_PItable (PItable, title='TD policy', path=plot, name='TD Policy')

    def run_td (K):
        # Run the TD learning for k values in K
        runs = 100
        num_episodes = 1000
        k = K
        alpha = 0.1

        Qtable = create_q_table()
        Qshape = Qtable.shape
        print ("Running %d runs" % runs)
        for r in tqdm(range(runs)):
            Qtable = (r / (r+1)) * Qtable + (1 / (r+1)) * temporaldifference (sim, PItable, alpha=alpha, k=k, num_episodes=num_episodes)
            assert (Qtable.shape == Qshape)
            # Take a deep breath
            time.sleep(1)

        name = 'TD -- %d RUNS -- %d EPISODES -- k = %d' % (runs, num_episodes, K)
        plot_Qtable(Qtable, title=name, path=plot, name=name)

    Ks = [1, 3, 5, 10, 100, 1000]
    for K in Ks:
        print ("Running for K = %d" % K)
        run_td (K)