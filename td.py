"""
Using Temporal Difference method for policy evaluation
"""

import copy
from environment.simulator import Simulator, GameEndError
from qpitables import *


def temporaldifference (PItable, k=1, alpha=0.1, decay=None, num_episodes=500, progess_check=100,
                plot=None, EPSILON=1e-5):
    """
    Given a policy pi, runs TD(k) method
    and do policy-evaluation to compute the 
    value of the policy, until the value converges

    decay: array of episode_number, to decrease alpha by 2 at these points
    progess_check: Check convergence after each 100 iterations
    plot: Path of folder to save the plots
    EPSILON: used for convergence checking
    """

    # Visualizing the policy
    plot_PItable (PItable, title='Temporal-Difference policy', path=plot, name='TD Policy')

    # Create the simulator
    sim = Simulator()

    # Generate an empty q table
    Qtable = create_q_table ()
    # Qtable_prev = copy.deepcopy(Qtable) # Used for convergence testing later

    episode_number = 0
    while True:
        episode = generate_episode (sim, PItable)
        # states, actions, rewards = episode
        states, final_reward = episode
        # final_reward = rewards[-1]

        for idx, state in enumerate(states):
            if idx + k < len(states):
                G = index_table(Qtable, states[idx+k])
            else:
                G = final_reward

            old_value = index_table(Qtable, state)
            new_value = old_value + alpha * (G - old_value)

            # Update the values
            modify_q_table (Qtable, state, new_value)

        episode_number += 1
        if decay is not None and episode_number in decay:
            alpha = alpha / 2 
        if (episode_number % progess_check == 0):
            print ("Episode Number: ", episode_number)
            plot_Qtable (Qtable, title="Temporal-Difference -- episode %s" % episode_number,
                         path=plot, name='TD Value -- episode %s' % episode_number)
            if (episode_number >= num_episodes):
            # if has_converged (Qtable, Qtable_prev, EPSILON):
                break
            # else:
                # Qtable_prev = copy.deepcopy(Qtable)


def generate_episode (sim, PItable):
    """
    Generate a series of episodes using the policy from the policy table pi
    """

    states = []
    # actions = []
    # rewards = []

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
        # actions.append(action)

        # Take action
        state, reward, done = sim.step(state, action)
        # rewards.append(reward)

    assert (len(states) > 0)

    # return (states, actions, rewards)
    return states, reward


if __name__ == '__main__':
    PItable = create_pi_table ()

    temporaldifference (PItable, plot='plots/TD', k=1000, alpha=0.1, decay=[10000, 100000], num_episodes=200000, progess_check=40000)
    # sim = Simulator ()

    # for e in range (10):
    #     # Create episode
    #     print (generate_episode(sim ,PItable))
    #     print()