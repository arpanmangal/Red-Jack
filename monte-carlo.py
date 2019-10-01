"""
Using monte-carlo method for policy evaluation
"""

import copy
from environment.simulator import Simulator, GameEndError
from qpitables import *


def montecarlo (PItable, first_visit=False, num_episodes=5000, progess_check=1000,
                plot=None, EPSILON=1e-5):
    """
    Given a policy pi, runs monte-carlo method
    and do policy-evaluation to compute the 
    value of the policy, until the value converges

    progess_check: Check convergence after each 100 iterations
    plot: Path of folder to save the plots
    EPSILON: used for convergence checking
    """

    # Visualizing the policy
    plot_PItable (PItable, title='Monte-Carlo policy', path=plot, name='MC Policy')

    # Create the simulator
    sim = Simulator()

    # Generate an empty q table
    Qtable = create_q_table ()
    # Qtable_prev = copy.deepcopy(Qtable) # Used for convergence testing later

    VisitCount = create_q_table ()

    episode_number = 0
    while True:
        episode = generate_episode (sim, PItable)
        # states, actions, rewards = episode
        states, final_reward = episode
        # final_reward = rewards[-1]

        for state in states:
            old_value = index_table(Qtable, state)
            old_count = index_table(VisitCount, state)

            # Update the values
            new_count = old_count + 1
            new_value = old_count * old_value / new_count + final_reward / new_count

            modify_q_table (Qtable, state, new_value)
            modify_q_table (VisitCount, state, new_count)

        episode_number += 1  
        if (episode_number % progess_check == 0):
            print ("Episode Number: ", episode_number)
            plot_Qtable (Qtable, title="Monte-Carlo -- episode %s" % episode_number,
                         path=plot, name='MC Value -- episode %s' % episode_number)
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

    montecarlo (PItable, plot='plots/MC', num_episodes=500000, progess_check=50000)
    # sim = Simulator ()

    # for e in range (10):
    #     # Create episode
    #     print (generate_episode(sim ,PItable))
    #     print()