"""
Using monte-carlo method for policy evaluation
"""

from tqdm import tqdm
from environment.simulator import Simulator, GameEndError
from qpitables import *


def montecarlo (sim, PItable, first_visit=False, num_episodes=100):
    """
    Given a policy pi, runs monte-carlo method
    and do policy-evaluation to compute the 
    value of the policy, until the value converges
    """

    # Generate an empty q table
    Qtable = create_q_table ()

    VisitCount = create_q_table ()

    for e in range(num_episodes):
        episode = generate_episode (sim, PItable)
        states, final_reward = episode

        for idx, state in enumerate(states):
            if (first_visit and state in states[:idx]):
                continue

            old_value = index_table(Qtable, state)
            old_count = index_table(VisitCount, state)

            # Update the values
            new_count = old_count + 1
            new_value = old_count * old_value / new_count + final_reward / new_count

            modify_q_table (Qtable, state, new_value)
            modify_q_table (VisitCount, state, new_count)

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
    runs = 1000
    num_episodes = 1000 # Number of episodes in each run
    PItable = create_pi_table ()
    plot='plots/MC' # Directory where to save plots
    firstVisit = True

    # Visualizing the policy
    plot_PItable (PItable, title='Monte-Carlo policy', path=plot, name='MC Policy')
    
    Qtable = create_q_table()
    print ("Running %d runs" % runs)
    for r in tqdm(range(runs)):
        table_r = (montecarlo (sim, PItable, num_episodes=num_episodes))

        Qtable = (r / (r+1)) * Qtable + (1 / (r+1)) * table_r

    name = 'MC -- %d RUNS -- %d EPISODES -- First Visit - %s' % (runs, num_episodes, ('YES' if firstVisit else 'FASLE'))
    plot_Qtable(Qtable, title=name, path=plot, name=name)
