"""
Comparing different algorithms
"""

from tqdm import tqdm
import time
from qpitables import *
from sarsa import sarsa_rewards
from qlearning import qlearning_rewards
from tdlambda import tdlambda_rewards

if __name__ == '__main__':
    # num_episodes = 100001
    # interval = 1000
    # decay = 10000
    num_episodes = 10001
    interval = 500
    decay = 1000

    sarsa = sarsa_rewards (k = 1, num_episodes=num_episodes,
                            interval=interval)
    sarsa_decay = sarsa_rewards (k = 1, num_episodes=num_episodes,
                            interval=interval, decay=decay)
    qlearn = qlearning_rewards (num_episodes=num_episodes, interval=interval)
    tdlambda = tdlambda_rewards (num_episodes=num_episodes, interval=interval)
    x = np.array([e for e in range(num_episodes) if e % interval == 0])

    X = [x] * 4
    Y = [sarsa, sarsa_decay, qlearn, tdlambda]
    labels = ['SARSA', 'SARSA_DECAY', 'QLEARN', 'TD Lambda']

    plot_curves (X, Y, labels, title='Comparision',
                xlabel='Number of episodes', ylabel='Average test reward',
                name='Algorithms', path='plots', show=True)