"""
Comparing different algorithms
"""

from tqdm import tqdm
import time
from qpitables import *
from sarsa import sarsa_rewards
from qlearning import qlearning_rewards
from tdlambda import tdlambda_rewards


def plot_num_episodes ():
    num_episodes = 100001
    interval = 1000
    decay = 10000
    # num_episodes = 10001
    # interval = 500
    # decay = 1000

    sarsa = sarsa_rewards (k = 1, num_episodes=num_episodes,
                            interval=interval)
    sarsa_decay = sarsa_rewards (k = 1, num_episodes=num_episodes,
                            interval=interval, decay=decay)
    qlearn = qlearning_rewards (num_episodes=num_episodes, interval=interval)
    tdlambda = tdlambda_rewards (num_episodes=num_episodes, interval=interval)
    tdlambda_decay = tdlambda_rewards (num_episodes=num_episodes, interval=interval, decay=decay)
    x = np.array([e for e in range(num_episodes) if e % interval == 0])

    X = [x] * 5
    Y = [sarsa, sarsa_decay, qlearn, tdlambda, tdlambda_decay]
    labels = ['SARSA', 'SARSA_DECAY', 'QLEARN', 'TD Lambda', 'TD Lambda Decay']

    plot_curves (X, Y, labels, title='Rate of Learning Comparision',
                xlabel='Number of episodes', ylabel='Average test reward',
                name='Algorithms %d' % num_episodes, path='plots', show=True)

def plot_alphas ():
    num_episodes = 10001
    interval = num_episodes - 1
    decay = 1000

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    sarsa = []; sarsa_decay = []; qlearn = []; tdlambda = []; tdlambda_decay = []

    for alpha in alphas:
        sarsa.append( sarsa_rewards (alpha=alpha, k = 1, num_episodes=num_episodes,
                                    interval=interval) [-1] )
        sarsa_decay.append( sarsa_rewards (alpha=alpha, k = 1, num_episodes=num_episodes,
                                          interval=interval, decay=decay)[-1] )
        qlearn.append( qlearning_rewards (alpha=alpha, num_episodes=num_episodes, interval=interval)[-1] )
        tdlambda.append( tdlambda_rewards (alpha=alpha, num_episodes=num_episodes, interval=interval)[-1] )
        tdlambda_decay.append( tdlambda_rewards (alpha=alpha, num_episodes=num_episodes, interval=interval, decay=decay)[-1] )

    X = [alphas] * 5
    Y = [sarsa, sarsa_decay, qlearn, tdlambda, tdlambda_decay]
    labels = ['SARSA', 'SARSA_DECAY', 'QLEARN', 'TD Lambda', 'TD Lambda Decay']

    plot_curves (X, Y, labels, title='Alphas',
                xlabel='Alpha Value', ylabel='Average test reward',
                name='alphas', path='plots', show=True)

if __name__ == '__main__':
    # plot_alphas()
    plot_num_episodes()