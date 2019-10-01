"""
Contains structures for creating and interacting with
q(s, a) and pi(s) tables
"""

import numpy as np
import os
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def create_q_table ():
    def __empty_q_table ():
        return np.zeros((4, 32, 10))

    return __empty_q_table()


def create_pi_table ():
    def __empty_pi_table ():
        PItable = np.chararray((4, 32, 10), unicode=True)
        PItable[:] = 'H' # Initialised with hit

        return PItable

    PItable = __empty_pi_table ()
    PItable[:,25:,:] = 'S'

    return PItable


def index_table(Table, state):
    """
    Index into a Qtable or PItable
    """
    table_no, dealer_card, player_sum = state

    assert (0 <= table_no <= 3 and 1 <= dealer_card <= 10 and 0 <= player_sum <= 31)

    return Table[table_no][player_sum][dealer_card - 1]


def modify_q_table(Qtable, state, value):
    """
    state == table_no, dealer_card, player_sum
    1 <= dealer_card <= 10
    0 <= player_sum <= 31
    """
    table_no, dealer_card, player_sum = state

    assert (0 <= table_no <= 3 and 1 <= dealer_card <= 10 and 0 <= player_sum <= 31)

    Qtable[table_no][player_sum][dealer_card - 1] = value
    
    return Qtable


def modify_pi_table(PItable, state, action):
    """
    state == table_no, dealer_card, player_sum
    1 <= dealer_card <= 10
    player_sum <= 31
    """
    table_no, dealer_card, player_sum = state

    assert action in ['H', 'S']
    assert (0 <= table_no <= 3 and 1 <= dealer_card <= 10 and 0 <= player_sum <= 31)

    PItable[table_no][player_sum][dealer_card - 1] = action

    return PItable


# def has_converged (Qtable1, Qtable2, epsilon):
#     q_table1, qc_table1 = Qtable1
#     q_table2, qc_table2 = Qtable2

#     return np.sum(abs(q_table1 - q_table2)) + np.sum(abs(qc_table1 - qc_table2)) < epsilon


def plot_Qtable (Qtable, title='', path=None, name='State Value'):
    X = np.array( list(range(1,11))*32 ).reshape(32, 10)
    Y = np.array( list(range(0,32))*10 ).reshape(10, 32).T

    fig = plt.figure(figsize=(18.0, 15.0))
    def plot_3d (X, Y, Z, title, loc=111):
        ax = fig.add_subplot(loc, projection='3d')
        ax.plot_wireframe(X, Y, Z)
        
        ax.set_xlabel('Dealer Card')
        ax.set_ylabel('Player max. sum')
        ax.set_zlabel('V')
        ax.set_zticks([-1, 0, 1])
        ax.set_zlim([-2, 2])
        ax.set_title(title)

    plot_3d (X, Y, Qtable[0], title="0 Special Cards Used", loc=221)
    plot_3d (X, Y, Qtable[1], title="1 Special Cards Used", loc=222)
    plot_3d (X, Y, Qtable[2], title="2 Special Cards Used", loc=223)
    plot_3d (X, Y, Qtable[3], title="3 Special Cards Used", loc=224)

    plt.suptitle(title)

    if path is not None:
        plt.savefig(os.path.join(path, name), dpi=100)
    plt.show()

def plot_PItable (PItable, title='', path=None, name='Policy'):
    X = np.array( list(range(1,11))*32 ).reshape(32, 10)
    Y = np.array( list(range(0,32))*10 ).reshape(10, 32).T

    fig = plt.figure(figsize=(18.0, 15.0))
    def plot_3d (X, Y, Z, title, loc=111):
        ax = fig.add_subplot(loc, projection='3d')
        ax.plot_wireframe(X, Y, Z)
        
        ax.set_xlabel('Dealer Card')
        ax.set_ylabel('Player max. sum')
        ax.set_zlabel('Pi')
        ax.set_zticks([-1, 0, 1])
        ax.set_zlim([-2, 2])
        ax.set_title(title)

    def int_table (pi_table):
        meaning = lambda a: 0 if a == 'H' else 1
        flattened = np.array(list(map(meaning, pi_table.flatten())))
        return flattened.reshape(32,10)

    plot_3d (X, Y, int_table(PItable[0]), title="0 Special Cards Used", loc=221)
    plot_3d (X, Y, int_table(PItable[1]), title="1 Special Cards Used", loc=222)
    plot_3d (X, Y, int_table(PItable[2]), title="2 Special Cards Used", loc=223)
    plot_3d (X, Y, int_table(PItable[3]), title="3 Special Cards Used", loc=224)

    plt.suptitle(title)

    if path is not None:
        plt.savefig(os.path.join(path, name), dpi=100)
    plt.show()

if __name__ == '__main__':
    # Test the table generation
    # Qtable = create_q_table()
    PItable = create_pi_table()
    plot_PItable (PItable)
    # modify_q_table (Qtable, (0, 1, 26), 0.5)
    # modify_pi_table (PItable, (0, 1, 26), 'H')
    # print (Qtable)
    # plot_Qtable (Qtable)

    # exit(0)

