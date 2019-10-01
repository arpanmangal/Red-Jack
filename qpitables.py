"""
Contains structures for creating and interacting with
q(s, a) and pi(s) tables
"""

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def create_q_table ():
    def __empty_q_table ():
        q_table = np.zeros((4, 10, 10))
        qc_table = np.zeros((4))

        Qtable = (q_table, qc_table)
        return Qtable

    return __empty_q_table()


def create_pi_table ():
    def __empty_pi_table ():
        pi_table = np.chararray((4, 10, 10), unicode=True)
        pi_table[:] = 'H' # Initialised with hit

        pic_table = np.chararray((4), unicode=True)
        pic_table[:] = 'H'

        PItable = (pi_table, pic_table)
        return PItable

    (pi_table, pic_table) = __empty_pi_table ()
    pi_table[:,3:,:] = 'S'

    return (pi_table, pic_table)


def index_table(Table, state):
    """
    Index into a Qtable or PItable
    """
    table_no, dealer_card, player_sum = state
    main, comp = Table

    assert (0 <= table_no <= 3 and 1 <= dealer_card <= 10 and player_sum <= 31)

    if player_sum < 22:
        return comp[table_no]

    return main[table_no][player_sum - 22][dealer_card - 1]


def modify_q_table(Qtable, state, value):
    """
    state == table_no, dealer_card, player_sum
    1 <= dealer_card <= 10
    player_sum <= 31
    """
    table_no, dealer_card, player_sum = state
    q_table, qc_table = Qtable

    assert (0 <= table_no <= 3 and 1 <= dealer_card <= 10 and player_sum <= 31)

    if player_sum < 22:
        qc_table[table_no] = value
    else:
        q_table[table_no][player_sum - 22][dealer_card - 1] = value
    
    return (q_table, qc_table)


def modify_pi_table(PItable, state, action):
    """
    state == table_no, dealer_card, player_sum
    1 <= dealer_card <= 10
    player_sum <= 31
    """
    table_no, dealer_card, player_sum = state
    pi_table, pic_table = PItable

    assert action in ['H', 'S']
    assert (0 <= table_no <= 3 and 1 <= dealer_card <= 10 and player_sum <= 31)

    if player_sum < 22:
        pic_table[table_no] = action
    else:
        pi_table[table_no][player_sum - 22][dealer_card - 1] = action

    return (pi_table, pic_table)


def has_converged (Qtable1, Qtable2, epsilon):
    q_table1, qc_table1 = Qtable1
    q_table2, qc_table2 = Qtable2

    return np.sum(abs(q_table1 - q_table2)) + np.sum(abs(qc_table1 - qc_table2)) < epsilon


def plot_q_table (Qtable, title=''):
    q_table, qc_table = Qtable
    # print (Qtable)
    Z = q_table[0]
    X = np.array( list(range(1,11))*10 ).reshape(10, 10)
    Y = np.array( list(range(22,32))*10 ).reshape(10, 10).T

    # print (X)
    # print (Y)
    # print (Z)
    # print (X.shape, Y.shape, Z.shape)

    fig = plt.figure()
    def plot_3d (X, Y, Z, title, loc=111):
        ax = fig.add_subplot(loc, projection='3d')
        ax.plot_wireframe(X, Y, Z)
        
        ax.set_xlabel('Dealer Card')
        ax.set_ylabel('Player max. sum')
        ax.set_zlabel('V')
        ax.set_zticks([-1, 0, 1])
        ax.set_zlim([-2, 2])
        ax.set_title(title)

    plot_3d (X, Y, q_table[0], title="0 Special Cards Used", loc=221)
    plot_3d (X, Y, q_table[1], title="1 Special Cards Used", loc=222)
    plot_3d (X, Y, q_table[2], title="2 Special Cards Used", loc=223)
    plot_3d (X, Y, q_table[3], title="3 Special Cards Used", loc=224)

    plt.suptitle(title)

    plt.show()

def plot_pi_table (PItable):
    # pi_table, pic_table = PItable
    print (PItable)


if __name__ == '__main__':
    # Test the table generation
    Qtable = create_q_table()
    PItable = create_pi_table()
    modify_q_table (Qtable, (0, 1, 26), 0.5)
    modify_pi_table (PItable, (0, 1, 26), 'H')
    print (Qtable)
    plot_q_table (Qtable)

    exit(0)
    print ()
    print (PItable)
    print (has_converged(Qtable, Qtable, 1e-5))


    print (Qtable)
    print ()
    print (PItable)
    print (has_converged(Qtable, Qtable, 1e-5))

    print (index_table(PItable, (3, 1, 26)))

