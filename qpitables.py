"""
Contains structures for creating and interacting with
q(s, a) and pi(s) tables
"""

import numpy as np

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


def plot_q_table (Qtable):
    # q_table, qc_table = Qtable
    print (Qtable)

def plot_pi_table (PItable):
    # pi_table, pic_table = PItable
    print (PItable)


if __name__ == '__main__':
    # Test the table generation
    Qtable = create_q_table()
    PItable = create_pi_table()
    print (Qtable)
    print ()
    print (PItable)
    print (has_converged(Qtable, Qtable, 1e-5))

    modify_q_table (Qtable, (3, 1, 26), 0.5)
    modify_pi_table (PItable, (3, 1, 26), 'H')

    print (Qtable)
    print ()
    print (PItable)
    print (has_converged(Qtable, Qtable, 1e-5))

    print (index_table(PItable, (3, 1, 26)))

