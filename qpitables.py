"""
Contains structures for creating and interacting with
q(s, a) and pi(s) tables
"""

import numpy as np
import os
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm


def create_q_table ():
    def __empty_q_table ():
        return np.zeros((4, 32, 10))

    Qtable = __empty_q_table()
    Qtable[:,31,:] = 1
    return Qtable


def create_pi_table ():
    def __empty_pi_table ():
        PItable = np.chararray((4, 32, 10), unicode=True)
        PItable[:] = 'H' # Initialised with hit

        return PItable

    PItable = __empty_pi_table ()
    PItable[:,25:,:] = 'S'

    return PItable


def create_qsa_table ():
    Htable = create_q_table()
    Stable = create_q_table()
    Htable[:,:,:] = Stable[:,:,:] = 0

    return np.array([Htable, Stable])


def derive_pi_table (QSAtable):
    """
    Making the policy from the q(s, a) table
    """
    # Optimize it!!
    PItable = create_pi_table()
    Htable = QSAtable[0]; Stable = QSAtable[1]
    for table_no in range(4):
        for player_sum in range(32):
            for dealer_card in range(10):
                hval = Htable[table_no][player_sum][dealer_card]
                sval = Stable[table_no][player_sum][dealer_card]
                PItable[table_no][player_sum][dealer_card] = 'H' if hval > sval else 'S'

    return PItable


def index_table(Table, state):
    """
    Index into a Qtable or PItable
    """
    table_no, dealer_card, player_sum = state

    assert (0 <= table_no <= 3 and 1 <= dealer_card <= 10 and 0 <= player_sum <= 31)

    return Table[table_no][player_sum][dealer_card - 1]


def index_qsa_table (action, Table, state):
    assert (action in ['H', 'S'])
    Htable = Table[0]; Stable = Table[1]
    if action == 'H':
        return index_table(Htable, state)
    else:
        return index_table(Stable, state)


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


def modify_qsa_table(action, QSAtable, state, value):
    assert (action in ['H', 'S'])

    if action == 'H':
        modify_q_table(QSAtable[0], state, value)
    else:
        modify_q_table(QSAtable[1], state, value)

    return QSAtable


# def has_converged (Qtable1, Qtable2, epsilon):
#     q_table1, qc_table1 = Qtable1
#     q_table2, qc_table2 = Qtable2

#     return np.sum(abs(q_table1 - q_table2)) + np.sum(abs(qc_table1 - qc_table2)) < epsilon

def plot_3d (fig, Z, title, zlabel='V', loc=111):
    X = np.array( list(range(1,11))*32 ).reshape(32, 10)
    Y = np.array( list(range(0,32))*10 ).reshape(10, 32).T

    ax = fig.add_subplot(loc, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.RdYlGn,
                    linewidth=0, antialiased=False,
                    vmin=-1, vmax=1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax.set_xlabel('Dealer Card')
    ax.set_ylabel('Player max. sum')
    ax.set_zlabel(zlabel)
    ax.set_zticks([-1, 0, 1])
    ax.set_zlim([-2, 2])
    ax.set_title(title)    

def plot_Qtable (Qtable, title='', path=None, name='State Value', show=False):
    fig = plt.figure(figsize=(18.0, 15.0))

    plot_3d (fig, Qtable[0], title="0 Special Cards Used", loc=221)
    plot_3d (fig, Qtable[1], title="1 Special Cards Used", loc=222)
    plot_3d (fig, Qtable[2], title="2 Special Cards Used", loc=223)
    plot_3d (fig, Qtable[3], title="3 Special Cards Used", loc=224)

    plt.suptitle(title)

    if path is not None:
        plt.savefig(os.path.join(path, name), dpi=100)
    if show:
        plt.show()
    plt.close('all')


def plot_PItable (PItable, title='', path=None, name='Policy', show=False):
    def int_table (pi_table):
        meaning = lambda a: -1 if a == 'H' else 1
        flattened = np.array(list(map(meaning, pi_table.flatten())))
        return flattened.reshape(32,10)

    fig = plt.figure(figsize=(15.0, 8.0))
    def plot_grid(Z, title, loc=111):
        Z = Z.T
        ax = fig.add_subplot(loc)
        # make a color map of fixed colors
        cmap = mpl.colors.ListedColormap(['r','lawngreen'])
        bounds=[-6,0,6]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # tell imshow about color map so that only set colors are used
        img = ax.imshow(Z,interpolation='nearest',
                            cmap = cmap,norm=norm)
        ax.set_title(title)
        ax.set_xlabel('Player max sum')
        ax.set_xticks(list(range(0, 32)))#  np.arange(0, 1, step=0.2))
        ax.set_ylabel('Dealer card')

    # plot_grid(int_table(PItable[0]), title="ldfj")
    plot_grid (int_table(PItable[0]), title="0 Special Cards Used", loc=221)
    plot_grid (int_table(PItable[1]), title="1 Special Cards Used", loc=222)
    plot_grid (int_table(PItable[2]), title="2 Special Cards Used", loc=223)
    plot_grid (int_table(PItable[3]), title="3 Special Cards Used", loc=224)

    plt.suptitle(title)

    if path is not None:
        plt.savefig(os.path.join(path, name), dpi=100)
    if show:
        plt.show()
    plt.close('all')


def plot_QSAtable (QSAtable, title='', path=None, name='QSA table', show=False):
    Htable = QSAtable[0]; Stable = QSAtable[1]

    titleH = title + " -- Hit"; titleS = title + " -- Stick"
    nameH = name + " -- Hit"; nameS = name + " -- Stick"

    # plot_Qtable (Htable, titleH, path, nameH, show=show)
    # plot_Qtable (Stable, titleS, path, nameS, show=show)
    fig = plt.figure(figsize=(20.0, 10.0))

    plot_3d (fig, Htable[0], title="0 Special Cards Used | Hit", loc=241)
    plot_3d (fig, Htable[1], title="1 Special Cards Used | Hit", loc=242)
    plot_3d (fig, Htable[2], title="2 Special Cards Used | Hit", loc=243)
    plot_3d (fig, Htable[3], title="3 Special Cards Used | Hit", loc=244)
    plot_3d (fig, Stable[0], title="0 Special Cards Used | Stick", loc=245)
    plot_3d (fig, Stable[1], title="1 Special Cards Used | Stick", loc=246)
    plot_3d (fig, Stable[2], title="2 Special Cards Used | Stick", loc=247)
    plot_3d (fig, Stable[3], title="3 Special Cards Used | Stick", loc=248)

    plt.suptitle(title)

    if path is not None:
        plt.savefig(os.path.join(path, name), dpi=100)
    if show:
        plt.show()
    plt.close('all')
    

def plot_curves (X=[], Y=[], labels=[], colors=None, 
                 xlabel='', ylabel='', title='',
                 name='', path=None, show=False):
    for x, y, l in zip(X, Y, labels):
        plt.plot(x, y, label=l)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if path is not None:
        plt.savefig(os.path.join(path, name), dpi=100)
    if show:
        plt.show()
    plt.close('all')

if __name__ == '__main__':
    # Test the table generation
    # Qtable = create_q_table()
    # PItable = create_pi_table()
    # plot_PItable (PItable)
    # modify_q_table (Qtable, (0, 1, 26), 0.5)
    # modify_pi_table (PItable, (0, 1, 26), 'H')
    # print (Qtable)
    # plot_Qtable (Qtable)

    # exit(0)

    x = [1,3,7]
    y = [1,9,49]
    plot_curves([x], [y])

