"""
Module defining Player functions
Player state = [min_score, isTen, isTwenty, isThirty]
"""

def create (card, suite):
    """
    Create a player state
    """
    __valid_card (card, suite)
    sum = card * __suite_score (suite)
    softs = __soft_vector (card, suite)
    return [sum] + softs 


def next_state (state, card, suite):
    """
    Update the state
    Inplace operation !
    """
    __valid_card (card, suite)
    state[0] += card * __suite_score (suite)
    state[1:] = __update_soft_vector (state[1:], card, suite)
    return state


def get_full_state (state):
    """
    Generate the possible totals you can have given a state
    """
    count = sum([1 for s in state[1:] if s == True])
    increments = [10 * c for c in range(count + 1)]
    return [state[0] + inc for inc in increments]


def printable_state (state):
    """
    String to pring the state
    """
    return str(state[0]) + " | " + str(state[1:]) + " || " + str(get_full_state(state))


# Helper functions
def __valid_card (card, suite):
    """
    The card should belong to 1-10
    with suite either 'B' or 'R'
    """
    assert (1 <= card <= 10)
    assert (suite in ['B', 'R'])


def __valid_action (action):
    """
    Action should be either 'H' or 'S'
    """
    assert (action in ['H', 'S'])


def __suite_score (suite):
    if suite == 'B':
        return 1
    else:
        return -1 

    
def __soft_vector (card, suite):
    if (suite == 'R'):
        return [False] * 3
    else:
        v = [False] * 3
        if (1 <= card <= 3):
            v[card - 1] = True
        return v


def __update_soft_vector (soft, card, suite):
    # inplace!
    if (suite == 'R'):
        return soft
    else:
        if (1 <= card <= 3):
            soft[card - 1] = True
        return soft
        