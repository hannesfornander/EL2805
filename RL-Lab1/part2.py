import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math


def initPoliceTransitionMatrix():
    n_states = 18
    transition_mtx = np.zeros((n_states, n_states, n_states))

    for pos in range(n_states):
        for pos_p in range(n_states):
            police_actions = getPoliceActions(pos, pos_p)
            for a_p in police_actions:
                pos_p_next = getNewPos(pos_p, a_p)
                transition_mtx[pos, pos_p, pos_p_next] = 1/len(police_actions)

    return transition_mtx


def getActions(pos, pos_p):
    right_list = [5, 11, 17]  # Cannot go right from these states
    left_list = [0, 6, 12]  # Cannot go left from these states
    up_list = [0, 1, 2, 3, 4, 5]  # Cannot go up from these states
    down_list = [12, 13, 14, 15, 16, 17]  # Cannot go down from these states
    actions = []  # right, left, up ,down
    if pos != pos_p:
        if pos not in right_list:
            actions.append('right')
        if pos not in left_list:
            actions.append('left')
        if pos not in up_list:
            actions.append('up')
        if pos not in down_list:
            actions.append('down')
        actions.append('still')
    else:
        actions.append('reset')
    return actions



def getPoliceActions(pos, pos_p):
    right_list = [5, 11, 17]  # Cannot go right from these states
    left_list = [0, 6, 12]  # Cannot go left from these states
    up_list = [0, 1, 2, 3, 4, 5]  # Cannot go up from these states
    down_list = [12, 13, 14, 15, 16, 17]  # Cannot go down from these states
    actions = []  # right, left, up ,down

    if pos != pos_p:
        x, y = getMapCoord(pos)
        x_p, y_p = getMapCoord(pos_p)
        if pos_p not in right_list and not y < y_p:
            actions.append('right')
        if pos_p not in left_list and not y > y_p:
            actions.append('left')
        if pos_p not in up_list and not x > x_p:
            actions.append('up')
        if pos_p not in down_list and not x < x_p:
            actions.append('down')
    else:
        actions.append('reset_p')
    return actions


def getMapCoord(pos):
    return np.unravel_index(pos, (3, 6))


def reward(pos, pos_p):
    if pos == pos_p:
        return -50
    elif pos == 0 or pos == 5 or pos == 12 or pos == 17:
        return 10
    else:
        return 0


def getNewPos(pos, action):
    if action == 'right':
        pos += 1
    elif action == 'left':
        pos -= 1
    elif action == 'up':
        pos -= 6
    elif action == 'down':
        pos += 6
    elif action == 'reset':
        pos = 0
    elif action == 'reset_p':
        pos = 8
    return pos



def stateTable(pos, pos_p):
    return pos_p + 18*pos


def drawLabyrinth(w=6, h=3):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlim(0, w)  # define labyrinth width
    ax.set_ylim(0, h)  # define labyrinth height
    ax.set_xticks(np.arange(0, 6))
    ax.set_yticks(np.arange(0, 3))


def transformState(i, w = 6):
    x = (i % w) + 0.5
    y = 2 - math.floor(i/w) + 0.5
    return x, y


def drawPolicy(pi):
    w = 5
    h = 2
    offset = 0.3
    pos_p = 8
    n_states = 18
    xt, yt = transformState(pos_p)

    for i in range(n_states):
        actions = pi[stateTable(i, pos_p)] # add step dimension when finite time horizon
        for action in actions:
            x, y = transformState(i)
            if action == 'up':
                plt.arrow(x, y, dx = 0, dy = offset, head_width = 0.1, head_length= 0.1)
            if action == 'down':
                plt.arrow(x, y, dx = 0, dy = -offset, head_width = 0.1, head_length= 0.1)
            if action == 'right':
                plt.arrow(x, y, dx= offset, dy=0, head_width = 0.1, head_length= 0.1)
            if action == 'left':
                plt.arrow(x, y, dx= -offset, dy=0, head_width = 0.1, head_length= 0.1)
            if action == 'still':
                plt.plot(x, y, 'ko')
        plt.draw()
    plt.plot(xt, yt, 'ro') #print minotaur position
    plt.draw()

def drawValueFunction(V):
    w = 5
    h = 2
    pos_p = 8
    n_states = 18

    xt, yt = transformState(pos_p)

    for i in range(n_states):
        value = V[stateTable(i, pos_p)]
        x, y = transformState(i)
        offset = 0.2
        plt.text(x-offset, y, round(value, 2))
        plt.draw()
    plt.plot(xt, yt, 'ro')
    plt.draw()


def policyIteration(police_transition_mtx, lmbda):
    n_states = 18
    V = [0 for i in range(n_states**2)]
    pi = ['' for i in range(n_states**2)]
    pi_prev = ['p' for i in range(n_states ** 2)]

    while pi != pi_prev:
        pi_prev[:] = pi[:]
        v_temp = V
        for pos in range(n_states):
            for pos_p in range(n_states):
                actions = getActions(pos, pos_p)
                police_actions = getPoliceActions(pos, pos_p)
                r = {}
                for a in actions:
                    pos_next = getNewPos(pos, a)
                    r[a] = reward(pos, pos_p)
                    for a_p in police_actions:
                        pos_p_next = getNewPos(pos_p, a_p)
                        r[a] += lmbda*police_transition_mtx[pos_next, pos_p, pos_p_next]*V[stateTable(pos_next, pos_p_next)]
                best_value = max(r.values())
                best_actions = [key for (key, value) in r.items() if value == best_value]
                best_action = best_actions
                v_temp[stateTable(pos, pos_p)] = best_value
                pi[stateTable(pos, pos_p)] = best_action
        V = v_temp

    return V, pi


def main():
    police_transition_mtx = initPoliceTransitionMatrix()

    precision = 0.01
    lmbdas = [precision*i for i in range(int(1/precision))]
    print(lmbdas)

    n_states = 18
    V_best = [0 for i in range(n_states**2)]
    pi_best = ['' for i in range(n_states ** 2)]
    lmbda_best = 0
    for lmbda in lmbdas:
        V, pi = policyIteration(police_transition_mtx, lmbda)
        if V[stateTable(0, 8)] > V_best[stateTable(0, 8)]:
            lmbda_best = lmbda
            V_best = V
            pi_best = pi

    print("Best lambda: ", lmbda_best)
    drawLabyrinth()
    #drawPolicy(pi_best)
    drawValueFunction(V_best)
    plt.show()



main()