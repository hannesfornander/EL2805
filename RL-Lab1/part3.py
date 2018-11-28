import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math


def initTransitionMatrix(n_states=16, still=False):
    transition_mtx = np.zeros((n_states, n_states))

    right_list = [3, 7, 11, 15]  # Cannot go right from these states
    left_list = [0, 4, 8, 12]  # Cannot go left from these states
    up_list = [0, 1, 2, 3]  # Cannot go up from these states
    down_list = [12, 13, 14, 15]  # Cannot go down from these states

    for i in range(n_states):
        for j in range(n_states):
            if j == i and still:
                transition_mtx[i, j] = 1
            elif (j == i + 1) and (i not in right_list):
                transition_mtx[i, j] = 1
            elif (j == i - 1) and (i not in left_list):
                transition_mtx[i, j] = 1
            elif (j == i - 6) and (i not in up_list):
                transition_mtx[i, j] = 1
            elif (j == i + 6) and (i not in down_list):
                transition_mtx[i, j] = 1

    for i in range(transition_mtx.shape[0]):
        transition_mtx[i] /= np.sum(transition_mtx[i])

    return transition_mtx


def getActions(state):
    right_list = [3, 7, 11, 15]  # Cannot go right from these states
    left_list = [0, 4, 8, 12]  # Cannot go left from these states
    up_list = [0, 1, 2, 3]  # Cannot go up from these states
    down_list = [12, 13, 14, 15]  # Cannot go down from these states

    actions = []  # right, left, up ,down
    if state not in right_list:
        actions.append('right')
    if state not in left_list:
        actions.append('left')
    if state not in up_list:
        actions.append('up')
    if state not in down_list:
        actions.append('down')
    actions.append('still')
    return actions


def getPoliceActions(state):
    right_list = [3, 7, 11, 15]  # Cannot go right from these states
    left_list = [0, 4, 8, 12]  # Cannot go left from these states
    up_list = [0, 1, 2, 3]  # Cannot go up from these states
    down_list = [12, 13, 14, 15]  # Cannot go down from these states

    actions = []  # right, left, up ,down
    if state not in right_list:
        actions.append('right')
    if state not in left_list:
        actions.append('left')
    if state not in up_list:
        actions.append('up')
    if state not in down_list:
        actions.append('down')
    return actions


def getState(state, action):
    if action == 'right':
        state += 1
    elif action == 'left':
        state -= 1
    elif action == 'up':
        state -= 4
    elif action == 'down':
        state += 4
    return state


def stateTable(pos, pos_p):
    n_states = 16
    return pos_p + n_states*pos


def ind(s, a, st, at):
    if s == st and a == at:
        return 1
    else:
        return 0

def getPos(st):
    return np.unravel_index(st, (16, 16))


def getQMax(Q, st, action):
    pos_p, pos = getPos(st)
    next_pos = getState(pos,action)
    actions = getActions(next_pos)
    Q_list = []
    for b in range(len(actions)):
        #pos_next = getState(pos, actions[b])
        st_next = stateTable(next_pos, pos_p)
        Q_list.append(Q[st_next, b])
    return max(Q_list)


def reward(s):
    if s % 17 == 0:
        return -10
    elif 16 * 5 < s or s < 16 * 6 - 1:
        return 1
    else:
        return 0

def QLearning():
    n_states = 16
    n_actions_max = 5
    lmbda = 0.8

    Q = np.zeros((n_states**2, n_actions_max))
    ind_mtx = np.zeros((n_states**2, n_actions_max))
    st_list = []
    at_list = []

    t = 0
    max_t = 10000000
    Q_plot = np.zeros((max_t, 1))
    st = 15
    while t < max_t:
        Q_temp = Q
        pos_p, pos = getPos(st)
        actions = getActions(pos)
        #at = np.argmax(Q[st,0:len(actions)])
        #print(at)
        at = rnd.randint(0, len(actions)-1)
        ind_mtx[st, at] += 1
        n = ind_mtx[st, at]
        alpha = 1 / (n**(2/3))

        Q_max = getQMax(Q, st, actions[at])
        Q_temp[st, at] = Q[st, at] + alpha * (reward(st) + lmbda*Q_max - Q[st, at])

        Q = Q_temp
       # states[s] = stateTable(getState(pos,actions[at]), pos_p)
        st = stateTable(getState(pos,actions[at]), pos_p)
        Q_plot[t] = max(Q[15])
        t += 1
        if t % 100000 == 0:
            print("Almost done: ", t/100000, "%")

    print(Q)
    return Q_plot

def SARSA():
    pass


def main():
    np.set_printoptions(threshold = np.nan)
    LEARNING_TYPE = 'Q_LEARNING' # 'SARSA'

    if LEARNING_TYPE == 'Q_LEARNING':
        Q_plot = QLearning()
        plt.plot(Q_plot)
        plt.show()
    elif LEARNING_TYPE == 'SARSA':
        SARSA()


main()
