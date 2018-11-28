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


def getQMax(Q, st):
    actions = getActions(st)
    Q_list = []
    for b in range(len(actions)):
        st_next = getState(st, actions[b])
        Q_list.append(Q[st_next, b])
    return max(Q_list)


def reward(s):
    if s % 17 == 0:
        return -10
    elif s % 16 == 5:
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
    st = 0
    while t < 100000000:
        Q_temp = Q
        #st, at = np.unravel_index(np.argmax(Q), Q.shape)
        actions = getActions(st)
        at = rnd.randint(0,len(actions)-1)
        ind_mtx[st, at] += 1
        n = ind_mtx[st, at]
        alpha = 1 / (n**(2/3))

        Q_max = getQMax(Q, st)
        Q_temp[st, at] = Q[st, at] + alpha * (reward(st) + lmbda*Q_max - Q[st, at])

        Q = Q_temp
        t += 1
        st = getState(st,actions[at])
        if t % 100000 == 0:
            print(st)
            print("Almost done: ", t/1000000, "%")

    print(Q)

def SARSA():
    pass


def main():
    np.set_printoptions(threshold = np.nan)
    LEARNING_TYPE = 'Q_LEARNING' # 'SARSA'

    if LEARNING_TYPE == 'Q_LEARNING':
        QLearning()
    elif LEARNING_TYPE == 'SARSA':
        SARSA()


main()
