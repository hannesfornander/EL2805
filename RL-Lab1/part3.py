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


def getQMax(Q, pos_next, pos_p):
    actions = getActions(pos_next)
    n_actions = len(actions)
    st_next = stateTable(pos_next, pos_p)
    return np.max(Q[st_next, 0:n_actions])


def reward(s):
    if s % 17 == 0:
        return -10
    elif 16 * 5 <= s <= 16 * 6 - 1:
        return 1
    else:
        return 0


def QLearning():
    n_states = 16
    n_actions_max = 5
    lmbda = 0.8

    Q = np.zeros((n_states**2, n_actions_max))
    ind_mtx = np.zeros((n_states**2, n_actions_max))

    t = 0
    t_max = 10000000
    Q_plot = np.zeros((t_max, 1))
    st = 15
    while t < t_max:
        Q_temp = Q
        pos_p, pos = getPos(st)
        actions = getActions(pos)
        at = rnd.randint(0, len(actions) - 1)
        ind_mtx[st, at] += 1
        n = ind_mtx[st, at]
        alpha = 1 / (n**(2/3))

        pos_next = getState(pos, actions[at])
        Q_max = getQMax(Q, pos_next, pos_p)
        Q_temp[st, at] = Q[st, at] + alpha * (reward(st) + lmbda*Q_max - Q[st, at])

        Q = Q_temp
        st = stateTable(pos_next, pos_p)
        Q_plot[t] = max(Q[15])
        t += 1
        if t % (t_max/100) == 0:
            print("Almost done: ", 100*t/t_max, "%")

    print(Q)
    return Q_plot


def SARSA():
    n_states = 16
    n_actions_max = 5
    lmbda = 0.8
    eps = 0.1

    Q = np.zeros((n_states ** 2, n_actions_max))
    ind_mtx = np.zeros((n_states ** 2, n_actions_max))

    t = 0
    t_max = 10000000
    Q_plot = np.zeros((t_max, 1))
    st = 15
    while t < t_max:
        Q_temp = Q
        pos_p, pos = getPos(st)
        actions = getActions(pos)
        at = rnd.randint(0, len(actions) - 1)
        ind_mtx[st, at] += 1
        n = ind_mtx[st, at]
        alpha = 1 / (n ** (2 / 3))

        pos_next = getState(pos, actions[at])
        actions_next = getActions(pos_next)
        st_next = stateTable(pos_next, pos_p)
        r = rnd.uniform(0, 1)
        if t == 0 or r < eps:
            at_next = rnd.randint(0, len(actions_next) - 1)
        else:
            at_next = np.argmax(Q[st_next, 0:len(actions_next)])

        Q_temp[st, at] = Q[st, at] + alpha * (reward(st) + lmbda * Q[st_next, at_next] - Q[st, at])

        Q = Q_temp
        st = st_next
        Q_plot[t] = max(Q[15])
        t += 1
        if t % (t_max/100) == 0:
            print("Almost done: ", 100 * t / t_max, "%")

    print(Q)
    return Q_plot


def main():
    np.set_printoptions(threshold = np.nan)
    LEARNING_TYPE = 'SARSA'

    if LEARNING_TYPE == 'QLEARNING':
        Q_plot = QLearning()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale("log", nonposx='clip')
        plt.plot(Q_plot)
        plt.show()
    elif LEARNING_TYPE == 'SARSA':
        Q_plot = SARSA()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale("log", nonposx='clip')
        plt.plot(Q_plot)
        plt.show()

main()
