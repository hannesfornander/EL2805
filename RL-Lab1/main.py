import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math


def initTransitionMatrix():
    transition_mtx = np.zeros((30, 30))

    right_list = [1, 5, 7, 9, 11, 13, 15, 17, 23, 27, 29]  # Cannot go right from these states
    left_list = [0, 2, 6, 8, 10, 12, 14, 16, 18, 24, 28]  # Cannot go left from these states
    up_list = [0, 1, 2, 3, 4, 5, 16, 17, 25, 26, 27, 28]  # Cannot go up from these states
    down_list = [10, 11, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29]  # Cannot go down from these states

    for i in range(30):
        for j in range(30):
            if j == i:
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


def initTaurTransitionMatrix(still=False):
    transition_mtx = np.zeros((30, 30))

    right_list = [5, 11, 13, 17, 23, 29]  # Cannot go right from these states
    left_list = [0, 6, 8, 12, 18, 24]  # Cannot go left from these states
    up_list = [0, 1, 2, 3, 4, 5]  # Cannot go up from these states
    down_list = [24, 25, 26, 27, 28, 29]  # Cannot go down from these states

    for i in range(30):
        for j in range(30):
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


# return action and next state due to this
def getActions(state, min):
    right_list = [1, 5, 7, 9, 11, 13, 15, 17, 23, 27, 28, 29]  # Cannot go right from these states
    left_list = [0, 2, 6, 8, 10, 12, 14, 16, 18, 24, 28]  # Cannot go left from these states
    up_list = [0, 1, 2, 3, 4, 5, 16, 17, 25, 26, 27, 28]  # Cannot go up from these states
    down_list = [10, 11, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29]  # Cannot go down from these states
    actions = []  # right, left, up ,down
    if state == 28 or state == min:
        return ['null']
    if state not in right_list:
        actions.append('right')
    if state not in left_list:
        actions.append('left')
    if state not in up_list:
        actions.append('up')
    if state not in down_list:
        actions.append('down')
    if state != 28 and state != min:
        actions.append('still')
    return actions


# Possible actions for minotaur
def getTaurActions(state, still_flag):
    right_list = [5, 11, 17, 23, 29]  # Cannot go right from these states
    left_list = [0, 6, 12, 18, 24]  # Cannot go left from these states
    up_list = [0, 1, 2, 3, 4, 5]  # Cannot go up from these states
    down_list = [24, 25, 26, 27, 28, 29]  # Cannot go down from these states
    actions = []  # right, left, up ,down
    if state not in right_list:
        actions.append('right')
    if state not in left_list:
        actions.append('left')
    if state not in up_list:
        actions.append('up')
    if state not in down_list:
        actions.append('down')
    if still_flag:
        actions.append('still')
    return actions


# returns value and a flag indicating end state
def reward(state, minotaur_state, action):
    if action == 'null':
        return 0
    if state == minotaur_state:  # eaten
        return -100
    # if state == minotaur_state - 1 or state == minotaur_state + 1 or state == minotaur_state - 6 or state == minotaur_state + 6: # close to minotaur
    #     return -10, 0
    if state == 28:  # winning
        return 100
    if action != 'still':
        return 0
    else:
        return 0


def getState(state, action):
    if action == 'right':
        state += 1
    elif action == 'left':
        state -= 1
    elif action == 'up':
        state -= 6
    elif action == 'down':
        state += 6
    return state


def stateTable(pos, min_pos):
    return min_pos + 30 * pos


def getPolicy(V):
    T = len(V) - 1
    pi = [0 for i in range(T)]
    state_progression = [0 for i in range(T + 1)]
    state = 0
    p = V[::-1]
    for t in range(T):
        action = p[t][state][1]
        state = getState(state, action)
        state_progression[t + 1] = state
        pi[t] = action

    return pi, state_progression


def createTaurPath(T, still_flag):
    taur_path = []  # np.zeros((T, 1))
    # taur_path[0, 0] = 28
    taur_path.append(28)
    for t in range(T - 1):
        state = taur_path[t]
        actions = getTaurActions(state, still_flag)
        a = actions[rnd.randint(0, len(actions) - 1)]
        # taur_path[t + 1, 0] = getState(state, a)
        taur_path.append(getState(state, a))
    # print(taur_path)

    return taur_path


def simulate(pi, still_flag):
    T = len(pi) - 1
    state_progression = []
    pi_sim = []
    taur_path = createTaurPath(T + 1, still_flag)
    state_progression.append(0)
    p = pi[::-1]
    for t in range(T):
        state = int(state_progression[t])
        taur_state = int(taur_path[t])
        action = p[t][stateTable(state, taur_state)]
        action = action[0]
        next_state = getState(state, action)
        state_progression.append(next_state)
        pi_sim.append(action)

    return state_progression, pi_sim, taur_path


def simulateInf(pi, itr):
    success = 0
    avg_pathlength = 0
    for i in range(itr):
        taur_state = 28
        state = 0
        path_length = 0
        while state != 28 and state != taur_state:
            action = pi[stateTable(state, taur_state)]
            action = action[rnd.randint(0, len(action) - 1)]
            state = getState(state, action)
            taur_actions = getTaurActions(taur_state, False)
            taur_action = taur_actions[rnd.randint(0, len(taur_actions) - 1)]
            taur_state = getState(taur_state, taur_action)
            path_length += 1
        if state == 28:
            success += 1
        avg_pathlength += path_length / itr
    return success, avg_pathlength


def drawLabyrinth(w=6, h=5):
    plt.xlim(0, w)  # define labyrinth width
    plt.ylim(0, h)  # define labyrinth height
    plt.grid(True)

    plt.plot([2, 2], [2, 5], 'k-', lw=4)
    plt.plot([4, 4], [2, 4], 'k-', lw=4)
    plt.plot([4, 6], [3, 3], 'k-', lw=4)
    plt.plot([4, 4], [0, 1], 'k-', lw=4)
    plt.plot([1, 5], [1, 1], 'k-', lw=4)


# best probably to let it finish simulating first, save the path, then draw the path to the plot
def drawPath(our_path, taur_path):
    w = 5
    h = 4
    # our_path = [[rnd.randint(0, w)+0.5, rdm.randint(0, h)+0.5] for i in range(T)]
    # taur_path = [[rnd.randint(0, w) + 0.5, rdm.randint(0, h) + 0.5] for i in range(len(our_path))]

    x, y = transformPath(our_path)
    xt, yt = transformPath(taur_path)
    for i in range(len(our_path) - 1):
        plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], 'bo-')
        #plt.text(x[i]-0.2, y[i]-0.2, i )
        # print(xt[i], yt[i])
        plt.plot([xt[i], xt[i + 1]], [yt[i], yt[i + 1]], 'ro-')
        #plt.text(xt[i]+0.2, yt[i]+0.2, i)
        text = "Our path: " + str(our_path) + "\nMinotaur path: "+ str(taur_path)
        plt.xlabel(text)
        plt.draw()


def drawPolicy(pi):
    w = 5
    h = 4
    offset = 0.3
    min_pos = 28
    step = 11
    xt, yt = transformState(min_pos)

    for i in range(30):
        actions = pi[stateTable(i, min_pos)]  # add step dimension when finite time horizon
        for action in actions:
            x, y = transformState(i)
            if action == 'up':
                plt.arrow(x, y, dx=0, dy=offset, head_width=0.1, head_length=0.1)
            if action == 'down':
                plt.arrow(x, y, dx=0, dy=-offset, head_width=0.1, head_length=0.1)
            if action == 'right':
                plt.arrow(x, y, dx=offset, dy=0, head_width=0.1, head_length=0.1)
            if action == 'left':
                plt.arrow(x, y, dx=-offset, dy=0, head_width=0.1, head_length=0.1)
            if action == 'still':
                plt.plot(x, y, 'ko')
        plt.draw()
    plt.plot(xt, yt, 'ro')  # print minotaur position
    plt.draw()


def drawValueFunction(V):
    w = 5
    h = 4
    min_pos = 28
    step = 14
    xt, yt = transformState(min_pos)

    for i in range(30):
        value = V[stateTable(i, min_pos)]
        x, y = transformState(i)
        offset = 0.2
        plt.text(x - offset, y, round(value, 2))
        plt.draw()
    plt.plot(xt, yt, 'ro')  # print minotaur position
    plt.draw()


def transformPath(path, w=6):
    x = [(x % w) + 0.5 for x in path]
    y = [4 - math.floor(y / w) + 0.5 for y in path]
    return x, y


def transformState(i, w=6):
    x = (i % w) + 0.5
    y = 4 - math.floor(i / w) + 0.5
    return x, y


def bellman(taur_transition_mtx, T, still_flag):
    n_states = 30
    V = [{key: 0 for key in range(n_states ** 2)} for i in range(T)]
    pi = [{key: '' for key in range(n_states ** 2)} for i in range(T)]

    for t in range(T):
        for i in range(n_states):
            V[t][stateTable(i, i)] = 0
            V[t][stateTable(28, i)] = 100
            pi[t][stateTable(i, i)] = 'null'
            pi[t][stateTable(28, i)] = 'null'

    for t in range(1, T):
        v_temp = V[t]
        for s in range(n_states):
            for m in range(n_states):
                actions = getActions(s, m)
                if actions[0] != 'null':
                    taur_actions = getTaurActions(m, still_flag)
                    r = {}
                    for a in actions:
                        res_s = getState(s, a)
                        r[a] = reward(s, m, a)
                        for a_m in taur_actions:
                            res_m = getState(m, a_m)
                            r[a] += taur_transition_mtx[m, res_m] * V[t - 1][stateTable(res_s, res_m)]
                    best_value = max(r.values())
                    best_actions = [key for (key, value) in r.items() if value == best_value]
                    best_action = best_actions
                    v_temp[stateTable(s, m)] = best_value
                    pi[t][stateTable(s, m)] = best_action
        V[t] = v_temp

    return V, pi


def policyIteration(taur_transition_mtx, still_flag):
    n_states = 30
    V = [0 for i in range(n_states ** 2)]
    pi = ['' for i in range(n_states ** 2)]
    pi_prev = ['' for i in range(n_states ** 2)]
    # print(V)
    lmbda = 1 - (1 / 30)
    for i in range(n_states):
        V[stateTable(i, i)] = 0
        V[stateTable(28, i)] = 100
        pi[stateTable(i, i)] = 'null'
        pi[stateTable(28, i)] = 'null'

    while (pi != pi_prev):
        pi_prev[:] = pi[:]
        v_temp = V
        for s in range(n_states):
            for m in range(n_states):
                actions = getActions(s, m)
                if actions[0] != 'null':
                    taur_actions = getTaurActions(m, still_flag)
                    r = {}
                    for a in actions:
                        res_s = getState(s, a)
                        r[a] = reward(s, m, a)
                        for a_m in taur_actions:
                            res_m = getState(m, a_m)
                            r[a] += lmbda * taur_transition_mtx[m, res_m] * V[stateTable(res_s, res_m)]
                    best_value = max(r.values())
                    best_actions = [key for (key, value) in r.items() if value == best_value]
                    best_action = best_actions
                    v_temp[stateTable(s, m)] = best_value
                    pi[stateTable(s, m)] = best_action
        V = v_temp
    return V, pi

def plotProb(V, T):
    plot_v = []
    for t in range(T):
        plot_v.append(V[t][stateTable(0, 28)] / 100)
    plt.plot(plot_v)
    plt.grid(True)
    plt.xlim(0, T)
    plt.ylim(0, 1.1)

def main():
    # change in drawValue, drawPolicy, bellman and policyIteration so that actions and step works
    still_flag = False
    T = 50
    itr = 10000
    taur_transition_mtx = initTaurTransitionMatrix(still_flag)
    V, pi = bellman(taur_transition_mtx, T, still_flag)
   # V, pi = policyIteration(taur_transition_mtx, still_flag)
    our_path, pi_sim, taur_path = simulate(pi, still_flag)


    #drawLabyrinth()
    #drawValueFunction(V)
    #drawPath(our_path, taur_path)
    #drawPolicy(pi)
    #plotProb(V,T)
    plt.show()

    success, avg_length = simulateInf(pi, itr)
    print(success / itr)
    print(avg_length)


main()
