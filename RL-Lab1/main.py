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


def initTaurTransitionMatrix(still = False):
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
def getTaurActions(state):
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
    actions.append('still') # TODO uncomment when he can stand still
    return actions

# returns value and a flag indicating end state
def reward(state, minotaur_state, action):
    if action == 'null':
        return 0
    if state == minotaur_state: # eaten
        return -100
    #if state == minotaur_state - 1 or state == minotaur_state + 1 or state == minotaur_state - 6 or state == minotaur_state + 6: # close to minotaur
    #     return -10, 0
    if state == 28: # winning
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
    return min_pos + 30*pos


# i varje steg, räkna ut V för varje state du kan befinna dig i, där V(s) = r(s) + värdet för den action som ger oss högst reward.
# man räknar alltså ut värdet för alla actions du kan få utifrån det state du är i just nu.

def valueIteration(T=15):
    no_states = 30
    end_state = 28
    V = [{key: [0, 0] for key in range(no_states**2)} for i in
         range(T)]  # T stycken Value dictionaries:( keys: states, values a list = [value, action])
    #V[0][end_state] = [100, 'null']    # define terminal state, V[0] is last step!!!

    for t in range(T):
        v_list = V[t]   # values for all states at timestep 0 (which is last timestep)
        # must accumulate for each timestep!!
        for s in range(no_states):
            for m in range(no_states): # sätt minotaur på varenda ställe
                # taur_actions = getTaurActions(m)
                actions = getActions(s, m)
                min_actions = getTaurActions(m)
                r = {} # collect rewards for each action here {key: action, value: value}
                for a in actions:
                    resulting_state = getState(s, a)
                    #r[a] = 0
                    #for m_a in min_actions:
                     #   res_min = getState(m, m_a)
                    r[a] = V[t - 1][stateTable(resulting_state, m)][0] if t > 0 else 0
                    r[a] += reward(s, m, a) # current reward

                best_value = max(r.values())
                best_actions = [key for (key, value) in r.items() if value == best_value]
                best_action = best_actions[0] # TODO save all actions?

                v_list[stateTable(s, m)] = [best_value, best_action]

        V[t] = v_list

    return V


def getPolicy(V):
    T = len(V) - 1
    pi = [0 for i in range(T)]
    state_progression = [0 for i in range(T+1)]
    state = 0
    p = V[::-1]
    for t in range(T):
        action = p[t][state][1]
        state = getState(state, action)
        state_progression[t+1] = state
        pi[t] = action

    return pi, state_progression

def createTaurPath(T):
    taur_path = []# np.zeros((T, 1))
    #taur_path[0, 0] = 28
    taur_path.append(28)
    for t in range(T-1):
        state = taur_path[t]
        actions = getTaurActions(state)
        a = actions[rnd.randint(0, len(actions)-1)]
        #taur_path[t + 1, 0] = getState(state, a)
        taur_path.append(getState(state, a))
       # print(taur_path)

    return taur_path

def simulate(V):
    T = len(V)-1
    state_progression = []
    pi = []
    taur_path = createTaurPath(T + 1)
    state_progression.append(0)
    p = V[::-1]
    for t in range(T):
        state = int(state_progression[t])
        taur_state = int(taur_path[t])
        action = p[t][stateTable(state, taur_state)][1]
        next_state = getState(state, action)
        state_progression.append(next_state)
        pi.append(action)

    print(taur_path)
    return state_progression, pi, taur_path




def bellman(trans_matrix, T=15):
    end_state = 29
    # pi = -np.ones((1,T))
    pi = np.zeros((1, 30))
    state = end_state
    u = -np.ones((1, 30))
    u_prev = 0
    # for t in range(T):
    for s in range(30):
        A = getActions(state, trans_matrix)  # define available actions, do we need taur_pos as arg?
        u_temp = 0
        state_curr = state
        print('....')
        for action in A:
            pos_state = getState(state_curr, action)
            print(pos_state)

            r = reward(pos_state, action)
            # u(0,pos_state) = r + u_prev
            print(u)

            if u > u_temp:
                u_temp = u(0, pos_state)
                state = pos_state

        u_prev += u_temp
    # pi[0,t] = state
    pi[0, s] = state
    return pi


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
       # print(xt[i], yt[i])
        plt.plot([xt[i], xt[i + 1]], [yt[i], yt[i + 1]], 'ro-')
        plt.draw()

def drawPolicy(V):
    w = 5
    h = 4
    offset = 0.4
    min_pos = 9
    step = 14
    xt, yt = transformState(min_pos)

    for i in range(30):
        actions = V[step][stateTable(i, min_pos)][1]
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
        plt.draw()
    plt.plot(xt, yt, 'ro') #print minotaur position
    plt.draw()

def drawValueFunction(V):
    w = 5
    h = 4
    min_pos = 28
    step = 14
    xt, yt = transformState(min_pos)

    for i in range(30):
        value = V[step][stateTable(i, min_pos)][0]
        x, y = transformState(i)
        offset = 0.2
        plt.text(x-offset, y, round(value, 2))
        plt.draw()
    plt.plot(xt, yt, 'ro') #print minotaur position
    plt.draw()


def transformPath(path, w = 6):
    x = [(x % w) + 0.5 for x in path]
    y = [4 - math.floor(y/w) + 0.5 for y in path]
    return x,y

def transformState(i, w = 6):
    x = (i % w) + 0.5
    y = 4 - math.floor(i/w) + 0.5
    return x,y


def bellman2(taur_transition_mtx, T=15):
    n_states = 30
    V = [{key: [0, 0] for key in range(n_states ** 2)} for i in range(T)]

    for t in range(T):
        for i in range(n_states):
            V[t][stateTable(i, i)] = [0, 'null']
            V[t][stateTable(28, i)] = [100, 'null']

    for t in range(1, T):
        v_temp = V[t]
        for s in range(n_states):
            for m in range(n_states):
                actions = getActions(s, m)
                if actions[0] != 'null':
                    taur_actions = getTaurActions(m)
                    r = {}
                    for a in actions:
                        res_s = getState(s, a)
                        r[a] = reward(s, m, a)
                        for a_m in taur_actions:
                            res_m = getState(m, a_m)
                            r[a] += taur_transition_mtx[m, res_m]*V[t-1][stateTable(res_s, res_m)][0]
                    best_value = max(r.values())
                    best_actions = [key for (key, value) in r.items() if value == best_value]
                    best_action = best_actions
                    v_temp[stateTable(s, m)] = [best_value, best_action]
        V[t] = v_temp

    return V


def main():
    transition_mtx = initTransitionMatrix()
    taur_transition_mtx = initTaurTransitionMatrix(True)
    # initTransitionMatrix()
    # pi = bellman(trans_matrix)
    # print(pi)
    # pi, path = getPolicy(V)
    #print(path)

    V = bellman2(taur_transition_mtx)
    our_path, pi, taur_path = simulate(V)
    print(pi)
    print(our_path)
    drawLabyrinth()
    drawValueFunction(V)
    #drawPath(our_path, taur_path)
    #drawPolicy(V)
    plt.show()


main()
