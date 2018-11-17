import numpy as np
import matplotlib.pyplot as plt
import random as rnd


def initBoard():
    board_mtx = np.zeros((30, 30))

    right_list = [1, 5, 7, 9, 11, 13, 15, 17, 23, 27, 29]  # Cannot go right from these states
    left_list = [0, 2, 6, 8, 10, 12, 14, 16, 18, 24, 28]  # Cannot go left from these states
    up_list = [0, 1, 2, 3, 4, 5, 16, 17, 25, 26, 27, 28]  # Cannot go up from these states
    down_list = [10, 11, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29]  # Cannot go down from these states

    for i in range(30):
        for j in range(30):
            if j == i:
                board_mtx[i, j] = 1
            elif (j == i + 1) and (i not in right_list):
                board_mtx[i, j] = 1
            elif (j == i - 1) and (i not in left_list):
                board_mtx[i, j] = 1
            elif (j == i - 6) and (i not in up_list):
                board_mtx[i, j] = 1
            elif (j == i + 6) and (i not in down_list):
                board_mtx[i, j] = 1
    return board_mtx


# return action and next state due to this
def getActions(state):
    right_list = [1, 5, 7, 9, 11, 13, 15, 17, 23, 27, 29]  # Cannot go right from these states
    left_list = [0, 2, 6, 8, 10, 12, 14, 16, 18, 24, 28]  # Cannot go left from these states
    up_list = [0, 1, 2, 3, 4, 5, 16, 17, 25, 26, 27, 28]  # Cannot go up from these states
    down_list = [10, 11, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29]  # Cannot go down from these states
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


def reward(state, action):
    if state == 2:
        return 100
    if action != 'still':
        return 10
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


# i varje steg, räkna ut V för varje state du kan befinna dig i, där V(s) = r(s) + värdet för den action som ger oss högst reward.
# man räknar alltså ut värdet för alla actions du kan få utifrån det state du är i just nu.

def valueIteration(T=15):
    no_states = 30
    end_state = 28
    V = [{key: [-1, 0] for key in range(no_states)} for i in
         range(T)]  # T stycken Value dictionaries:( keys: states, values a list = [value, action])
    V[0][end_state] = [100, 'still']  # define terminal state, V[0] is last step!!!

    for t in range(T):
        v_list = V[t]  # values for all states at timestep 0 (which is last timestep)
        # must accumulate for each timestep!!
        for s in range(no_states):
            actions = getActions(s)
            r = {} # collect rewards for each action here
            for a in actions:
                resulting_state = getState(s, a)
                r[a] = V[t - 1][resulting_state][0] if t > 0 else 0
                r[a] += reward(s, a) # current reward TODO kontrollera det ackumulerade värdet!
            best_value = max(r.values())
            if best_value != 0:
                best_actions = [key for (key, value) in r.items() if value == best_value]
                best_action = best_actions[0] # if there are two choices with equal values
            else:
                best_action = 'null'
            v_list[s] = [best_value, best_action]
        V[t] = v_list

    return V


def getPolicy(V):
    T = len(V)
    pi = [0 for i in range(T)]
    state_progression = [0 for i in range(T+1)]
    state = 0
    p = V[::-1]
    for t in range(T):
        action = p[t][state][1]
        print(action)
        print(state)
        state = getState(state, action)
        state_progression[t+1] = state
        pi[t] = action

    return pi, state_progression


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
def drawPath(our_path, taur_path = []):
    w = 5
    h = 4
    # our_path = [[rnd.randint(0, w)+0.5, rdm.randint(0, h)+0.5] for i in range(T)]
   # taur_path = [[rnd.randint(0, w) + 0.5, rdm.randint(0, h) + 0.5] for i in range(len(our_path))]

    x, y = transformPath(our_path)
    #print(x,y)
    for i in range(len(our_path) - 1):
        plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], 'o-')
       # print(x[i], y[i])
       # plt.plot([xt[i], xt[i + 1]], [yt[i], yt[i + 1]], 'r-')
        plt.draw()


def transformPath(path, w = 6):
    x = [(x % w) + 0.5 for x in path]
    y = [4 - round(y/w) + 0.5 for y in path] # TODO double-check this
    return x,y



def main():
    trans_matrix = initBoard()
    # print(trans_matrix)
    # initTransitionMatrix()
    # pi = bellman(trans_matrix)
    # print(pi)
    V = valueIteration()
    pi, path = getPolicy(V)
    print(path)



    drawLabyrinth()
    drawPath(path)
    plt.show()


main()
