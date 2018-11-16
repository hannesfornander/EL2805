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


def getActions(state, trans_matrix):
    right_list = [1, 5, 7, 9, 11, 13, 15, 17, 23, 27, 29]  # Cannot go right from these states
    left_list = [0, 2, 6, 8, 10, 12, 14, 16, 18, 24, 28]  # Cannot go left from these states
    up_list = [0, 1, 2, 3, 4, 5, 16, 17, 25, 26, 27, 28]  # Cannot go up from these states
    down_list = [10, 11, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29]  # Cannot go down from these states
    actions = [] # right, left, up ,down
    if state not in right_list:
        actions.append('right')
    if state not in left_list:
        actions.append('left')
    if state not in up_list:
        actions.append('up')
    if state not in down_list:
        actions.append('down')
    #actions.append('still')
    return actions



def reward(state, action):
    if state == 28:
        return 100
    if action != 'still':
        return 10
    else:
        return 0

def getState(state, action):
    if action == 'right':
        state+=1
    elif action == 'left':
        state -=1
    elif action == 'up':
        state -=6
    elif action == 'down':
        state +=6
    return state

def bellman(trans_matrix, T=15):
    end_state = 29
    #pi = -np.ones((1,T))
    pi = np.zeros((1,30))
    state =  end_state
    u = -np.ones((1,30))
    u_prev = 0
    #for t in range(T):
    for s in range(30):
        A = getActions(state, trans_matrix)  # define available actions, do we need taur_pos as arg?
        u_temp = 0
        state_curr = state
        print('....')
        for action in A:
            pos_state = getState(state_curr, action)
            print(pos_state)

            r = reward(pos_state, action)
            u(0,pos_state) = r + u_prev
            print(u)

            if u > u_temp:
                u_temp = u(0,pos_state)
                state = pos_state

        u_prev += u_temp
     #pi[0,t] = state
    pi[0,s] = state
    return pi






def drawLabyrinth(w=6,h=5):
    plt.xlim(0,w) #define labyrinth width
    plt.ylim(0,h) # define labyrinth height
    plt.grid(True)

    plt.plot([2, 2], [2, 5], 'k-', lw=4)
    plt.plot([4,4],[2,4], 'k-', lw=4)
    plt.plot([4, 6], [3, 3], 'k-', lw=4)
    plt.plot([4, 4], [0, 1], 'k-', lw=4)
    plt.plot([1, 5], [1, 1], 'k-', lw=4)


# best probably to let it finish simulating first, save the path, then draw the path to the plot
def drawPath(our_path,taur_path):

    w = 5
    h = 4
    # our_path = [[rnd.randint(0, w)+0.5, rnd.randint(0, h)+0.5] for i in range(T)]
    taur_path = [[rnd.randint(0, w)+0.5, rnd.randint(0, h)+0.5] for i in range(len(our_path))]

    x, y = zip(*our_path)
    xt,yt = zip(*taur_path)
    for i in range(len(our_path)-1):
        plt.plot([x[i], x[i+1]],[y[i], y[i+1]],'k-')
        plt.plot([xt[i], xt[i+1]],[yt[i], yt[i+1]],'r-')
        plt.draw()

def main():
    T = 15
    trans_matrix = initBoard()
   # print(trans_matrix)
    #initTransitionMatrix()
    pi = bellman(trans_matrix)
    print(pi)
    #drawLabyrinth()
# drawPath(pi)
#plt.show()



main()

