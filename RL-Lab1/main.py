import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random as rnd
import time


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
    pass


def reward(state, action):
    if state == end_state:
        return 100
    if state == death:
        return -100
    else:
        return 0


def bellman(trans_matrix, T=15):
    our_state = end_state
    r = np.zeros((1,T)) # vector of rewards
    r[0] = reward(end_state)
    for t in range(T):
        A = getActions(our_state, trans_matrix)  # define available actions, do we need taur_pos as arg?
        for a in range(A):
            u = reward(our_state, action)

            #state =


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
def drawPath():
  #  plt.ion()
    T = 4
    w = 5
    h = 4
    our_path = [[rnd.randint(0, w)+0.5, rnd.randint(0, h)+0.5] for i in range(T)]
    taur_path = [[rnd.randint(0, w)+0.5, rnd.randint(0, h)+0.5] for i in range(T)]

    print(our_path)
    x, y = zip(*our_path)
    xt,yt = zip(*taur_path)
    for i in range(T-1):
        plt.plot([x[i], x[i+1]],[y[i], y[i+1]],'k-')
        plt.plot([xt[i], xt[i+1]],[yt[i], yt[i+1]],'r-')
        plt.draw()



def main():
    T = 15
    trans_matrix = initBoard()
    #initTransitionMatrix()
    #bellman(trans_matrix)
    drawLabyrinth()
    drawPath()
    plt.show()



main()

