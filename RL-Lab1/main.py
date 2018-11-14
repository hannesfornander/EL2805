import numpy as np


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

def bellman(taur_pos, our_pos):
    for t in range(T):
        for a in range(A):




def main():
    T = 15
    initBoard()
    #initTransitionMatrix()
    bellman()

main()

