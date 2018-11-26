
def QLearning():
    pass


def SARSA():
    pass


def main():
    LEARNING_TYPE = 'Q_LEARNING' # 'SARSA'

    if LEARNING_TYPE == 'Q_LEARNING':
        QLearning()
    elif LEARNING_TYPE == 'SARSA':
        SARSA()


main()
