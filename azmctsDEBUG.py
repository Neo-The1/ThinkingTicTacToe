from tttBoard import tttBoard
from alphaZeroMCTS_SL import alphaZeroMCTS
from dNNTF import dNNTF
import numpy as np
def gameOver(board):
    if (board.winner()):
        if(board.winner()==1):
            return 1
        elif(board.winner()==2):
            return 1
        elif(board.winner()==-1):
            return 1
    else:
        return 0

board1DSize = 3
brain = dNNTF(board1DSize)


board = tttBoard(board1DSize)

#board.makeMove(4)
#board.makeMove(0)
#for ii in range(500):
#    alphaZeroTTT.runSimulation()

#lm = board.legalMoves()
#s = board.getState()

#for a in lm:
#    print('q', a, alphaZeroTTT._Q_sa[(s, a)])
#    print('p', a, alphaZeroTTT._P_sa[(s, a)])
#    print('w', a, alphaZeroTTT._W_sa[(s, a)])
#    print('n', a, alphaZeroTTT._N_sa[(s, a)])

while not gameOver(board):
    print('make a move')
    n = int(input())
    board.makeMove(n)
    board.display()
    boardState = board.getState()
    legalMoves = board.legalMoves()
    statesMoves = [(board.getState(),a) for a in legalMoves]
    alphaZeroTTT = alphaZeroMCTS(board,brain)
    print("Thinking...")
    for ii in range(1000):
        alphaZeroTTT.runSimulation()
    print("win stats")
    alphaZeroTTT.printStats(alphaZeroTTT._W_sa,alphaZeroTTT._N_sa,boardState,statesMoves)
    print("loss stats")
    alphaZeroTTT.printStats(alphaZeroTTT._L_sa,alphaZeroTTT._N_sa,boardState,statesMoves)
#    print("Q stats")
#    alphaZeroTTT.printStats(alphaZeroTTT._Q_sa,alphaZeroTTT._N_sa,boardState,statesMoves)
    s = board.getState()
    prob = alphaZeroTTT.getMCTSMoveProbs()
    move = np.argmax(prob)
    print(prob)
    print(move)
    assert(move in board.legalMoves())
    board.makeMove(move)
    board.display()