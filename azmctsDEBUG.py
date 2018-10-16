from tttBoard import tttBoard
from alphaZeroMCTS_SL import alphaZeroMCTS
from convNeuralNetwork import cnNetwork

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
brain = cnNetwork(inputShape=(board1DSize,board1DSize,7),
                  outputSize=board1DSize*board1DSize+1)


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
    alphaZeroTTT = alphaZeroMCTS(board,brain)
    print("Thinking...")
    for ii in range(100):
        alphaZeroTTT.runSimulation()
    alphaZeroTTT.logSimulationStats(board)
    s = board.getState()
    #prob, move = max((alphaZeroTTT._W_sa[(s, a)] / alphaZeroTTT._N_sa[(s, a)], a) for a in board.legalMoves())
    prob, move = max((alphaZeroTTT._W_sa.get((s,a),0), a) for a in board.legalMoves())
    #for a in board.legalMoves():
    #    print(a, alphaZeroTTT._Q_sa[(s,a)], '\t', alphaZeroTTT._W_sa[(s,a)], '\t', alphaZeroTTT._N_sa[(s,a)], '\t', alphaZeroTTT._P_sa[(s,a)])
    print(prob, move)
    assert(move in board.legalMoves())
    board.makeMove(move)
    board.display()