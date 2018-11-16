from alphaZeroMCTS_SL import alphaZeroMCTS
from tttBoard import tttBoard
#from convNeuralNetwork import cnNetwork
from deepNeuralNetwork import dnNetwork
import numpy as np

board1DSize = 3
gamesTrainBatch = 1
totalBatches =1
#brain = cnNetwork(inputShape=(board1DSize,board1DSize,7),
#                  outputSize=board1DSize*board1DSize+1)
brain = dnNetwork(2*board1DSize*board1DSize+1,board1DSize*board1DSize+1)
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

def playGame(brain,TotalGames):
    games = 0
    allStates = []
    allPiZLabels = []
    while games < TotalGames:
        print(games)
        playedMoves = {}
        board = tttBoard(board1DSize)
        nMoves = 0
        #brain.loadModel()
        while len(board.legalMoves()) > 0:
            state = board.getState()
#            print("state ",state)
            alphaZeroTTT = alphaZeroMCTS(board,brain)
            pi = alphaZeroTTT.getMCTSMoveProbs()
            playedMoves[state] = pi
#            print("pi ", pi)
            board.makeMove(np.argmax(pi))
#            print("move ",np.argmax(pi))
#            board.display()
            nMoves += 1
            if gameOver(board):
                games += 1
                winner = board.winner()
                if winner == 1:     # O is winner
                    z = 1
                elif winner == -1:  # draw
                    z = 0
                else:               # X is winner
                    assert(winner == 2)
                    z = -1
                break
        ind = 0
        piZLabel = np.zeros((nMoves,board1DSize*board1DSize))
        states = np.zeros((nMoves,2*board1DSize*board1DSize+1))
        Z = np.zeros((nMoves))
#        statesCNN = np.zeros((nMoves,board1DSize,board1DSize,7))
        for state in playedMoves:
            pi = playedMoves[state]
            #define the training data structure here, 
            
#            statesCNN[ind] = np.float32(board.decodeStateCNN(board._stateHistory))
            states[ind] = np.float32(board.decodeState(state))
            piZLabel[ind] = np.float32(pi)
            Z[ind] = z
            ind+=1
        allStates.append(states)
        allPiZLabels.append([piZLabel,Z])
    
    allStates = np.concatenate(allStates)
    allPiZLabels = np.concatenate(allPiZLabels)
    return (allStates, allPiZLabels)
        
for ii in range(totalBatches):
    print(ii)
    brain.loadModel()
    (inp,piZ) = playGame(brain,gamesTrainBatch)
    np.savetxt("train_x.txt",inp, fmt='%2d', delimiter=',', newline='\n')
    np.savetxt("train_y.txt",piZ, fmt='%2d', delimiter=',', newline='\n')
    train_x = np.loadtxt("train_x.txt",delimiter=',')
    train_y = np.loadtxt("train_y.txt",delimiter=',')
    print(train_x)
    print(train_y)
    brain.train(train_x,train_y)
    brain.saveModel()
