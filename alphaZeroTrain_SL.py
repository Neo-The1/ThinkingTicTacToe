from alphaZeroMCTS_SL import alphaZeroMCTS
from tttBoard import tttBoard
#from convNeuralNetwork import cnNetwork
from deepNeuralNetwork_SL import dnNetwork
import numpy as np

board1DSize = 3
gamesTrainBatch = 1000
totalBatches =1
#brain = cnNetwork(inputShape=(board1DSize,board1DSize,7),
#                  outputSize=board1DSize*board1DSize+1)
board = tttBoard(3)
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
    allPiLabels = []
    allZLabels = []
    while games < TotalGames:
        print(games)
        playedMoves = {}
        board = tttBoard(board1DSize)
        nMoves = 0
#        brain.loadModel()
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
        piLabel = np.zeros((board1DSize*board1DSize,nMoves))
        ZLabel = np.zeros((1,nMoves))
        states = np.zeros((2*board1DSize*board1DSize+1,nMoves))
#        statesCNN = np.zeros((nMoves,board1DSize,board1DSize,7))
        for state in playedMoves:
            pi = playedMoves[state]
            #define the training data structure here,           
#            statesCNN[ind] = np.float32(board.decodeStateCNN(board._stateHistory))
            states[:,ind] = board.decodeState(state)
            piLabel[:,ind] = pi
            ZLabel[:,ind] = z
            ind+=1
        
        allStates.append(states)
        allPiLabels.append(piLabel)
        allZLabels.append(ZLabel)
    allStates = np.concatenate(allStates,axis=1)
    allPiLabels = np.concatenate(allPiLabels,axis=1)
    allZLabels = np.concatenate(allZLabels,axis=1)
    return allStates, allPiLabels, allZLabels
        
for ii in range(totalBatches):
    print(ii)
#    brain.loadWeights()
    inp,pi,z = playGame(brain,gamesTrainBatch)
    np.savetxt("trainX.txt",inp, fmt='%2d', delimiter=',', newline='\n')
    np.savetxt("trainYPi.txt",pi, fmt='%2d', delimiter=',', newline='\n')
    np.savetxt("trainYZ.txt",z, fmt='%2d', delimiter=',', newline='\n')
    trainX = np.loadtxt("trainX.txt",delimiter=',').astype(np.float32)
    trainYPi = np.loadtxt("trainYPi.txt",delimiter=',').astype(np.float32)
    trainYZ = np.loadtxt("trainYZ.txt",delimiter=',').astype(np.float32)
    loadedShape = trainYZ.shape[0]
    trainYZ = np.reshape(trainYZ,[1,loadedShape])
#    trainY = np.concatenate((trainYPi,trainYZ))
    brain.train(trainX.T,trainYPi.T)
    brain.saveModel()
