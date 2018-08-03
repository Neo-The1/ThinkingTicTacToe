from alphaZeroMCTS import alphaZeroMCTS
from tttBoard import tttBoard
from deepNeuralNetwork import dnNetwork
import numpy as np

boardSize = 3
gamesTrain = 2
board = tttBoard(boardSize)
brain = dnNetwork(layers=[boardSize*boardSize,64,32,boardSize*boardSize+1])

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
    
games = 0
while games < gamesTrain:
    states = np.zeros([boardSize*boardSize,boardSize*boardSize])
    piZ = np.zeros([boardSize*boardSize,boardSize*boardSize+1])
    playedMoves = {}
    while len(board.legalMoves()) > 0 and not(gameOver(board)):
        state = board.getState()
        #load saved weights
        #brain.loadModel()
        alphaZeroTTT = alphaZeroMCTS(board,brain)
        pi = alphaZeroTTT.getMCTSMoveProbs()
        playedMoves[state] = pi
        player = board.currPlayer()
        board.makeMove(np.argmax(pi))
        print("pi ", pi)
        print("move ",np.argmax(pi))
        board.display()
        if gameOver(board):
            games+=1
            if player == board.winner():
                z = 1
            else:
                z = -1
            break
    ind = 0
    for state in playedMoves:
        pi = playedMoves[state]
        #define the training data structure here, 
        #add z to pi to make output vector
        #add states to make input vector
        piZ[ind] = np.append(np.array(np.float32(pi)),z)
        states[ind] = board.decodeState(state)            
        ind +=1
    #train data
    brain.train(states,piZ)
    #save weights
    brain.saveModel()