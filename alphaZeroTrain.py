from alphaZeroMCTS import alphaZeroMCTS
from tttBoard import tttBoard
from deepNeuralNetwork import dnNetwork
import numpy as np
import time

board1DSize = 3
gamesTrain = 100
brain = dnNetwork(inputSize=2*board1DSize*board1DSize+1,
                  outputSize=board1DSize*board1DSize+1)

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
ts = time.time()
while games < gamesTrain:
    #if new run, don't load old model, else load old model
    if games > 0:
        brain.loadModel()

    playedMoves = []
    nMoves = 0
    board = tttBoard(board1DSize)
    while len(board.legalMoves()) > 0:
        state = board.getState()
        alphaZeroTTT = alphaZeroMCTS(board,brain)
        pi = alphaZeroTTT.getMCTSMoveProbs()
        playedMoves.append((state, pi))
        player = board.currPlayer()
        move = np.argmax(pi)
        board.makeMove(move)
        print("pi ", pi)
        print("move ", move)
        board.display()
        nMoves+=1
        if gameOver(board):
            games+=1
            if player == board.winner():
                z = 1
            elif player== -1:
                z = 0
            else:
                z = -1
            break
    ind = 0
    piLabel = np.zeros((nMoves,board1DSize*board1DSize))
    states = np.zeros((nMoves,2*board1DSize*board1DSize+1))
    Z = np.zeros((nMoves))
    for state, pi in playedMoves:
        #define the training data structure here,
        #add z to pi to make output vector
        #add states to make input vector
        piLabel[ind] = pi
        states[ind] = np.float32(board.decodeState(state))
        Z[ind] = z
        ind +=1
    print(games)
    #train data
    brain.train(states,[piLabel,Z])
    #save weights
    brain.saveModel()
te = time.time()
print("took " + str(te-ts) + " to train on " + str(gamesTrain) + " games")
