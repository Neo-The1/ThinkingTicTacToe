from alphaZeroMCTS import alphaZeroMCTS
from tttBoard import tttBoard
from deepNeuralNetwork import dnNetwork
import numpy as np

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
while games < gamesTrain:
    playedMoves = {}
    nMoves = 0
    board = tttBoard(board1DSize)
    #if new run, don't load old model, else load old model
    if games < 1:
        pass
    else:
        #load saved weights
        brain.loadModel()
    while len(board.legalMoves()) > 0 and not(gameOver(board)):
        state = board.getState()            
        alphaZeroTTT = alphaZeroMCTS(board,brain)
        pi = alphaZeroTTT.getMCTSMoveProbs()
        playedMoves[state] = pi
        player = board.currPlayer()
        board.makeMove(np.argmax(pi))
        print("pi ", pi)
        print("move ",np.argmax(pi))
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
    for state in playedMoves:
        pi = playedMoves[state]
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