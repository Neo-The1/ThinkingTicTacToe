from alphaZeroMCTS import alphaZeroMCTS
from tttBoard import tttBoard
from deepNeuralNetwork import dnNetwork
import numpy as np

playedMoves = set()
boardSize = 3
gamesTrain = 500
board = tttBoard(boardSize)
brain = dnNetwork(layers=[2*boardSize*boardSize+1,64,32,boardSize*boardSize+1])

def checkWin(board):
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
    while len(board.legalMoves()) > 0 and not(checkWin(board)):
        state = board.getState()
        #load saved wrights
        brain.loadModel()
        alphaZeroTTT = alphaZeroMCTS(board,brain)
        pi = alphaZeroTTT.getMCTSMoveProbs()
        playedMoves.add((state,pi))
        player = board.currPlayer()
        board.makeMove(np.argmax(pi))
        print('train labels'+str(pi))
        if checkWin(board):
            games+=1
            if player == board.winner():
                z = 1
            else:
                z = -1
            break
    for state,pi in playedMoves:
        #define the training data structure here, 
        #add z to pi to make output vector
        #add states to make input vector
        states = None
    #train data
    brain.train(states,[pi,z])
    #save weights
    brain.saveModel()
    
    

