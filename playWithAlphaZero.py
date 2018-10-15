#code to play with AI
import numpy as np
from tttBoard import tttBoard
board1DSize = 3
#from convNeuralNetwork import cnNetwork
#alphaZero = cnNetwork(inputShape=(board1DSize,board1DSize,7),
#                  outputSize=board1DSize*board1DSize+1)
from deepNeuralNetwork import dnNetwork
alphaZero = dnNetwork(2*board1DSize*board1DSize+1,board1DSize*board1DSize+1)
alphaZero.loadModel()
board = tttBoard(board1DSize)
board.display()
def gameOver(board):
    if (board.winner()):
        if(board.winner()==1):
            print("O wins!")
            return 1
        elif(board.winner()==2):
            print("X wins!")
            return 1
        elif(board.winner()==-1):
            print("It's a Draw!")
            return 1
    else:
        return 0

while len(board.legalMoves()) > 0 and not(gameOver(board)):
    print(board.legalMoves())
    pMove = int(input("choose from above list of moves"))
    board.makeMove(pMove)
    board.display()
    if gameOver(board):
        break
    print('Thinking...')
#    alphaZeroPredict = alphaZero.predict(board.decodeStateCNN(board._stateHistory))
    alphaZeroPredict = alphaZero.predict(board.decodeState(board.getState()))
    alphaZeroMovesProbs = alphaZeroPredict[0].flatten()
    print(alphaZeroMovesProbs)
    board.makeMove(np.argmax(alphaZeroMovesProbs))
#    print('Thinking Tic Tac Toe move...'+str(alphaZeroMove))
    board.display()
    if gameOver(board):
        break
