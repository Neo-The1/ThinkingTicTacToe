from alphaZeroMCTS import alphaZeroMCTS
from tttBoard import tttBoard
from deepNeuralNetwork import dnNetwork

trainLabels = {}
states = {}
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
        alphaZeroTTT = alphaZeroMCTS(board,brain)
        tttMove = alphaZeroTTT.getMove()
        trainLabels[board.getState()] = alphaZeroMCTS.getMCTSMove()
        player = board.currPlayer()
        board.makeMove(tttMove)
        print('train labels'+str(tttMove))
        if checkWin(board):
            games+=1
            if player == board.winner():
                z = 1
            else:
                z = -1
            break
    
    for move in trainLabels:
        trainLabels
    

