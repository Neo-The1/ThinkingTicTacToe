from alphaZeroMCTS import alphaZeroMCTS
from tttBoard import tttBoard
from deepNeuralNetwork import dnNetwork

trainLabels = {}
boardSize = 3
board = tttBoard(boardSize)
brain = dnNetwork(layers=[2*boardSize*boardSize+1,64,32,boardSize*boardSize+1])
mc = alphaZeroMCTS(board,brain)
trainLabels[board.getState()] = alphaZeroMCTS.