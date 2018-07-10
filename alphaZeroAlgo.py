# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
from tttBoard import tttBoard
from monteCarlo import monteCarlo
from deepNeuralNetwork import dnNetwork

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class alphaZeroAlgo:
    """ alpha zero reenforcement learning algorithm for learning the game
        of tic tac toe. The algorithm uses monte carlo tree search simulations
        to generate good moves and train a neural network to mimic the mcts
        search i.e. to train the network to achieve parameters such that it
        generates similar move probabilities and winner prediction
    """
    def __init__(self, *kargs, *kwds):
        self._boardSize = kwds.get('boardsize', 3)
        inputLayerSize  = 2 * self._boardSize * self._boardSize + 1
        outputLayerSize = self._boardSize + 1
        self._network   = dnNetwork(layers = [inputLayerSize, 64, 32, outputLayerSize])
        self._trainingData = []

    def selfPlay(self):
        """ play a game using mcts and return the generated game play data
            which can be used to train the network
        """
        board = tttBoard(self._boardSize)
        mc    = monteCarlo(board, self._network)
        return mc.selfPlay()

    def train(self, iterations):
        """ train the network for requested number of iterations """
        for it in range(iterations):
            self._trainingData.append(training self.selfPlay())
            self._network.train(self._trainingData)

    def saveNetwork(self, filename):
        self._network(filename)

    def getMove(self, board):
        """ use the network to predict a move for passed board position """
        pass

