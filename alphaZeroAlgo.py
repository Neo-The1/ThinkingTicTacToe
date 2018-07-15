# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
from tttBoard import tttBoard
from deepNeuralNetwork import dnNetwork
import copy
import numpy as np

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class alphaZeroMCTS:
    """ alpha zero monte carlo tree search algorithm. Refer to alpha zero paper
        for notations. Variables directly corresponds to the notation in paper
        s : used for board
        a : used for a move
     """
    def __init__(self,board, *kargs, **kwds):
        self._P_sa = {}
        self._N_sa = {}
        self._Q_sa = {}
        # if move 'a' from board pos 's' led to board pos 'sp' following
        # dictionary will store opinion of neural network about winner for 'sp'
        self._saTosp = {}
        self._board = board
        self._p = [0]*self._board.getSize()
        self._v = None

    def hashAction(self, board, move):
        pass
        
    def ucb(self, s, a):
        """ returns upper confidence bound for choosing move a from board
            position s
        """
        K = 1.0
        return self._Q_sa[(s, a)] + K * self._P[(s, a)] / ( 1.0 + self._N_sa[(s, a)] )

    def expandNode(self, s, network):
        """ expands the leaf node i.e. board s if no simulation data from this
            position is available. Move probabilities will be generated and winner
            will be predicted  neural network
        """
        return network.predict(s)

    def runSimulation(self, s, maxMoves, network):
        """ runs a monte carlo tree search simulation and updates search
            statistics
        """
        visitedActions = set()

        # should run this simulation on a copy so as not to corrupt the actual
        # board by making moves on it
        simulationBoard = copy.deepcopy(self._board)

        nodeExpanded = False

        for t in range(maxMoves):
            legalMoves = simulationBoard.legalMoves()
            s = simulationBoard.getState()
            if len(legalMoves) == 0:
                break
            # if stats exist for all legal moves
            # use the UCB formula
            if all(self._N_sa.get((s, a)) for a in legalMoves):
                ucbValue, move= max((self.ucb(s, a), a) for a in legalMoves)
                visitedActions.add((s, move))
                simulationBoard.makeMove(move)
                winner = simulationBoard.winner()
                if winner:
                    break
                continue

            # use neural network to predict this leaf node and stop simulating
            # networkPredict is a list of probabilities of making a move on each square
            # of the board and a last entry {-1, 0, 1} to estimate winner
            networkPredict = self.expandNode(simulationBoard, network)
            nodeExpanded = True
            break

        # Update the statistics for this simulation
        if nodeExpanded:
            for ii in range(simulationBoard.getSize()):
                move = ii
                self._P_sa[(s, move)] = networkPredict[ii]

        for board, move in visitedActions:
            self._N_sa[(board, move)] += 1
            hashVal = self.hashAction(board, move)
            self._saTosp[(hashVal, simulationBoard.getState())] += networkPredict[-1]
            self._Q_sa[(board, move)] = self._saTosp[(hashVal, simulationBoard.getState())] / self._N_sa[(board, move)]
            
        def selfPlay(self):
            """returns  the vector p and scalar v as a list"""
            legalMoves = self._board.legalMoves()
            for ii in range(simulationBoard.getSize()):
                if ii in legalMoves:
                    self._p = self._P_sa[(self._board,move)]
            self._v = winner
            return self._p.append(self._v)
                    
                
            

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class alphaZeroAlgo:
    """ alpha zero reenforcement learning algorithm for learning the game
        of tic tac toe. The algorithm uses monte carlo tree search simulations
        to generate good moves and train a neural network to mimic the mcts
        search i.e. to train the network to achieve parameters such that it
        generates similar move probabilities and winner prediction
    """
    def __init__(self, *kargs, **kwds):
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
        mc    = alphaZeroMCTS(board, self._network)
        return mc.selfPlay()

    def train(self, iterations):
        """ train the network for requested number of iterations """
        for it in range(iterations):
            self._trainingData.append(self.selfPlay())
            self._network.train(self._trainingData)

    def saveNetwork(self, filename):
        self._network(filename)

    def getMove(self, board):
        """ use the network to predict a move for passed board position """
        pass

