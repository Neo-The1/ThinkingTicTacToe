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
    def __init__(self,board, network, *kargs, **kwds):
        self._P_sa = {}
        self._N_sa = {}
        self._Q_sa = {}
        self._W_sa = {}
        # if move 'a' from board pos 's' led to board pos 'sp' following
        # dictionary will store opinion of neural network about winner for 'sp'
        #self._saTosp = {}
        self._board = board
        self._p = [0]*self._board.getSize()
        self._v = None
        self._pi = [0]*self._board.getSize()
        self._z = None
        self._network = network
        self._maxMoves = 1000


#    def hashAction(self, board, move):
#        pass
        
    def ucb(self, s, a):
        """ returns upper confidence bound for choosing move a from board
            position s
        """
        K = 1.0
        return self._Q_sa[(s, a)] + K * self._P[(s, a)] / ( 1.0 + self._N_sa[(s, a)] )

    def runSimulation(self):
        """ runs a monte carlo tree search simulation and updates search
            statistics
        """
        visitedActions = set()
        # should run this simulation on a copy so as not to corrupt the actual
        # board by making moves on it
        simulationBoard = copy.deepcopy(self._board)
        s = simulationBoard.getState()
        nodeExpanded = False

        for t in range(maxMoves):
            legalMoves = simulationBoard.legalMoves()
            #stop if no legal moves
            if len(legalMoves) == 0:
                break
            # if stats exist for all legal moves
            # use the UCB formula
            if all(self._N_sa.get((s, a)) for a in legalMoves):
                ucbValue, move= max((self.ucb(s, a), a) for a in legalMoves)
                visitedActions.add((s, move))
                simulationBoard.makeMove(move)
                winner = simulationBoard.winner()
                s = simulationBoard.getState()
            if winner:
               break
            # use neural network to predict this leaf node
            # networkPredict is a list of probabilities of making a move on each square
            # of the board and a last entry {-1, 0, 1} to estimate winner
            networkPredict = self._network.predict(s, network)
            nodeExpanded = True
            break
        # Update the statistics for this simulation
        if  nodeExpanded:
            for ii in range(simulationBoard.getSize()):
                move = ii
                self._N_sa[(s,move)] = 0
                self._Q_sa[(s,move)] = 0
                self._W_sa[(s,move)] = 0
                self._P_sa[(s, move)] = networkPredict[ii]

        for s, move in visitedActions:
            self._N_sa[(s, move)] += 1
            #hashVal = self.hashAction(board, move)
            #self._saTosp[(hashVal, simulationBoard.getState())] += networkPredict[-1]
            #self._Q_sa[(board, move)] = self._saTosp[(hashVal, simulationBoard.getState())] / self._N_sa[(board, move)]
            self._W_sa[(s, move)] +=networkPredict[-1] #winner is last entry in network output
            self._Q_sa[(s, move)] = self._W_sa[(s, move)]/self._N_sa[(s, move)]
            
        def getMCTSMove(self,tau):
            """ returns  the vector pi of move probability at each move
                and scalar winner z
                tau is a parameter which determines whether max move is returned (tau=0)
                or whether a proportional probability is returned (tau = 1)
            """
            legalMoves = self._board.legalMoves()
            s = self._board.getState()
            # no need to run simulation if there are no real choices
            # so return accordingly
            if not legalMoves:
                return None
            if len(legalMoves) == 1:
                return legalMoves[0]
            games = 0
            while games < 5000:
                self.runSimulation()
                games+=1

            for ii in range(self._board.getSize()):
                if ii in legalMoves:
                    self._pi[ii] = self._N_sa[(s,ii)]
            self._z = winner
            #normalize pi is tau = 1, convert it to one hot if tau = 0
            N = np.sum(self._pi) #total N, needed to normalize
            if tau == 1:
                self._pi = np.divide(self._pi,N)
            if tau == 0:
                newPi = [0]*simulationBoard.getSize()
                newPi[(np.argmax(self._pi))] = 1.0
                self._pi = newPi
            
            return self._pi,self._z
            

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
        return mc.getMCTSMove()

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

