# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
import copy
import numpy as np

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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
        self._board = board
        self._pi = [0]*self._board._boardSize
        self._z = None
        self._network = network
        self._maxMoves = 1000

    def ucb(self, s, a, cumulativeVisitCount):
        """ returns upper confidence bound for choosing move a from board
            position s
        """
        K = 1.0
        return self._Q_sa[(s, a)] + K * self._P_sa[(s, a)] * np.sqrt(cumulativeVisitCount) / ( 1.0 + self._N_sa[(s, a)] )

    def dirichletNoise(self, param, count):
        """ random number generator fitting to dirichlet noise
            https://en.wikipedia.org/wiki/Dirichlet_distribution
         """
        sample = [np.random.gamma(param, 1) for ii in range(count)]
        return [v / sum(sample) for v in sample]

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

        for t in range(self._maxMoves):
            legalMoves = simulationBoard.legalMoves()
            #stop if no legal moves
            if len(legalMoves) == 0:
                break
            # if stats exist for all legal moves
            # use the UCB formula
            if all((s, a) in self._N_sa for a in legalMoves):
                cumulativeVisitCount = sum([self._N_sa.get((s, b), 0) for b in range(self._board._boardSize)])
                ucbValue, move= max((self.ucb(s, a, cumulativeVisitCount), a) for a in legalMoves)
                visitedActions.add((s, move))
                simulationBoard.makeMove(move)
                winner = simulationBoard.winner()
                s = simulationBoard.getState()
                if winner:
                    break
            # use neural network to predict this leaf node
            # networkPredict is a list of probabilities of making a move on each square
            # of the board and a last entry {-1, 0, 1} to estimate winner
            else:
                networkPredict = self._network.predict(self._board.decodeState(s))
                netPredictPi = networkPredict[0].flatten()
                netPredictZ = networkPredict[1].flatten()
                nodeExpanded = True
                break
        # Update the statistics for this simulation
        if  nodeExpanded:
            eps    = 0.25
            dnoise = self.dirichletNoise(0.03, len(legalMoves))
            moveIndex = 0
            for move in legalMoves:
                self._N_sa[(s,move)] = 0
                self._Q_sa[(s,move)] = 0
                self._W_sa[(s,move)] = 0
                self._P_sa[(s, move)] = (1. - eps) * netPredictPi[move] + eps * dnoise[moveIndex]
                moveIndex += 1
        print(visitedActions)
        for s, move in visitedActions:
            print(visitedActions)
            print("IN")
            self._N_sa[(s, move)] += 1
            if nodeExpanded:  # network predicted winner
                self._W_sa[(s, move)] += netPredictZ[0]
            else: # true winner
                #TODO: I am not sure if W should be updated relative to whose move is it or absolutely
                # doing absolutely here. -1 if O wins 1 if X wins 0 if a draw
                if winner == 1:
                    self._W_sa[(s, move)] += -1
                elif winner == 2:
                    self._W_sa[(s, move)] += 1
            self._Q_sa[(s, move)] = self._W_sa[(s, move)]/self._N_sa[(s, move)]

    def getMCTSMoveProbs(self,tau=0):
        """ returns  the vector pi of move probability at each move
            and scalar winner z
            tau is a parameter which determines whether max move is returned (tau=0)
            or whether a proportional probability is returned (tau = 1)
        """
        legalMoves = self._board.legalMoves()
        s = self._board.getState()
        # no need to run simulation if there are no real choices
        # so return accordingly
        games = 0
        while games < 500:
            self.runSimulation()
            games+=1
        #define new empty list
        newPi = [0]*self._board._boardSize
        for ii in range(self._board._boardSize):
            if ii in legalMoves:
                newPi[ii] = self._P_sa[(s,ii)]
        #normalize pi is tau = 1, convert it to one hot if tau = 0
        N = np.sum(newPi) #total N, needed to normalize
        if tau == 1:
            newPi = np.divide(newPi,N)
            self._pi = newPi.copy()
        if tau == 0:
            newOneHotPi = [0]*self._board._boardSize
            newOneHotPi[(np.argmax(newPi))] = 1
            self._pi = newOneHotPi.copy()
            return self._pi

    def logSimulationStats(self, board):
        """ treating board as node, prints stats for leaves emanating from it
            formatting is done in each leaf cell as shown
            Q and P are printed in format %.2f hence are 4 character wide
            +-------+
            |   W   |
            | Q - P |
            |   N   |
            +-------+
            height = 5
        """
        s = board.getState()
        lenN = max([len(str(self._N_sa[(s, a)])) for a in board.legalMoves()])
        lenhyp = 1 + 4 + 1 + lenN + 1 + 4 + 1
        atomicHeader = '+' + lenhyp * '-'
        for cellRow in range(board.getSize()):
            print(board.getSize() * atomicHeader + '+')
            wrow  = ''
            qprow = ''
            nrow  = ''
            for cellCol in range(board.getSize()):
                cellID = cellRow * board.getSize() + cellCol
                if cellID not in board.legalMoves():
                    # occupied square
                    wrow  += "|      {0:^{wwidth}}      ".format('', wwidth = lenN)
                    if board.playerAt(cellID) == 1:
                        qprow += "| {0:4} {OorX:^{nwidth}} {0:4} ".format(' ', ' ', OorX='O', nwidth = lenN)
                    else:
                        assert(board.playerAt(cellID) == 2)
                        qprow += "| {0:4} {OorX:^{nwidth}} {0:4} ".format(' ', ' ', OorX='X', nwidth = lenN)
                    nrow  += "|      {0:^{nwidth}}      ".format('', nwidth = lenN)
                else:
                    wrow  += "|      {0:^{wwidth}}      ".format(int(self._W_sa[(s, cellID)]), wwidth = lenN)
                    qprow += "| {0:.2f} {den:{nwidth}} {0:.2f} ".format(self._Q_sa[(s, cellID)], self._P_sa[(s, cellID)], den = lenN * '-', nwidth = lenN)
                    nrow += "|      {0:^{nwidth}}      ".format(self._N_sa[(s, cellID)], nwidth = lenN)
            print(wrow + '|')
            print(qprow + '|')
            print(nrow + '|')
        print(board.getSize() * atomicHeader + '+')

