# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
import copy
import numpy as np
from random import choice
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class alphaZeroMCTS:
    """ alpha zero monte carlo tree search algorithm. Refer to alpha zero paper
        for notations. Variables directly corresponds to the notation in paper
        s : used for current board position
        a : used for move a from position s
     """
    def __init__(self,board, network, *kargs, **kwds):
        self._N_sa = {}
        self._W_sa = {}
        self._Q_sa = {}
        self._P_sa = {}
        self._board = board
        self._network = network
        self._maxMoves = 10
        self._maxGameSim =5
        self._ucbK = 1.4
        self._pi = [0]*self._board._boardSize
        self._v = 0
        self._p = [0]*self._board._boardSize
    
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
        simBoardState = simulationBoard.getState()
        Q,N = self._Q_sa,self._N_sa

        for t in range(self._maxMoves):
            legalMoves = simulationBoard.legalMoves()
            #stop if no legal moves
            if len(legalMoves) == 0:
                break
            # check if node has been expanded
            if  all(N.get((simBoardState,a)) for a in legalMoves):
                 #use the UCB formula
                Ntotal = sum(filter(None,(N.get((simBoardState, a)) for a in legalMoves)))
                logNtotal = np.log(Ntotal)
                ucbVal, move= max( ( Q[(simBoardState,a)]
                + self._ucbK*np.sqrt(logNtotal/N[(simBoardState,a)]),a) for a in legalMoves)
                visitedActions.add((simBoardState,move))
            else:
                s = np.zeros((2*self._board._1Dsize**2+1,1))
                netPredict = self._network.predict(s)
                self._p = netPredict[0][:,0]
                self._v = netPredict[1][:,0][0]
                dnoise = self.dirichletNoise(0.03, len(legalMoves))
                eps = 0.25
                moveIndex = 0
                for a in legalMoves:
                    self._N_sa[(simBoardState,a)]=0
                    self._Q_sa[(simBoardState,a)]=0
                    self._W_sa[(simBoardState,a)]=0
                    self._P_sa[(simBoardState,a)]=(1 - eps)*self._p[a] + eps*dnoise[moveIndex]
                    visitedActions.add((simBoardState,a))
                    moveIndex+=1
                break
                
            simulationBoard.makeMove(move)
            simBoardState  = simulationBoard.getState()
            winner = simulationBoard.winner()
            if winner:
                break

        for simBoardState, move in visitedActions:
            if (simBoardState,move) not in self._N_sa:
                continue
            self._N_sa[(simBoardState, move)] += 1
            self._W_sa[(simBoardState,move)] += self._v
            self._Q_sa[(simBoardState, move)] = self._W_sa[(simBoardState,move)]/self._N_sa[(simBoardState,move)]

    def getMCTSMoveProbs(self,tau=0):
        """ returns  the vector pi of move probability at each move
            and scalar winner z
            tau is a parameter which determines whether max move is returned (tau=0)
            or whether a proportional probability is returned (tau = 1)
        """
        legalMoves = self._board.legalMoves()
        boardState = self._board.getState()
        # no need to run simulation if there are no real choices
        # so return accordingly
        games = 0
        while games < self._maxGameSim:
            self.runSimulation()
            games+=1
        prob, move = max((self._Q_sa[(boardState,a)], a) for a in legalMoves)
        self._pi[move] = 1
        return self._pi

    def printStats(self,dicStats,dicN,state,statesMoves):
        for x in sorted(((100*dicStats.get((state,a),0)/
                          dicN.get((state,a),1),
                          dicStats.get((state,a),0),
                          dicN.get((state,a),0),a)
                            for s,a in statesMoves),
                            reverse=True) :
            print("{3}:{0:.2f}%({1}/{2})".format(*x))
