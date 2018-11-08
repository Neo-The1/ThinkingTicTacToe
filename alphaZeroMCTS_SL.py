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
        s : used for board
        a : used for a move
     """
    def __init__(self,board, network, *kargs, **kwds):
        self._N_sa = {}
        self._W_sa = {}
        self._L_sa = {}
        self._board = board
        self._network = network
        self._maxMoves = 100
        self._maxGameSim =500
        self._ucbK = 1.4

    def runSimulation(self):
        """ runs a monte carlo tree search simulation and updates search
            statistics
        """
        visitedActions = set()
        # should run this simulation on a copy so as not to corrupt the actual
        # board by making moves on it
        simulationBoard = copy.deepcopy(self._board)
        simBoardState = simulationBoard.getState()
        expandNode = True
        W,N = self._W_sa,self._N_sa

        for t in range(self._maxMoves):
            legalMoves = simulationBoard.legalMoves()
            statesMoves = [(simulationBoard.getStateAfterMove(a),a) for a in legalMoves]
            #stop if no legal moves
            if len(legalMoves) == 0:
                break
            # if stats exist for all legal moves
            # use the UCB formula
            if all(N.get((simBoardState,a)) for s,a in statesMoves):
                Ntotal = sum(N.get((simBoardState, a)) for s,a in statesMoves)
                logNtotal = np.log(Ntotal)
                ucbVal, state, move= max( ( (W[(simBoardState,a)]/N[(simBoardState,a)]) + self._ucbK*np.sqrt(logNtotal/N[(simBoardState,a)]), s, a) for s,a in statesMoves)
            else:                
                state, move = choice(statesMoves)
                
            Pi = [0]*self._board._boardSize
            Pi[move] = 1
            if expandNode and (simBoardState,move) not in self._N_sa:
                expandNode = False
                self._N_sa[(simBoardState,move)] = 0
                self._W_sa[(simBoardState,move)] = 0
                self._L_sa[(simBoardState,move)] = 0
                
            visitedActions.add((simBoardState, move))            
            simulationBoard.makeMove(move)
            simBoardState  = simulationBoard.getState()
            winner = simulationBoard.winner()
            if winner:
                break
        loser = simulationBoard.opponent(winner)        

        for simBoardState, move in visitedActions:
            currPlayer = self._board.stateToPlayer(simBoardState)
            if (simBoardState,move) not in self._N_sa:
                continue
            self._N_sa[(simBoardState, move)] += 1
            if currPlayer == winner:
                self._W_sa[(simBoardState, move)] += 1
            elif currPlayer == loser:
                self._L_sa[(simBoardState, move)] += 1

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
        prob, move = max(( (self._W_sa[(boardState,a)]-self._L_sa[(boardState,a)])/self._N_sa[(boardState,a)], a) for a in legalMoves)
        pi = [0]*9
        pi[move] = 1
        return pi

    def printStats(self,dicStats,dicN,state,statesMoves):
        for x in sorted(((100*dicStats.get((state,a),0)/
                          dicN.get((state,a),1),
                          dicStats.get((state,a),0),
                          dicN.get((state,a),0),a)
                            for s,a in statesMoves),
                            reverse=True) :
            print("{3}:{0:.2f}%({1}/{2})".format(*x))