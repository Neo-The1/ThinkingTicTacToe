from __future__ import division
import datetime
from random import choice
from math import log, sqrt
class MonteCarlo:
    #initialize
    def __init__(self,board,**kwargs):
        self._board = board
        self._states = []
        seconds = kwargs.get('time',30)
        self._simTime = datetime.timedelta(seconds = seconds)
        self._maxMoves = kwargs.get('maxMoves',100)
        self._wins = {}
        self._plays = {}
        self._C = kwargs.get('C',1.4)
        
    #Call AI to calculate best move from current state and return it
    def getMove(self):
        player = self._board.currPlayer()
        legalMoves = self._board.legalMoves()
        # no need to run simulation if there are no real choices
        #so return accordingly
        if not legalMoves:
            return
        if (len(legalMoves)==1):
            return legalMoves[0]
        games = 0
        begin = datetime.datetime.utcnow() #gets current time
        #run the simulation till the specified time
        while datetime.datetime.utcnow() - begin < self._simTime:
            self.runSimulation()
            games+=1
        #list of tuples of move and state resulting from move
        movesStates = [(p,self._board.makeMove(p)) for p in legalMoves]            
        # Display the number of calls of `run_simulation` and the
        # time elapsed.
        print(games, (datetime.datetime.utcnow() - begin))
        
        #Pick move with highest win percentage
        percentWins, move = max(
                (self._wins.get((player,S),0)/
                self._plays.get((player,S),1),
                p)
                for p,S in movesStates
                )
        
        #display stats for each possible play
        for x in sorted(
                ((100*self._wins.get((player,S),0)/
                self._plays.get((player,S),1),
                self._wins.get((player,S),0),
                self._plays.get((player,S),0),p)
                for p,S in movesStates),
                reverse = True
                ):
                    print("{3}:{0:.2f}%({1}/{2})".format(*x))
        print("Maximum Moves Searched: ",self._maxMoves)
        
        return move
    #playout a random game and update the statistics table
    def runSimulation(self):
        #copying some variables so taht we have variable lookup instead of
        #attrivute call, to make code faster
        plays,wins = self._plays,self._wins
        expandTree =True
        visitedStates = set()    
        player = self._board._currPlayer()
        statesCopy = self._states[:]
        state = statesCopy[-1]
            
        for t in range(1,self._maxMoves+1):
            legalMoves = self._board.legalMoves()
            movesStates = [(p,self._board.makeMove(p)) for p in legalMoves]            
            #if stats exist for all legal moves
            #use the UCB formula
           if all(plays.get((player,S)) for p,S in movesStates):
               logN = log(sum(plays.get[(player,S)] for p,S in movesStates))
               value, move, state = max(
                       ((wins[(player,S)]/plays[(player,S)])+
                        self._C*sqrt(logN/plays[(player,S)]),p,S)
                        for p,S in movesStates
                       )
           else:
               #play randomly
               move,state = choice(movesStates)
            
            statesCopy.append(state)
            
            #if this is a new leaf, set statistics to 0
            if expandTree and (player,state) not in self._plays:
                expandTree = False
                self._plays[(player,state)] = 0
                self._wins[(player,state)] = 0
                if t>self._maxMoves:
                    self._maxMoves = t
            
            #add the current position to visited boards
            visitedStates.add((player,state))
            #set board and player
            player = self._board.currPlayer()
            state = self._board._board
            winner = self._board.winner()
            if winner:
                break
            
        #update the win and play stats for the simulation
        for player,state in visitedStates:
            if (player,state) not in self._plays:
                continue
            self._plays[(player,state)] += 1
            if player == winner:
                self._wins[(player,state)] += 1