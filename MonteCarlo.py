from __future__ import division
import datetime
from random import choice
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
    
    #Append the passed game state to history
    def update(self,state):
        self._states.append(states)
        
    #Call AI to calculate best move from current state and return it
    def getMove(self):
        state = self._board.currBoard()
        player = self._board._sideToMove
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
            self.runSimulation
            games+=1
        
        movesStates = [] #list of tuples of move and state resulting from move
        for p in legalMoves:
            self._board.makeMove(p)
            movesStates.apped((p,self._board.currBoard()))
            
        # Display the number of calls of `run_simulation` and the
        # time elapsed.
        print (games, datetime.datetime.utcnow() - begin)
        
        #Pick move with highest win percentage
        percentWins, move = max(
                (self._wins.get((player,S),0)/
                self._plays.get((player,S),1),
                p)
                for p,S in movesStates
                )
        
    #playout a random game and update the statistics table
    def runSimulation(self):
        expandTree =True
        visitedStates = set()    
        player = self._board._sideToMove
        state = self._board.currBoard()
            
        for x in range(self._maxMoves):
            #play randomly
            legalMoves = self._board.legalMoves(self._board)
            move = choice(legalMoves)
            self._board.makeMove(move)

            #if this is a new leaf, set statistics to 0
            if expandTree and (player,state) not in self._plays:
                expandTree = False
                self._plays[(player,state)] = 0
                self._wins[(player,state)] = 0
            
            #add the current position to visited and set boards
            visitedStates.add((player,state))
            player = self._board._sideToMove
            state = self._board.currBoard()
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