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
        begin = datetime.datetime.utcnow() #gets current time
        #run the simulation till the specified time
        while datetime.datetime.utcnow() - begin < self._simTime:
            self.runSimulation
        
    #playout a random game and update the statistics table
    def runSimulation(self):
        expandTree =True
        visitedPositions = set()    
        player = self._board.sideToMove
        if player == 0:
            playerBoard = self._board.Oboard
        else :
            playerBoard = self._board.Xboard
            
        for x in range(self._maxMoves):
            #play randomly
            legalMoves = self._board.legalMoves(self._board)
            move = choice(legalMoves)
            self._board.makeMove(move)

            #if this is a new leaf, set statistics to 0
            if expandTree and (player,playerBoard) not in self._plays:
                expandTree = False
                self._plays[(player,playerBoard)] = 0
                self._wins[(player,playerBoard)] = 0
            
            #add the current position to visited and set boards
            visitedPositions.add((player,playerBoard))
            player = self._board._sideToMove
                
            if self._board.checkWinner():
                
                break
        
        