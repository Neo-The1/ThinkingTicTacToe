# ------------------------------------------------------------------------------
# import required modules
# ------------------------------------------------------------------------------
import math

# ------------------------------------------------------------------------------
# Tests if bitpos in num is on
# ------------------------------------------------------------------------------
def testBit(num, bitpos):
    return ( num & ( 1 << bitpos ) )

# ------------------------------------------------------------------------------
# NOTES
# Size of the board is n * n
# An integer of size n*n will be sufficient to store some information about it.
# Consider the binary representation of this integer. Each bit can store info
# about a square. So bit 0 can represent square 0, bit 1 can represent square 1
# and so on... Note bits can only be true or false... So here in this scenario
# lets say bits will inform us that whether or not the square is empty or
# occupied. We will need two integers here for each player ( player O and
# player X). Integer for player zero will let us know of what squares are
# occupied by player 0 and likewise for player cross. Some assertions are
# assumed here that same bit in both the integers can not be true at the same
# time; as that would imply that same square is occupied by both players.
# BOARD OPERATIONS
# To make a move for a player at square s we just need to turn the corresponding
# bit in player's integer on. To undo a move made on a square we turn it off.
# ------------------------------------------------------------------------------
class tttBoard:

    # Initialize the board
    def __init__(self, boardSize):
        self._Oboard =  0               # bitboard for player zero
        self._Xboard = 0                # bitboard for player cross
        self._boardSize = boardSize     # size of board i.e. number of squares
        self._sideToMove = 1            # side to make move, 1 for O 2 for X

    # Print the board
    def display(self):
        boardString = ""
        oneDBoardSize = int(math.sqrt(self._boardSize))
        for boardSq in range(self._boardSize):
            if boardSq and boardSq % oneDBoardSize == 0:
                boardString += "\n"
            if testBit(self._Oboard, boardSq):
                boardString += "O "
            elif testBit(self._Xboard, boardSq):
                boardString += "X "
            else:
                boardString += ". "
        print(boardString)
    
    #current board
    def currBoard(self):
        return self._Xboard | self._Oboard
    # Generate all possible moves from current board state
    # A move is an integer with single bit on for the square
    # at which move is to be made
    def legalMoves(self):
        legalmoves = []
        for boardSq in range(self._boardSize):
            if not testBit(self.currBoard(), boardSq):
                legalmoves.append(2**boardSq)
        return legalmoves

    # Make the passed move on the board for the side whose
    # turn it is. After making the move update the side to
    # make next move
    def makeMove(self, move):
        if self._sideToMove == 1:
            self._Oboard = self._Oboard | move
            self._sideToMove = 2
        else:
            self._Xboard = self._Xboard | move
            self._sideToMove = 1            
    
    #Check the winner by checking all rows, all columns and then 2 diagonals
    # we will assume indexing of positons in board and corresponding in integer
    #representation of each player as per following 2x2 example
    #Board
    #0 1
    #2 3
    # player integer positions= 3 2 1 0
    
    def checkWin(self,evalBoard):
        oneDBoardSize = int(math.sqrt(self._boardSize))            
        #we set these values to be true and later and them with test bit for
        #locations we need to check. If unoccupied, they will turn False
        winDiag1 = True
        winDiag2 = True
        #check diagonals first
        for jj in range(oneDBoardSize):
            winDiag1 = winDiag1 and testBit(evalBoard,jj*(oneDBoardSize+1))
            winDiag2 = winDiag2 and testBit(evalBoard,(jj+1)*(oneDBoardSize-1))
        if winDiag1 or winDiag2:
            return True
        #check rows and columns
        for ii in range(oneDBoardSize):
            winRow = True
            winCol = True
            startRow = ii*oneDBoardSize 
            startCol = ii
            row  = startRow
            col =  startCol           
            for jj in range(oneDBoardSize):
                #win will be set to False if testing pos is empty
                #check rows
                row = startRow+jj
                winRow = winRow and testBit(evalBoard,row)
                #check column
                col = startCol + jj*oneDBoardSize
                winCol = winCol and testBit(evalBoard,col)
            if winRow or winCol:
                return True
        # if no win occured
        return False
    
    def winner(self):
        winner = 0
        if self.checkWin(self._Oboard):
            winner = 1
        elif self.checkWin(self._Xboard):
            winner = 2
        else:
            if not self.legalMoves():
                winner = -1
        return winner
            
            