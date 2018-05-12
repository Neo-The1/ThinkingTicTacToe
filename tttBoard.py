# ------------------------------------------------------------------------------
# import required modules
# ------------------------------------------------------------------------------
import math

# ------------------------------------------------------------------------------
# Tests if bitpos in num is on
# ------------------------------------------------------------------------------
def TestBit(num, bitpos):
    return ( num & ( 1 << bitpos ) )

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class tttBoard:

    # Initialize the board
    def __init__(self, boardSize):
        self._zerosboard =  0           # bitboard for player zero
        self._crosssboard = 0           # bitboard for player cross
        self._boardSize = boardSize     # size of board i.e. number of squares
        self._sideToMove = 0            # side to make move

    # Print the board
    def display(self):
        boardString = ""
        oneDBoardSize = math.sqrt(self._boardSize)
        for boardSq in range(self._boardSize):
            if boardSq and boardSq % oneDBoardSize == 0:
                boardString += "\n"
            if TestBit(self._zerosboard, boardSq):
                boardString += "O "
            elif TestBit(self._crosssboard, boardSq):
                boardString += "X "
            else:
                boardString += ". "
        print(boardString)

    # Generate all possible moves from current board state
    # A move is an integer with single bit on for the square
    # at which move is to be made
    def generateMoves(self):
        return None

    # Make the passed move on the board for the side whose
    # turn it is. After making the move update the side to
    # make next move
    def makeMove(self, move):
        return None
