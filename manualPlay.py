#manual play to see if board is peforming correctly
#Copying everything from tttBoard.py so that this is a standalone code
#***************** BEGIN COPY tttBoard.py*************************************
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
# NOTES
# Size of the board is n * n
# An integer of size n*n will be sufficient to store some information about it.
# Consider the binary representation of this integer. Each bit can store info
# about a square. So bit 0 can represent square 0, bit 1 can represent square 1
# and so on... Note bits can only be true or false... So here in this scenario
# lets say bits will inform us that whether or not the square is empty or
# occupied. We will need two integers here for each player ( player_O and
# player_X ). Integer for player_O will let us know of what squares are
# occupied by player_O and likewise for player_X. Some assertions are
# assumed here that same bit in both the integers can not be true at the same
# time; as that would imply that same square is occupied by both players.
# BOARD OPERATIONS
# To make a move for a player at square s we just need to turn the corresponding
# bit in player's integer on. To undo a move made on a square we turn it off.
# ------------------------------------------------------------------------------
class tttBoard:

    # Initialize the board
    def __init__(self, boardSize):
        self._Oboard =  0           # bitboard for player zero
        self._Xboard = 0           # bitboard for player cross
        self._boardSize = boardSize     # size of board i.e. number of squares
        self._sideToMove = 0            # side to make move

    # Print the board
    def display(self):
        boardString = ""
        oneDBoardSize = math.sqrt(self._boardSize)
        for boardSq in range(self._boardSize):
            if boardSq and boardSq % oneDBoardSize == 0:
                boardString += "\n"
            if TestBit(self._Oboard, boardSq):
                boardString += "O "
            elif TestBit(self._Xboard, boardSq):
                boardString += "X "
            else:
                boardString += ". "
        print(boardString)

    # Generate all possible moves from current board state
    # A move is an integer with single bit on for the square
    # at which move is to be made
    def legalMoves(self):
        legalmoves = []
        curr_board = self._Oboard | self._Xboard
        for boardSq in range(self._boardSize):
            if(not TestBit(curr_board,boardSq)):
                legalmoves.append(2**boardSq)
        return legalmoves

    # Make the passed move on the board for the side whose
    # turn it is. After making the move update the side to
    # make next move
    def makeMove(self, move):
        if(self._sideToMove == 0):
            self._Oboard = self._Oboard | move
            self._sideToMove = 1
        else:
            self._Xboard = self._Xboard | move
            self._sideToMove = 0
        return None
    
    #check if player O r player X have won
    def check_win(self):
        return None
        
#******** END Copy tttBoard.py************************************************
        
board = tttBoard(3)
board.display()
while len(board.legalMoves()) > 0:
    print(board.legalMoves())
    move = int(input("choose from above list of moves"))
    board.makeMove(move)
    board.display()    