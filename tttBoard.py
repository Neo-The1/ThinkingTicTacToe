#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# tic tac toe board will be indexed in this convention:
# left to right starting from 0. See example: 2x2 board 
# 0 1
# 2 3
# Players are 0 and 1
# the state will be a nxn string made by joining the list
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import numpy as np
class tttBoard:
    """ Initialize the board, needs n to create nxn board
    """
    def __init__(self, n):
        """ board is a size nxn list containing 0, 1 or 2 depending upon whether
            it is empty, O or X respectively
        """
        assert(n > 2)
        self._board     = [0] * n * n
        self._boardSize = n * n
        self._1Dsize    = n
        self._winner    = -1

    def currPlayer(self):
        return _player

    def opponent(self, player):
        """ returns opponent of passed player
        """
        assert(player == 0 or player == 1)
        return (player ^ 1)

    def display(self):
        """ Print the board
        """
        boardString = ""
        for ii in range(self._boardSize):
            boardSq = self._board[ii]
            if ii % self._1Dsize == 0:
                boardString += "\n"
            if boardSq == 1:
                boardString += "O "
            elif boardSq == 2 :
                boardString += "X "
            else:
                assert(boardSq == 0)
                boardString += ". "
        print(boardString)

    def legalMoves(self):
        """ Generate all possible moves from current board state
            A move is an integer position for boardsq at which move is to be made
        """
        return [i for i in range(self._boardSize) if self._board[i] == 0]

    def makeMove(self, move):
        """ Make the passed move on the board for the side whose
            turn it is. After making the move update the side to
            make next move
        """
        assert(move < self._boardSize and self._board[move] == 0)
        if not isGameOver():
            self._board[move] = self.currPlayer() + 1
            self.checkWinPrivate(move)
            self._player ^= 1

    def getSize(self):
        return self._1Dsize

    def isGameOver(self,evalBoard):
        """ check if a player occupies all of a row/col/diagonal or
            there are no more moves left for a player
        """
       return self.winner() in [0, 1] or (self._board.count(0) == 0)

    def winner(self):
        """ see if there is a winner if game is over
        """
        return self._winner

    def fetchRowPrivate(self, row):
        """ extract a row
        """
        return self._board[row * self.getSize() : (row + 1) * self.getSize()]

    def fetchColPrivate(self, col):
        """ extract a col
        """
        return [self._board[i] for i in range(self._boardSize) if i % getSize() == col]

    def fetchDiagonalsPrivate(self):
        """ extract both diagonals
        """
        return self._board[row * self.getSize() : (row + 1) * self.getSize()]

    def allNonZeroAndSamePrivate(self, lst):
        """ checks if all entries in the passed list are same and non zero
        """
        return lst[1:] == lst[:-1] and lst[0]

    def checkWin(self, movepos):
        """ caches a player who won after a move at movepos was made
        """
        movrow = int(movepos / self.getSize())
        movrowitems = self.fetchRowPrivate(movrow)
        if self.allNonZeroAndSamePrivate(moverowitems):
            self._winner = moverowitems[0]
            return
        movcol = int(movepos % self.getSize())
        movcolitems = self.fetchColPrivate(movcol)
        if self.allNonZeroAndSamePrivate(movecolitems):
            self._winner = movecolitems[0] - 1
            return
        if movrow == movcol:
            diagonals = self.fetchDiagonalsPrivate()
            if self.allNonZeroAndSamePrivate(diagonals[0]):
                self._winner = diagonals[0][0] - 1
                return
            if self.allNonZeroAndSamePrivate(diagonals[1]):
                self._winner = diagonals[1][0] - 1
                return

        assert(self._winner int [-1, 0, 1])

