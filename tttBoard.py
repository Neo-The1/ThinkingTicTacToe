#tic tac toe board will be indexed in this convention:
# left to right starting from 0. See example: 2x2 board 
#0 1
#2 3  
#Player O is 1 and Player X is 2
#history of moves is stored in a list as 'O2','X0' etc.
#the state will be a nxn string made by joining the list
import numpy as np
class tttBoard:

    # Initialize the board, needs n to create nxn board
    def __init__(self, n):
        #board is a size nxn list containing 0,1 or 2 depending upon whether 
        #it is empty, O or X respectively
        self._board = [0]*n*n
        self._state = "".join([str(p) for p in self._board])
        #history contains the list of all moves
        self._history = []
        self._boardSize = n*n
        self._1Dsize = n

    def decodeState(self, s):
        """ takes state list of array boardSize and returns
            a list of length 2*boardSize + 1 elements
            first boardSize represent O's position
            second boardSize represent X's position
            last element is player to make next move
        """
        state = np.zeros((1, 2 * self._boardSize + 1))
        numOccupiedSq = 0;
        for i in range(self._boardSize):
            if s[i] == '1':
                state[0, i] = 1
                numOccupiedSq += 1;
            elif s[i] == '2':
                state[0, self._boardSize+i] = 1
                numOccupiedSq += 1;
        # every even's turn is O's move
        if numOccupiedSq % 2 == 0:
            state[0, -1] = 1
        else:
            state[0, -1] = 2
        return state

    def currPlayer(self):
        if len(self._history)==0:
            #if fist move, currPlayer is 1 = O
            return 1
        if 'O' in self._history[-1] :
            #if last move was made by O, current player is X
            return 2 
        else:
            return 1

    def opponent(self, player):
        """ returns opponent of passed player """
        if player == 1:
            return 2
        if player == 2:
            return 1
        return None

    # Print the board
    def display(self):
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
                boardString += ". "
        print(boardString)

    # Generate all possible moves from current board state
    # A move is an integer position for Boardsq at which move is to be made
    def legalMoves(self):
        legalmoves = []
        for i in range(self._boardSize):
            boardSq = self._board[i]
            if boardSq == 0: #if position empty
                legalmoves.append(i)
        return legalmoves

    # Make the passed move on the board for the side whose
    # turn it is. After making the move update the side to
    # make next move
    def makeMove(self, move):
        assert( move in self.legalMoves())
        self._board[move] = self.currPlayer()
        if self.currPlayer() == 1:
            self._history.append('O'+str(move))
        else:
            self._history.append('X'+str(move))
        return self._board

    def playerAt(self, cell):
        """ returns id of player occupying cell
        """
        assert(cell >= 0 and cell < self.getSize() * self.getSize())
        return self._board[cell]

    def getSize(self):
        return self._1Dsize

    def getState(self):
        return "".join([str(p) for p in self._board])

    def getStateAfterMove(self, move):
        boardcopy = self._board[:]
        boardcopy[move] = self.currPlayer()
        return "".join([str(p) for p in boardcopy])

    #Check the winner by checking all rows, all columns and then 2 diagonals
    # we will assume indexing of positons in board and corresponding in integer
    #evalBoard is one player's board
    #a n-bit integer with 0s at empty places and 1 at places occupied by player

    def checkWin(self,evalBoard):
        # function to check if a given bit in n-bit is 1 or not
        def testBit(num, bitpos):
            return ( num & ( 1 << bitpos ) )
        #we set these values to be true and later and them with test bit for
        #locations we need to check. If unoccupied, they will turn False
        oneDBoardSize = self._1Dsize
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

    #To check winner, we create a n bit integer for each player with 0s at 
    #empty places and 1s at places player occupies
    def winner(self):
        Oboard = 0
        Xboard = 0
        for ii in range(self._boardSize):
            if self._board[ii] == 1:
                Oboard += 2**ii
            if self._board[ii] == 2:
                Xboard += 2**ii
        winner = 0
        if self.checkWin(Oboard):
            winner = 1
        elif self.checkWin(Xboard):
            winner = 2
        else:
            if not self.legalMoves():
                winner = -1
        return winner
