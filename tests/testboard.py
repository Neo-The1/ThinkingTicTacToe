#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import os, sys
rootDir = os.getcwd()
rootDir = rootDir.split('/')
rootDir.pop()
rootDir = '/'.join(rootDir)

sys.path.insert(0, rootDir)
from tttBoard import tttBoard
from movePolicy import randomMovePolicy

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def testBoard(size):
    board = tttBoard(size)
    assert( board.getSize() == size )
    assert( board.opponent(0) == 1 )
    assert( board.opponent(1) == 0 )
    assert( board.legalMoves() == [i for i in range(board.getSize()**2)] )

    assert( board.currPlayer() == 0 )

    assert( board.makeMove(0) )
    assert( board.currPlayer() == 1 )
    assert( board.legalMoves() == [i for i in range(1, board.getSize()**2)] )
    assert( board.winner() == -1 )
    assert( not board.isGameOver() )

    assert( board.makeMove(board.getSize()**2 - 1) )
    assert( board.currPlayer() == 0 )
    assert( board.legalMoves() == [i for i in range(1, board.getSize()**2 - 1)] )
    assert( board.winner() == -1 )
    assert( not board.isGameOver() )
    return True

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def testSelfPlay(size):
    print('Starting game on a ' + str(size) + ' x ' + str(size) + ' board')
    board = tttBoard(size)

    movePolicy = randomMovePolicy()
    board.selfPlay(movePolicy)

    board.display()
    if board.winner() == -1:
        print('Game over, it is a draw')
    elif board.winner() == 0:
        print('Game over, O wins')
    elif board.winner() == 1:
        print('Game over, X wins')
    else:
        return False
    return True

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == '__main__':
    if testBoard(3) and testBoard(4) and testBoard(5) and\
       testSelfPlay(3) and testSelfPlay(4) and testSelfPlay(5):
        print('All passed')
