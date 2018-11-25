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
def testHash(size):
    board = tttBoard(size)

    hashlist = []
    movePolicy = randomMovePolicy()
    while not board.isGameOver():
        hashlist.append(board.__hash__())
        assert(board.makeMove(movePolicy.getMove(board)))

    return len(hashlist) == len(set(hashlist))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def testHashPerf(size):
    board = tttBoard(size)
    board.__hash__()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == '__main__':
    print("Running sanity tests")
    if testBoard(3)    and testBoard(4)    and testBoard(5)   and\
       testHash(3)     and testHash(4)     and testHash(5)    and\
       testSelfPlay(3) and testSelfPlay(4) and testSelfPlay(5):
        print('All passed')

    import timeit
    numIterations = 1000000
    print("Running performance tests")
    print("Time spent hashing 5 x 5 board 1000000 times")
    print(timeit.timeit("testHashPerf(5)", setup="from __main__ import testHashPerf", number = numIterations))


