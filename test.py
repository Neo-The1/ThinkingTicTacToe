from tttBoard import tttBoard

def makemove(board, player, square):
    assert square >= 1 and square <= 9
    if player == 0:
        board._zerosboard |= (1 << (square - 1))
    elif player == 1:
        board._crosssboard |= (1 << (square - 1))
    else:
        assert false
    board.display()

myboard = tttBoard(9)
myboard.display()

print("making move for zero on square 1")
makemove(myboard, 0, 1)
print("making move for cross on square 9")
makemove(myboard, 1, 9)

print("making move for zero on square 3")
makemove(myboard, 0, 3)
print("making move for cross on square 7")
makemove(myboard, 1, 7)

print("making move for zero on square 2")
makemove(myboard, 0, 2)
