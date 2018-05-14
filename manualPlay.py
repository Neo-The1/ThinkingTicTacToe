#manual play to see if board is peforming correctly
from tttBoard import tttBoard

board = tttBoard(9)
board.display()
while len(board.legalMoves()) > 0:
    print(board.legalMoves())
    move = int(input("choose from above list of moves"))
    board.makeMove(move)
    board.display()
