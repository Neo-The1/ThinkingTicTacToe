#manual play to see if board is peforming correctly
from tttBoard import tttBoard

board = tttBoard(9)
board.display()
while len(board.legalMoves()) > 0 and not(board.checkWinner()):
    print(board.legalMoves())
    move = int(input("choose from above list of moves"))
    board.makeMove(move)
    board.display()
if (board.checkWinner()):
    if(board._sideToMove == 0): 
        print("X wins!") #if win occured in O's turn, X won!
    else:
        print("O wins!")
else:
    print("It's a Draw!")