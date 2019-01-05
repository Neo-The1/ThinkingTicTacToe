#manual play to see if board is peforming correctly
from tttBoard import tttBoard

board = tttBoard(3)
board.display()
while len(board.legalMoves()) > 0 and not(board.winner()):
    print(board.legalMoves())
    move = int(input("choose from above list of moves"))
    board.makeMove(move)
    board.display()
    
if (board.winner()):
    if(board.winner()==1): 
        print("O wins!")
    elif(board.winner()==2):
        print("X wins!")
    elif(board.winner()==-1):
        print("It's a Draw!")