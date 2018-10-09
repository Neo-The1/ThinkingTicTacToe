#code to play with AI
from tttBoard import tttBoard
from monteCarlo import monteCarlo
board = tttBoard(3)
board.display()
def checkWin(board):
    if (board.winner()):
        if(board.winner()==1):
            print("O wins!")
            return 1
        elif(board.winner()==2):
            print("X wins!")
            return 1
        elif(board.winner()==-1):
            print("It's a Draw!")
            return 1
    else:
        return 0
        
while len(board.legalMoves()) > 0 and not(checkWin(board)):
    print(board.legalMoves())
    pMove = int(input("choose from above list of moves"))
    board.makeMove(pMove)
    board.display()
    if checkWin(board):
        break
    print('Thinking...')
    ttt = monteCarlo(board)
    tttMove = ttt.getMove()
    board.makeMove(tttMove)
    print('Thinking Tic Tac Toe move...')
    board.display()
    if checkWin(board):
        break
