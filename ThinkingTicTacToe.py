#code to play with AI
from tttBoard import tttBoard
from MonteCarlo import MonteCarlo
pBoard = tttBoard(3)
tttBoard = tttBoard(3)
pBoard.display()
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
        
while len(pBoard.legalMoves()) > 0 and not(checkWin(pBoard)):
    print(pBoard.legalMoves())
    pMove = int(input("choose from above list of moves"))
    pBoard.makeMove(pMove)
    print('your move...')
    pBoard.display()
    if checkWin(pBoard):
        break
    board_copy = list(pBoard._board)
    tttBoard._board = board_copy
    print('Thinking...')
    ttt = MonteCarlo(tttBoard)
    tttMove = ttt.getMove()
    pBoard.makeMove(tttMove)
    print('Thinking Tic Tac Toe move...')
    pBoard.display()
    if checkWin(pBoard):
        break

