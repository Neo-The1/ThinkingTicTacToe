#code to play with AI
from tttBoard import tttBoard
from MonteCarlo import MonteCarlo
board = tttBoard(3)
board.display()
while len(board.legalMoves()) > 0 and not(board.winner()):
    print(board.legalMoves())
    p_move = int(input("choose from above list of moves"))
    board.makeMove(p_move)
    print('your move...')
    board.display()
    ttt = MonteCarlo(board)
    ttt_move = ttt.getMove()
    board.makeMove(ttt_move)
    print('Thinking Tic Tac Toe move...')
    board.display()
    
if (board.winner()):
    if(board.winner()==1):
        print("O wins!")
    elif(board.winner()==2):
        print("X wins!")
    elif(board.winner()==-1):
        print("It's a Draw!")
