# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
import sys
from tttBoard import tttBoard
from monteCarlo import monteCarlo
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QPointF, QRect
from PyQt5.QtGui import QPen, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem

# ------------------------------------------------------------------------------
# a graphics scene to draw the board
# ------------------------------------------------------------------------------
class tttScene(QGraphicsScene):
    def __init__(self, *args, **kwds):
        QGraphicsScene.__init__(self, *args, **kwds)
        self._canvasSize = kwds.get('size', 520)
        self._borderMargin = 10
        self._board = kwds.get('board', tttBoard(3))
        self._OImage = QImage("res/O.png")
        self._XImage = QImage("res/X.png")

    def boardSize(self):
        # size of board in pixels
        return self._canvasSize - 2 * self._borderMargin

    def cellSize(self):
        # size of a sigle board cell
        return int(self.boardSize() / self._board._1Dsize)

    def cellCenter(self, cellID):
        col = int(cellID / self._board._1Dsize)
        row = cellID % self._board._1Dsize
        cellSize = self.cellSize()
        x = col * cellSize + 0.5 * cellSize + self._borderMargin
        y = row * cellSize + 0.5 * cellSize + self._borderMargin
        return [x,  y]

    def cellAt(self, x, y):
        cellSize = self.cellSize()
        row = int((x - self._borderMargin) / cellSize)
        col = int((y - self._borderMargin) / cellSize)
        return row * self._board._1Dsize + col

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            pos = ev.scenePos()
            cellID = self.cellAt(pos.x(), pos.y())
            if not self.makeMove(cellID):
                return
            self.update()
            # follows engine's play
            mc = monteCarlo(self._board)
            self.makeMove(mc.getMove())

    def makeMove(self, move):
        if move >= self._board._boardSize       or\
           move not in self._board.legalMoves() or\
           self._board.winner() > 0:
           # illegal move
           return False

        image = None
        sideToMove = self._board.currPlayer()
        if sideToMove == 1:
            image = self._OImage
        if sideToMove == 2:
            image = self._XImage

        self._board.makeMove(move)

        playerIcon = QGraphicsPixmapItem(QPixmap.fromImage(image))
        playerIcon.setScale(0.15)
        cellID = move
        cc = self.cellCenter(cellID)
        cellSize = self.cellSize()
        playerIcon.setPos(int(cc[0] - 0.25 * cellSize), int(cc[1] - 0.25 * cellSize))
        self.addItem(playerIcon)
        return True

    def drawCells(self):
        cellSize = self.cellSize()
        pen = QPen(QtCore.Qt.black)
        pen = QPen(QtCore.Qt.black)
        upperLeft  = QPointF(self._borderMargin, self._borderMargin)
        lowerRight = QPointF(self._canvasSize - self._borderMargin, self._canvasSize - self._borderMargin)
        rect = QtCore.QRectF(upperLeft, lowerRight)
        self.addRect(rect, pen)
        for row in range(self._board._1Dsize):
            for col in range(self._board._1Dsize):
                upperLeft  = QPointF(col * cellSize + self._borderMargin, row * cellSize + self._borderMargin)
                lowerRight = QPointF((col + 1) * cellSize + self._borderMargin, (row + 1) * cellSize + self._borderMargin)
                rect = QtCore.QRectF(upperLeft, lowerRight)
                self.addRect(rect, pen)

# ------------------------------------------------------------------------------
# widget to display board graphics
# ------------------------------------------------------------------------------
class tttGui:

    def __init__(self):
        # size parameters
        # create a qt application
        self._app = QApplication(sys.argv)

        # set up a main window
        self._mainWindow = QDialog()

        # create a graphics area in main window
        self._graphicsScene = tttScene()
        self._graphicsView = QGraphicsView(self._mainWindow)
        self._graphicsView.setMouseTracking(True)
        h = self._graphicsScene._canvasSize
        self._graphicsView.setGeometry(QRect(0, 0, h, h))
        self._graphicsView.setScene(self._graphicsScene)
        self._graphicsScene.drawCells()

        w = self._graphicsScene._canvasSize + 200
        h = self._graphicsScene._canvasSize
        self._mainWindow.setFixedSize(QtCore.QSize(w,  h))
        self._mainWindow.setWindowTitle('Thinking Tic-Tac-Toe')

        # display log
        self._displayLabel = QLabel(self._mainWindow)
        self._displayLabel.setText("TODO:\n Area to display Messages")

        # create a layout
        layout = QHBoxLayout()
        layout.addWidget(self._graphicsView)
        layout.addWidget(self._displayLabel)
        self._mainWindow.setLayout(layout)

    def show(self):
        self._mainWindow.show()
        sys.exit(self._app.exec_())
