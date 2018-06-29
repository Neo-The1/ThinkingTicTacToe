import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QPointF, QRect
from PyQt5.QtGui import QPen, QBrush, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel
from PyQt5.QtWidgets import QVBoxLayout, QPushButton
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem

#creating a test nn usng tensorflow
import tensorflow as tf
#importing tensorflow example for handwritten digit recognition
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#read data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#set hyper-parameters
learning_rate = 0.5
epochs = 10
batch_size = 100

#define placeholders for input and output
#input is 28x28 images for mnist. 28x28 = 764
x = tf.placeholder(tf.float32,[None,784])
#output is digits- 0 to 9
y = tf.placeholder(tf.float32,[None,10])

#initialize wrights and biases: random initialization
# we will use 2 layers
#Layer-1 : Hidden layer with 300 outputs
W1 = tf.Variable(tf.random_normal([784,300],stddev = 0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
#Layer-2 : Output layer
W2 = tf.Variable(tf.random_normal([300,10],stddev = 0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

#calculate the output of hidden layer, using relu activation
hidden_out = tf.nn.relu(tf.add(tf.matmul(x,W1),b1))

#calculate output of output layer with softmax activation
out = tf.nn.softmax(tf.add(tf.matmul(hidden_out,W2),b2))

#calculate the cross entropy cost
#avoid 0 as log input by clipping
y_clipped = tf.clip_by_value(out,1e-10,0.9999999)
cross_entropy = -tf.reduce_mean( tf.reduce_sum(y*tf.log(y_clipped)
                +(1-y)*tf.log(1-y_clipped),axis=1) )

#add an optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)

#setup the initialization operator
init_op = tf.global_variables_initializer()

#define accurace
#get True or False based on output and y
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(out,1))
#convert correct prediction to float and take a mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#Train
#setup the session
sess = tf.Session()
sess.run(init_op)
total_batch = int(len(mnist.train.labels)/batch_size)
for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
        _,c = sess.run([optimizer,cross_entropy],
                       feed_dict = {x:batch_x,y:batch_y})
        avg_cost +=c /total_batch
    print("Epoch:",(epoch+1)," Cost:","{:.3f}".format(avg_cost))
print(sess.run(accuracy,feed_dict = {x:mnist.test.images, y:mnist.test.labels}))

# ------------------------------------------------------------------------------
# a graphics area to draw
# ------------------------------------------------------------------------------
class tttScene(QGraphicsScene):
    def __init__(self, *args, **kwds):
        QGraphicsScene.__init__(self, *args, **kwds)
        self._canvasSize = kwds.get('size', 520)
        self.setSceneRect(0, 0, self._canvasSize, self._canvasSize)
        self._drawing = False;

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            pos = ev.scenePos()
            self.addEllipse(pos.x()-20, pos.y()-20, 40, 40, QPen(QtCore.Qt.NoPen), QBrush(QtCore.Qt.red))
            self._drawing = True;
            self._prevPoint = pos

    def mouseMoveEvent(self, ev):
        if not self._drawing:
            return

        pos = ev.scenePos()
        self.addLine(self._prevPoint.x(), self._prevPoint.y(), pos.x(), pos.y(), QPen(QtCore.Qt.red, 40, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        self._prevPoint = pos

    def mouseReleaseEvent(self, ev):
        self._drawing = False

    def getGrayScaleImage(self, size):
        img = QImage(size, size, QImage.Format_ARGB32)
        img.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(img)
        self.render(painter)
        img.save("test.png")
        painter.end()
        imgarr = []
        for ii in range(size):
            for jj in range(size):
                c = img.pixelColor(jj, ii).getRgb()
                imgarr.append(c[0])

        return np.array(imgarr).reshape(1, size*size)

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

        w = self._graphicsScene._canvasSize + 200
        h = self._graphicsScene._canvasSize
        self._mainWindow.setFixedSize(QtCore.QSize(w,  h))
        self._mainWindow.setWindowTitle('Digit identifier')

        self._identifyButton = QPushButton(self._mainWindow)
        self._identifyButton.setText("Identify")
        self._identifyButton.clicked.connect(self.OnIdentify)

        self._clearButton = QPushButton(self._mainWindow)
        self._clearButton.setText("Clear")
        self._clearButton.clicked.connect(self.OnClear)

        # display log
        self._displayLabel = QLabel(self._mainWindow)
        self._displayLabel.setText("Paint digits between 0 to 9 on the left")

        # create a layout
        layout = QHBoxLayout()
        layout.addWidget(self._graphicsView)
        sublayout = QVBoxLayout()
        sublayout.addWidget(self._identifyButton)
        sublayout.addWidget(self._clearButton)
        sublayout.addWidget(self._displayLabel)
        layout.addLayout(sublayout)
        self._mainWindow.setLayout(layout)

    def OnClear(self, evt):
        self._graphicsScene.clear()

    def OnIdentify(self, evt):
        inputimg = self._graphicsScene.getGrayScaleImage(28)
        # debug code start
        # writing this to a data file which can be loaded in octave and verified
        f = open("text.dat",'w')
        size = int(np.sqrt(inputimg.shape[1]))
        for ii in range(size):
            for jj in range(size):
                f.write(str(inputimg[0][ii*size+jj]) + ' ')
            f.write('\n')
        f.close()
        # debug code end
        no = sess.run(y_clipped, feed_dict = {x:inputimg})
        print("output layer\n")
        print(no)
        self._displayLabel.setText("Identified as " + str(no.argmax()))

    def show(self):
        self._mainWindow.show()
        sys.exit(self._app.exec_())

gui = tttGui()
gui.show()
a = input()
exit(0)


