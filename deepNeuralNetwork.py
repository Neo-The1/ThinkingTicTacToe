# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
from tensorflow import keras
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class dnNetwork():

    """ class representing a deep neural network with multiple hidden layers
    """
    def __init__(self,inputSize,outputSize, *args, **kwds):
        """ a list of dnn layer size's including input and output layer.
            first item in the list refers to the size of input layer.
            last item in the list refers to the size of output layer.
        """
        #currently the code is only for 2 hidden layers, apart from in and out
        self._saveFile = kwds.get('saveFile')
        self._inputSize = inputSize
        self._outputSize= outputSize
        self._layer1 = keras.layers.Dense(128,activation='relu')
        self._layer2 = keras.layers.Dense(64,activation='relu')    
        self._layer3 = keras.layers.Dense(128,activation='relu')
        self._piLayer = keras.layers.Dense(self._outputSize-1,activation='softmax')
#        self._zLayer = keras.layers.Dense(1,activation='tanh')
        self._inputs = keras.Input(shape=(self._inputSize,)) #returns placeholder
        x = self._layer1(self._inputs)
        x = self._layer2(x)
        x = self._layer3(x)
        self._outPi = self._piLayer(x)
#        self._outZ = self._zLayer(x)
#        self._model = keras.Model(inputs=self._inputs,outputs=[self._outPi,self._outZ])
        self._model = keras.Model(inputs=self._inputs,outputs=self._outPi)
        self._model.compile(optimizer=keras.optimizers.Adam(lr=0.005, beta_1=0.99, beta_2=0.999, epsilon=1e-10, decay=0.0005),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        self._epochSize = 128

    def loss(self,yTrue,yPred):
        z = keras.backend.flatten(yTrue[-1])
        v = keras.backend.flatten(yPred[-1])
        pi = keras.backend.flatten(yTrue[0])
        p = keras.backend.flatten(yPred[0])
        loss = 0.0*keras.backend.square(z-v)
        - 0*keras.backend.sum(keras.backend.transpose(pi)*keras.backend.log(p),axis=-1,keepdims=True)
        return 0
    
    def loadModel(self):
        """ Load the network parameters from a file
        """
        self._model.load_weights('my_model')
        return None

    def saveModel(self):
        """ Save the network parameters to a file
        """
        self._model.save_weights('./my_model')
        return None

    def train(self, train_x,train_y):
        """ Train the network using passed training data as numpy array
        """
        self._model.fit(train_x,train_y,batch_size=8,epochs = self._epochSize)
        return None
    
    def predict(self,x):
        """Predict the output, given input
        """
        return self._model.predict(x)

    def evaluate(self,test_x,test_y):
        """ evaluate accuracy
        """
        evalLoss, evalAcc = self._model.evaluate(test_x,test_y)
        print("Evaluation Accuracy :",evalAcc)
        return None
        
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == "__main__":
#    #testing on mnist data
#    mnist = input_data.read_data_sets('MNIST_data')
#    testNetwork = dnNetwork(layers = [784,64,32,10])
#    train_x = mnist.train.images
#    train_y = keras.utils.to_categorical(mnist.train.labels)
#    test_x = mnist.test.images
#    test_y = keras.utils.to_categorical(mnist.test.labels)
#    testNetwork.train(train_x,train_y)
#    print("evaluation of test net")
#    testNetwork.evaluate(test_x,test_y)
#    testNetwork.saveModel()
#    newNetwork = dnNetwork(layers = [784,64,32,10])
#    newNetwork.loadModel()
#    predictionsTest = testNetwork.predict(train_x)
#    predictionsNew = newNetwork.predict(train_x)
#    print("evaluation of new net")
#    testNetwork.evaluate(test_x,test_y)
#    print("Label: ",np.argmax(train_y[1]))
#    print("test Net Prediction: ",np.argmax(predictionsTest[1]))
#    print("New Net Prediction: ",np.argmax(predictionsTest[1]))
    from tttBoard import tttBoard
    board = tttBoard(3)
    board.makeMove(5)
    states = np.zeros((1,19))
    testNet = dnNetwork(19,10)
#    states[0,:] = board.decodeState(board.getState())
#    print(states)
#    print(testNet.predict(states))
    result = testNet.predict(board.decodeState(board.getState()))
    print(result[0].flatten())
    testNet.saveModel()
#    print(result[1].flatten())