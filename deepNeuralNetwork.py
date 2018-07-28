# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class dnNetwork():

    """ class representing a deep neural network with multiple hidden layers
    """
    def __init__(self, *args, **kwds):
        """ a list of dnn layer size's including input and output layer.
            first item in the list refers to the size of input layer.
            last item in the list refers to the size of output layer.
        """
        #currently the code is only for 2 hidden layers, apart from in and out
        self.saveFile = kwds.get('saveFile')
        self._layerSizes = kwds.get('layers', [])
        self._layer1 = keras.layers.Dense(self._layerSizes[1],activation='relu')
        self._layer2 = keras.layers.Dense(self._layerSizes[2],activation='relu')    
        self._outLayer = keras.layers.Dense(self._layerSizes[-1],activation='softmax')
        self._inputs = keras.Input(shape=(self._layerSizes[0],)) #returns placeholder
        x = self._layer1(self._inputs)
        x = self._layer2(x)
        self._outputs = self._outLayer(x)
        self._model = keras.Model(inputs=self._inputs,outputs=self._outputs)
        self._model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


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
        self._model.fit(train_x,train_y,batch_size=32,epochs = 1)
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
#testing on mnist data
if __name__ == "__main__":
    #testing on mnist data
    mnist = input_data.read_data_sets('MNIST_data')
    testNetwork = dnNetwork(layers = [784,64,32,10])
    train_x = mnist.train.images
    train_y = keras.utils.to_categorical(mnist.train.labels)
    test_x = mnist.test.images
    test_y = keras.utils.to_categorical(mnist.test.labels)
    testNetwork.train(train_x,train_y)
    print("evaluation of test net")
    testNetwork.evaluate(test_x,test_y)
    testNetwork.saveModel()
    newNetwork = dnNetwork(layers = [784,64,32,10])
    newNetwork.loadModel()
    predictionsTest = testNetwork.predict(train_x)
    predictionsNew = newNetwork.predict(train_x)
    print("evaluation of new net")
    testNetwork.evaluate(test_x,test_y)
    print("Label: ",np.argmax(train_y[1]))
    print("test Net Prediction: ",np.argmax(predictionsTest[1]))
    print("New Net Prediction: ",np.argmax(predictionsTest[1]))