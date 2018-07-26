# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
from tensorflow import keras
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
        self.saveFile = kwds.get('saveFile')
        self._layerSizes = kwds.get('layers', [])
        self._layer1 = keras.layers.Dense(self._layerSizes[1],activation='relu')
        self._layer2 = keras.layers.Dense(self._layerSizes[2],activation='relu')    
        self._outLayer = keras.layers.Dense(self._layerSizes[0],activation='softmax')
        self._inputs = keras.Input(shape=(self._layerSizes[0],)) #returns placeholder
        x = self._layer1(self._inputs)
        x = self._layer2(x)
        self._outputs = self._outLayer(x)
        self._model = keras.Model(inputs=self._inputs,outputs=self._outputs)
        self._model.compile(optimizer=tf.train.AdamOptimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


    def loadFromFile(self):
        """ Load the network parameters from a file
        """
        self._model.load_weights('my_model')
        return None

    def saveToFile(self):
        """ Save the network parameters to a file
        """
        self._model.save_weights('./my_model')
        return None

    def train(self, trainData,trainLabels):
        """ Train the network using passed training data as numpy array
        """
        self._model.fit(trainData,trainLabels,batch_size=1,epochs=1)
    
    def predict(self,inputState):
        return None

#    def evaluate(self, evalData):
#        """ evaluate accuracy
#        """
#        eval_fn = 
#        accuracy_score = self._estimator.evaluate(input_fn=self._testInputFn)["accuracy"]
#        print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
