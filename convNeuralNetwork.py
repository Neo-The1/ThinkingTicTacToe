# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow import keras
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class cnNetwork():

    """ class representing a convolutional neural network with multiple hidden layers
    """
    def __init__(self, inputShape,outputSize,**kwds):
        self._optimizer = kwds.get('optimizer',tf.train.AdamOptimizer())
        self._inputShape = inputShape
        self._outputSize= outputSize
        self._layer1 = keras.layers.Conv2D(16,kernel_size=(3,3),strides=(1,1),
                                           padding='same',kernel_initializer='glorot_uniform',
                                           kernel_regularizer=keras.regularizers.l2(0.0001))
        self._layer2 = keras.layers.BatchNormalization()
        self._layer3 = keras.layers.Activation('relu')
#        self._layerPool = keras.layers.MaxPool2D(pool_size=(2,2),strides=2)
        self._layer4 = keras.layers.Conv2D(16,kernel_size=(3,3),strides=(1,1),
                                           padding='same',kernel_initializer='glorot_uniform',
                                           kernel_regularizer=keras.regularizers.l2(0.0001))
        self._layer5 = keras.layers.BatchNormalization()
        self._layer6 = keras.layers.Activation('relu')
        self._layerFlat = keras.layers.Flatten()
        self._piLayer = keras.layers.Dense(self._outputSize-1,
                                           kernel_initializer='glorot_uniform',
                                           kernel_regularizer=keras.regularizers.l2(0.0001),
                                           activation='relu')
#        self._zLayer = keras.layers.Dense(1,activation='tanh')
        self._inputs = keras.Input(shape=self._inputShape) #returns placeholder
        x = self._layer1(self._inputs)
        x = self._layer2(x)
        x = self._layer3(x)
        x = self._layer4(x)
        x = self._layer5(x)
        x = self._layer6(x)
        x = self._layerFlat(x)
        self._outPi = self._piLayer(x)
#        self._outZ = self._zLayer(x)
#        self._model = keras.Model(inputs=self._inputs,outputs=[self._outPi,self._outZ])
        self._model = keras.Model(inputs=self._inputs,outputs=self._outPi)
        self._model.compile(optimizer=tf.train.AdamOptimizer(0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self._epochSize = 128
        
    def loss(self,yTrue,yPred):
        z = keras.backend.flatten(yTrue[-1])
        v = keras.backend.flatten(yPred[-1])
        pi = keras.backend.flatten(yTrue[0])
        p = keras.backend.flatten(yPred[0])
        loss = keras.backend.square(z-v)
        - keras.backend.sum(keras.backend.transpose(pi)*keras.backend.log(p),axis=-1,keepdims=True)
        return loss

    def loadModel(self):
        """ Load the network parameters from a file
        """
        self._model.load_weights('my_cnn_model')
        return None

    def saveModel(self):
        """ Save the network parameters to a file
        """
        self._model.save_weights('./my_cnn_model')
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
    from tttBoard import tttBoard
    board = tttBoard(3)
    board.makeMove(5)
    states = np.zeros((1,19))
    testNet = cnNetwork((3,3,7),10)
#    states[0,:] = board.decodeState(board.getState())
#    print(states)
#    print(testNet.predict(states))
    result = testNet.predict(board.decodeStateCNN(board._stateHistory))
    print(result[0].flatten())
#    print(result[1].flatten())
    print(np.argmax(result[0].flatten()))
    board.makeMove(7)
    result = testNet.predict(board.decodeStateCNN(board._stateHistory))
    print(result[0].flatten())
#    print(result[1].flatten())
    print(np.argmax(result[0].flatten()))
    print(result[0].shape)
    testNet.saveModel()