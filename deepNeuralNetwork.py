# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
import tensorflow as tf
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
        self._layerSizes = kwds.get('layers', [])
        self._optimizer = kwds.get('optimizer', tf.train.AdamOptimizer(1e-4))
        assert len(self._layerSizes) > 2
        # placeholder for input features
        self._featureColumns = [tf.feature_column.numeric_column('p', shape=10)]
        """ now initialize the underlying estimator
        """
        self._estimator = tf.estimator.DNNClassifier( feature_columns = self._featureColumns,
                                                    hidden_units = [layerSize for layerSize in self._layerSizes[1:-1]],
                                                    optimizer = self._optimizer,
                                                    n_classes = 10,
                                                    dropout = 0.1,
                                                    model_dir = './tmp/dNNmodel' )



    def loadFromFile(self, filename):
        """ Load the network parameters from a file
        """
        return None

    def saveToFile(self, fileName):
        """ Save the network parameters to a file
        """
        return None

    def train(self, trainingData):
        """ Train the network using passed training data
        """
        inputFn = tf.estimator.inputs.numpy_input_fn(trainingData,
                                                     y = None,
                                                     num_epochs=None,
                                                     batch_size = 1,
                                                     shuffle = True)
        self._estimator.train( input_fn = inputFn, steps = 2000 )
    
    def predict(self,inputState):
        prediction = self._estimator.predict(input_fn = inputState)
        return prediction

#    def evaluate(self, evalData):
#        """ evaluate accuracy
#        """
#        eval_fn = 
#        accuracy_score = self._estimator.evaluate(input_fn=self._testInputFn)["accuracy"]
#        print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
