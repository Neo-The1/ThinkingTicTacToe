# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf

# TODO: to be removed from this file. This is being kept here just to test
# the dnNetwork implementation
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)

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
        self._featureColumns = [tf.feature_column.numeric_column('x', shape=[28, 28])]
        """ now initialize the underlying estimator
        """
        self._network = tf.estimator.DNNClassifier( feature_columns = self._featureColumns,
                                                    hidden_units = [layerSize for layerSize in self._layerSizes[1:-1]],
                                                    optimizer = self._optimizer,
                                                    n_classes = 10,
                                                    dropout = 0.1,
                                                    model_dir = './tmp/mnist_model' )

        self._trainInputFn = tf.estimator.inputs.numpy_input_fn( x = {"x": input(mnist.train)[0]},
                                                                 y = input(mnist.train)[1],
                                                                 num_epochs = None,
                                                                 batch_size = 50,
                                                                 shuffle = True )

        self._testInputFn = tf.estimator.inputs.numpy_input_fn( x = {"x": input(mnist.test)[0]},
                                                                y = input(mnist.test)[1],
                                                                num_epochs = 1,
                                                                shuffle = False )



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
        self._network.train( input_fn = self._trainInputFn, steps = 2000 )

    def evaluate(self):
        """ evaluate accuracy
        """
        accuracy_score = self._network.evaluate(input_fn=self._testInputFn)["accuracy"]
        print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    testNetwork = dnNetwork(layers = [784, 256, 10])
    testNetwork.train(None)
    testNetwork.evaluate()

