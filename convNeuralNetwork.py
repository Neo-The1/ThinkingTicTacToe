# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class cnNetwork():

    """ class representing a convolutional neural network with multiple hidden layers
    """
    def __init__(self, inputShape,labels,**kwds):
        self._optimizer = kwds.get('optimizer',tf.train.AdamOptimizer())
        self._inputShape = inputShape
        self._network = tf.estimator.Estimator(
        model_fn = self.cnn_model,
        model_dir = '/tmp/mnist_convnet_model'
        )
    def cnn_model(self,features,labels,mode):
        #x = tf.placeholder(tf.float32,shape=[28,28])
        input_layer = tf.reshape(features["x"],shape =[-1,self._inputShape[0],self._inputShape[1],1])
        #convolutional layer 1
        conv1 = tf.layers.conv2d(
                inputs=input_layer, 
                filters=32,
                kernel_size=[5,5],
                padding="same",
                activation=tf.nn.relu
                )
        #pooling layer 1
        pool1 = tf.layers.max_pooling2d(
                inputs=conv1,
                pool_size=[2,2],
                strides=2
                )
        
        #convolutional layer 2
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5,5],
                padding="same",
                activation=tf.nn.relu
                )
        
        #pooling layer 2
        pool2 = tf.layers.max_pooling2d(
                inputs=conv2,
                pool_size=2,
                strides=2
                )
        
        #dense layer 1
        pool2_flat = tf.reshape(pool2,[-1,7*7*64])
        dense = tf.layers.dense(
                inputs=pool2_flat,
                units = 1024,
                activation  = tf.nn.relu
                )
        #implement droput regularization to prevent overfitting
        dropout = tf.layers.dropout(
                inputs=dense,
                rate=0.4,
                training=mode == tf.estimator.ModeKeys.TRAIN
                )
        
        # dense layer 2 = logits , the final classification, here (0 to 9)
        logits = tf.layers.dense(
                inputs=dropout,
                units=10
                #activation is default = linear
                )
        predictions = {
            #Generate predictions for PREDICT and EVAL mode
            "classes":tf.argmax(input=logits,axis=1),
            #Add "softmax_tensor" to graph. It is used for PREDICT and by
            #"logging_hook"
            "probabilities":tf.nn.softmax(logits=logits,name = "softmax_tensor")
            }
    
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,prediction=predictions)            
        
        #calculate loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
        
        #configure training for TRAIN mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step()
                    )
            return tf.estimator.EstimatorSpec(
                    mode=mode,loss=loss,train_op=train_op)
        
        #Add evaluation metric for EVAL mode
        eval_metric_ops = {
                "accuracy":tf.metrics.accuracy(
                        labels = labels,
                        predictions = predictions["classes"])
                }
        return tf.estimator.EstimatorSpec(
                mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)


            
    def loadFromFile(self, filename):
        """ Load the network parameters from a file
        """
        return None

    def saveToFile(self, fileName):
        """ Save the network parameters to a file
        """
        return None

    def train(self, trainData, trainLabels):
        """ Train the network using passed training data
        """
        trainInputFn = tf.estimator.inputs.numpy_input_fn( x = {"x":trainData},
                                                                 y = trainLabels,
                                                                 num_epochs = None,
                                                                 batch_size = 50,
                                                                 shuffle = True )
        self._network.train( input_fn = trainInputFn, steps = 2000 )

    def evaluate(self, evalData, evalLabels):
        """ evaluate accuracy
        """
        evalInputFn = tf.estimator.inputs.numpy_input_fn( x = {"x": evalData},
                                                                y = evalLabels,
                                                                num_epochs = 1,
                                                                shuffle = False )
        accuracy_score = self._network.evaluate(input_fn=evalInputFn)["accuracy"]
        print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
        
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    testNetwork = cnNetwork(inputShape=[28,28],labels=train_labels)
    testNetwork.train(train_data,train_labels)
    testNetwork.evaluate(eval_data,eval_labels)