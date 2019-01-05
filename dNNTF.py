import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
tf.reset_default_graph()

class dNNTF():
    """class for deep Neural Network using Tensorflow
    """
    def __init__(self,board1DSize,*args,**kwds):
        self._saveFile = kwds.get('saveFile')
        self._inputSize = 2*board1DSize*board1DSize+1
        self._outPiSize= board1DSize*board1DSize
        self._parameters = self.initializeParameters()
    def createPlaceholders(self):
        x = tf.placeholder(tf.float32,shape=(self._inputSize,None),name="x")
        yPi = tf.placeholder(tf.float32,shape=(self._outPiSize,None),name="yPi")
        yZ = tf.placeholder(tf.float32,shape=(1,None),name="yZ")
        return x,yPi,yZ
    
    def initializeParameters(self):
        """Initializes parameters to build a neural network with tensorflow. The shapes are:    
           Returns:
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
        """
        w1 = tf.get_variable("w1",[32,self._inputSize],initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1",[32,1],initializer=tf.zeros_initializer())
        w2 = tf.get_variable("w2",[16,32],initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2",[16,1],initializer=tf.zeros_initializer())
        wPi = tf.get_variable("wPi",[self._outPiSize,16],initializer=tf.contrib.layers.xavier_initializer())
        bPi = tf.get_variable("bPi",[self._outPiSize,1],initializer=tf.zeros_initializer())
        wZ = tf.get_variable("wZ",[1,16],initializer=tf.contrib.layers.xavier_initializer())
        bZ = tf.get_variable("bZ",[1,1],initializer=tf.zeros_initializer())
        parameters = {"w1":w1,"b1":b1,"w2":w2,"b2":b2,"wPi":wPi,"bPi":bPi,"wZ":wZ,"bZ":bZ}
        return parameters
    
    def fwdProp(self,x,parameters):
        """Performs fwd prop LINEAR->RELU->LINEAR->RELU
        Then for Pi layer LINEAR-> SOFTMAX
        For Z layer LINEAR->TANH
        """
        # Retrieve the parameters from the dictionary "parameters" 
        x = tf.cast(x,tf.float32)
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']
        wPi = parameters['wPi']
        bPi = parameters['bPi']
        wZ = parameters['wZ']
        bZ = parameters['bZ']
        z1 = tf.add(tf.matmul(w1,x),b1)
        a1 = tf.nn.relu(z1)
        z2 = tf.add(tf.matmul(w2,a1),b2)
        a2 = tf.nn.relu(z2)
        zPi = tf.add(tf.matmul(wPi,a2),bPi)
        aPi = tf.nn.softmax(zPi,axis=0)
        zZ = tf.add(tf.matmul(wZ,a2),bZ)
        aZ = tf.nn.tanh(zZ)
        
        return aPi,aZ
    
    def computeCost(self,aPi,aZ,yPi,yZ,parameters):
        eps = 1e-12 #to prevent log(0)
        w1 = parameters['w1']
        w2 = parameters['w2']
        wPi = parameters['wPi']
        wZ = parameters['wZ']
        beta = 0.0001
        regularizer = tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)+tf.nn.l2_loss(wPi)+tf.nn.l2_loss(wZ)
#        cost = tf.reduce_mean( 0.01*tf.square(aZ-yZ) - tf.matmul(tf.transpose(yPi),tf.log(aPi+eps)) 
#                + beta*regularizer)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(aPi),labels=tf.transpose(yPi)))
        return cost
    
    def train(self,trainX,trainYPi,trainYZ, lr = 0.0001, numEpochs = 100000, printCost=True):
        if numEpochs<200:
            lr = 0.1
        elif numEpochs<400:
            lr = 0.01
        elif numEpochs<600:
            lr = 0.001
        elif numEpochs<700:
            lr = 0.0001
        else:
            lr = 0.00001
                
        ops.reset_default_graph()
        costs = []
        (_,m) = trainX.shape
        x,yPi,yZ = self.createPlaceholders()
        parameters = self.initializeParameters()
        aPi,aZ = self.fwdProp(x,parameters)
        cost = self.computeCost(aPi,aZ,yPi,yZ,parameters)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9).minimize(cost)
        
        #initialize all variables
        init = tf.global_variables_initializer()
        
        #start session to compute tensorflow graph
        with tf.Session() as sess:
            #Run initialization
            sess.run(init)
            
            #Do training loop
            for epoch in range(numEpochs):
                epochCost = 0. #Defines cost related to an epoch
                _, c = sess.run([optimizer,cost],feed_dict={x:trainX,yPi:trainYPi,yZ:trainYZ})
                epochCost += c
                if printCost and epoch % 100 == 0:
                    print("Cost after %i:%f" % (epoch,epochCost))
                if printCost and epoch % 5 == 0:
                    costs.append(epochCost)
            
            self._parameters = sess.run(parameters)
            
    
    def predict(self,x):
        t = self.fwdProp(x,self._parameters)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            p,v=sess.run(t)
        return p,v
    
    def saveModel(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
         sess.run(init)
         saver.save(sess,'/tmp/model.ckpt')
        
    def loadModel(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
         sess.run(init)
         saver.restore(sess,'/tmp/model.ckpt')

if __name__ == "__main__":
    from tttBoard import tttBoard
    board = tttBoard(3)
    states = np.random.randn(19,1)
#    states[:,0] = board.decodeState(board.getState())
    testNet = dNNTF(3)
    print(states)
    result = testNet.predict(states)
    testNet.saveWeights()
    print(result)
#    print("res0  ",result[0][:,0])
#    print("res1  ",result[1][:,0][0])
                    
                