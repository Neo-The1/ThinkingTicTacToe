#creating a test nn usng tensorflow
import tensorflow as tf
#importing tensorflow example for handwritten digit recognition
from tensorflow.examples.tutorials.mnist import input_data

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
# we will use only 1 layer (output layer = hidden layer)
W1 = tf.Variable(tf.random_normal([784,10],stddev = 0.03), name='W1')
b1 = tf.Variable(tf.random_normal([10],stddev = 0.03), name='b1')

#calculate the output
out = tf.add(tf.matmul(x,W1),b1)
out = tf.nn.relu(out)

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
with tf.Session() as sess:
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

