import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 
from tensorflow.contrib import rnn
def scaling_data(X):
    min = np.amin(X)
    max = np.amax(X)
    mean = np.mean(X)
    scale = (X-min)/(max-min)
    return scale, min, max
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv('./data/google_trace_timeseries/data_resource_usage_10Minutes_6176858948.csv', header=None, index_col=False, names=colnames, usecols=[3,4,9,10], engine='python')
scaler = MinMaxScaler(feature_range=(0, 1))
cpu = df['cpu_rate'].values.reshape(-1,1)
mem = df['mem_usage'].values
disk_io_time = df['disk_io_time'].values
disk_space = df['disk_space'].values

# cpu_nomal = scaler.fit_transform(cpu)
cpu_nomal, minCPU,maxCPU = scaling_data(cpu)
length = len(cpu_nomal)
slidings = [2]
data = [] 
print length
for sliding in slidings:
    print sliding
    for i in range(length-sliding):
        datai=[]
        for j in range(sliding):
            datai.append(cpu_nomal[i+j])
        data.append(datai)
    data = np.array(data)
y = cpu_nomal[sliding:]

train_size = int(length*0.8)
# val_size = int(length*0.2)
val_size = 0

train_X = data[0:train_size]
train_y = y[0:train_size]
val_X = data[train_size:train_size+val_size]
val_Y = y[train_size:train_size+val_size]
test_X = data[train_size+val_size:]
test_y = y[train_size+val_size:]
# print train_X
print len(test_X)
print "len test_X"
sliding = 2
num_units = 2
n_input = 1
n_output = 1
learning_rate = 0.001
training_epochs = 200
batch_size = 64
display_step = 1

out_weights=tf.Variable(tf.random_normal([num_units,n_output]))
out_bias=tf.Variable(tf.random_normal([n_output]))
x=tf.placeholder("float",[None,sliding,n_input])
y=tf.placeholder("float",[None,n_output])
lstm_layer=tf.nn.rnn_cell.LSTMCell(num_units,forget_bias=1)

input=tf.unstack(x ,sliding,1)

outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")
prediction=tf.nn.sigmoid(tf.add(tf.matmul(outputs[-1],out_weights),out_bias))

#loss_function
loss=tf.reduce_mean(tf.square(y-prediction))
#optimization
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
avg_set = []
epoch_set=[]
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter=1
    for epoch in range(training_epochs):
        # Train with each example
        total_batch = int(len(train_X)/batch_size)
        for i in range(total_batch):
            # sess.run(updates)
            avg_cost = 0
            for i in range(total_batch):
                batch_xs, batch_ys = train_X[i*batch_size:(i+1)*batch_size], train_y[i*batch_size:(i+1)*batch_size]
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += sess.run(loss,feed_dict={x: batch_xs,y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost)
            avg_set.append(avg_cost)
            epoch_set.append(epoch+1)
            print "Training phase finished"
    print avg_set
    print sess.run(out_weights)
    # print sess.run(tf.size(outputs),feed_dict={x:test_X})
    # print sess.run(tf.size(outputs[-1]),feed_dict={x:test_X})
    # out = rnn.static_rnn(lstm_layer,test_X,dtype="float32")
    out = sess.run(prediction, feed_dict={x:test_X})
    predict = out * (maxCPU - minCPU) + minCPU
    # predict = tf.matmul(outputs[-1],out_weights) + out_bias
    # print predict
    error = tf.reduce_sum(tf.abs(tf.subtract(predict,test_y)))/len(test_y)
    print sess.run(error)
predictionDf = pd.DataFrame(np.array(predict))
predictionDf.to_csv('result.csv', index=False, header=None)
# def RNN(x, weight, bias):
#     """
#     Forward-propagation.
#     IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
#     """
#     # inputs = tf.placeholder(tf.float32, [None, config.sliding, config.input_size])
#     # targets = tf.placeholder(tf.float32, [None, config.output_size])
#     def _create_one_cell():
#         return tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
#         if config.keep_prob < 1.0:
#             return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
#     cell = tf.contrib.rnn.MultiRNNCell(
#         [_create_one_cell() for _ in range(num_layers)], 
#         state_is_tuple=True
#     ) if config.num_layers > 1 else _create_one_cell()
#     weight = tf.Variable(tf.random_normal([lstm_size, N_OUTPUTS]))
#     bias = tf.Variable(tf.random_normal([N_OUTPUTS]))
#     predictions = tf.matmul(outputs, weight) + bias
#     return tf.matmul(outputs[-1], weights['out']) + biases['out']