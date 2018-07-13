import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 
from tensorflow.contrib import rnn
from itertools import chain
def scaling_data(X):
    min = np.amin(X)
    max = np.amax(X)
    mean = np.mean(X)
    scale = (X-min)/(max-min)
    return scale, min, max
def prepare_data(link, sliding):
    colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
    df = read_csv(link, header=None, index_col=False, names=colnames, usecols=[3,4,9,10], engine='python')
    scaler = MinMaxScaler(feature_range=(0, 1))
    cpu = df['cpu_rate'].values.reshape(-1,1)
    mem = df['mem_usage'].values
    disk_io_time = df['disk_io_time'].values
    disk_space = df['disk_space'].values

    # cpu_nomal = scaler.fit_transform(cpu)
    cpu_nomal, minCPU, maxCPU = scaling_data(cpu)
    print minCPU, maxCPU
    length = len(cpu_nomal)

    data = [] 
    print length
    print sliding
    for i in range(length-sliding):
        datai=[]
        for j in range(sliding):
            datai.append(cpu_nomal[i+j])
        data.append(datai)
    data = np.array(data)
    y = cpu_nomal[sliding:]

    train_size = int(length*0.6)
    val_size = int(length*0.2)

    train_X_encoder = data[0:train_size]
    train_X_decoder = []
    train_X_decode = cpu_nomal[sliding-1:train_size-1]
    for i in range(len(train_X_decode)):
        aaa = []
        aaa.append(train_X_decode[i])
        train_X_decoder.append(aaa)
    train_y = y[0:train_size]
    val_X_encoder = data[train_size:train_size+val_size]
    val_X_decoder = cpu_nomal[train_size+sliding-1:train_size+val_size-1]
    val_Y = y[train_size:train_size+val_size]
    test_X_encoder = data[train_size+val_size:]
    test_X_decoder = cpu_nomal[train_size+val_size-1:length-1]
    test_y = y[train_size+val_size:]

    return train_X_encoder, np.array(train_X_decoder), train_y, val_X_encoder, val_X_decoder, val_Y, test_X_encoder, test_X_decoder, test_y

# print "test"
# a=[[1]]
# b = [[1]]
# a1 = tf.concat([a,b],0)
def init_encoder(num_units, num_layers,sliding):
    cell = tf.contrib.rnn.LSTMCell(num_units, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
    rnn_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple = True)
    # x = tf.placeholder("float",[None,sliding,1])
    # input_encoder=tf.unstack(x ,sliding,1)
    # outputs,_=rnn.static_rnn(rnn_cells,input_encoder, scope = "layer",dtype="float32")
    # return outputs
    # if num_layers == 1:
    #     lstm_layer=tf.nn.rnn_cell.LSTMCell(num_units,forget_bias=1)
    # else:
    #     lstm_layer = rnn.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units,forget_bias=1) for _ in range(num_layers)], state_is_tuple=True)
        # for i in range(num_layers):
        #     lstm_layer = tf.contrib.rnn.MultiRNNCell(tf.nn.rnn_cell.LSTMCell(num_units,forget_bias=1),state_is_tuple=True)
    return rnn_cells
def init_decoder(num_units,num_layers,sliding):
    cell = tf.contrib.rnn.LSTMCell(num_units, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
    rnn_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple = True)
    return rnn_cells
# def encode(lstm_layer,sliding):
#     x = tf.placeholder("float",[None,sliding,1])
#     input_encoder=tf.unstack(x ,sliding,1)
#     outputs,_=tf.nn.dynamic_rnn(lstm_layer,input_encoder, scope = "layer",dtype="float32")
#     return outputs

link = './data/google_trace_timeseries/data_resource_usage_10Minutes_6176858948.csv'
sliding = 2
sliding_decode = 1
num_units = 4
num_layers = 1
time_step = 1
n_output = 1
learning_rate = 0.01
training_epochs = 2
batch_size = 4
display_step = 1

train_X_encoder, train_X_decoder, train_y, val_X_encoder, val_X_decoder, val_Y, test_X_encoder, test_X_decoder, test_y = prepare_data(link,sliding)
print train_X_decoder
print train_y
# cell = tf.contrib.rnn.LSTMCell(num_units, state_is_tuple=True)
# cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
# rnn_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple = True)

rnn_cells_encoder = init_encoder(num_units, num_layers,sliding)
x1 = tf.placeholder("float",[None,sliding/time_step,time_step])
input_encoder=tf.unstack(x1 ,sliding/time_step,time_step)
outputs_encoder,state_encoder=rnn.static_rnn(rnn_cells_encoder,input_encoder, scope = "layer",dtype="float32")

# rnn_cells_decoder = init_decoder(num_units,num_layers,sliding_decode)
x2 = tf.placeholder("float",[None,sliding/time_step,time_step])
input_decoder=tf.unstack(x2 ,sliding/time_step,time_step)
# outputs_decoder,state_decoder=rnn.static_rnn(rnn_cells_decoder,input_decoder, scope = "layer",dtype="float32", initial_state=init_state)
outputs_decoder,state_decoder=rnn.static_rnn(rnn_cells_encoder,input_decoder, scope = "layer",dtype="float32", initial_state=state_encoder)

# # decoder
# cell_state = tf.placeholder(tf.float32, [batch_size, num_units])
# hidden_state = tf.placeholder(tf.float32, [batch_size, num_units])
# # init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
# init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, num_units])

# state_per_layer_list = tf.unstack(init_state, axis=0)
# init_state = tuple(
#     [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
#      for idx in range(num_layers)]
# )
# out_weights=tf.Variable(tf.random_normal([num_units,n_output]))
# out_bias=tf.Variable(tf.random_normal([n_output]))
# x1=tf.placeholder("float",[None,sliding,n_input])
# x2=tf.placeholder("float",[None,1,n_input])
# y=tf.placeholder("float",[None,n_output])

# lstm_layer=tf.nn.rnn_cell.LSTMCell(num_units,forget_bias=1)

# input_encoder=tf.unstack(x1 ,sliding,1)

# outputs_encoder,_=rnn.static_rnn(lstm_layer,input_encoder,dtype="float32")
# # input_decoder_raw = tf.concat([outputs_encoder[-1], x2],0)
# input_decoder = tf.unstack(outputs_encoder,num_units,1)
# # outputs_decoder,_=rnn.static_rnn(lstm_layer,input_decoder,dtype="float32")
# prediction=tf.matmul(outputs_encoder[-1],out_weights)+out_bias

#loss_function
# loss=tf.reduce_mean(tf.square(y-prediction))
# #optimization
# optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
avg_set = []
epoch_set=[]
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # print sess.run(a1)
    # encoder = sess.run(init_encoder(64,1,2))
    iter=1
    for epoch in range(training_epochs):
        # Train with each example
        total_batch = int(len(train_X_encoder)/batch_size)
        for i in range(total_batch):
            # sess.run(updates)
            avg_cost = 0
            for i in range(total_batch):
                batch_xs_encoder,batch_xs_decoder ,batch_ys = train_X_encoder[i*batch_size:(i+1)*batch_size], train_X_decoder[i*batch_size:(i+1)*batch_size],train_y[i*batch_size:(i+1)*batch_size]
                print "batch"
                print batch_xs_encoder
                print "batch"
                print batch_xs_decoder
                print batch_ys
                # sess.run(optimizer, feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder,y: batch_ys})
                # avg_cost += sess.run(loss,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder,y: batch_ys})/total_batch
                print "outputs"
                state = sess.run(state_encoder,feed_dict={x1: batch_xs_encoder})
                # print state
                j = 0
                c = []
                h = []
                # for i in chain.from_iterable(state):
                #     if(j == 0):
                #         c = i 
                #     else:
                #         h = i
                #     j+=1
                # print 'c'
                # print c 
                # print 'h' 
                # print h 
                print sess.run(outputs_decoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_encoder})
                # print sess.run(outputs_decoder,feed_dict={x: batch_xs_encoder,cell_state: c,hidden_state:h})
                # print "outputs"
                # print sess.run(outputs[-1])
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost)
            avg_set.append(avg_cost)
            epoch_set.append(epoch+1)
            
            print "Training phase finished"
    print "out_weights"
    print sess.run(out_weights)
    
    # out = rnn.static_rnn(lstm_layer,test_X,dtype="float32")
    out = sess.run(prediction, feed_dict={x1:test_X_encoder,x2:test_X_decoder})
    predict = out * (maxCPU - minCPU) + minCPU
    # predict = tf.matmul(outputs[-1],out_weights) + out_bias
    print predict
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