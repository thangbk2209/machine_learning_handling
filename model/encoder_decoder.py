import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 
from tensorflow.contrib import rnn
# from utils.preprocessing_data import Timeseries
from utils.preprocessing_data_forBNN import TimeseriesBNN
class Model:
    def __init__(self, original_data = None, train_size = None, valid_size = None, 
    sliding_encoder = None, sliding_decoder = None, batch_size = None,
    num_units = None, activation = None, num_layers = None, 
    # n_input = None, n_output = None,
    learning_rate = None, epochs = None, time_step = None,
    display_step = None ):
        self.original_data = original_data
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder
        self.batch_size = batch_size
        self.num_units = num_units
        self.num_layers = num_layers
        # self.n_input = n_input
        # self.n_output = n_output
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.time_step = time_step
        self.display_step = display_step
    def preprocessing_data(self):
        print 'timeseries1'
        timeseries = TimeseriesBNN(self.original_data, self.train_size, self.valid_size, self.sliding_encoder, self.sliding_decoder)
        print 'timeseries'
        self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y, self.valid_y, self.test_y, self.min_y, self.max_y = timeseries.prepare_data()
    def init_RNN(self, num_units, num_layers):
        cell = tf.contrib.rnn.LSTMCell(num_units, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
        rnn_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple = True)
        return rnn_cells
    def fit(self):
        self.preprocessing_data()
        self.train_x_encoder = np.array(self.train_x_encoder)
        self.train_x_decoder = np.array(self.train_x_decoder)
        self.test_x_encoder = np.array(self.test_x_encoder)
        self.test_x_decoder = np.array(self.test_x_decoder)
        self.test_y = np.array(self.test_y)
        self.n_input_encoder = self.train_x_encoder.shape[1]
        self.n_input_decoder = self.train_x_decoder.shape[1]
        self.n_output = self.train_y.shape[1]
        print 'self.test_y'
        print self.train_x_encoder[0]
        print self.train_x_decoder[0]
        print self.train_y[0]
        print self.max_y
        print self.min_y
        # print self.test_y
        encoder = self.init_RNN(self.num_units, self.num_layers)
        x1 = tf.placeholder("float",[None,self.sliding_encoder/self.time_step,self.time_step])
        input_encoder=tf.unstack(x1 ,self.sliding_encoder/self.time_step,self.time_step)
        outputs_encoder,state_encoder=rnn.static_rnn(encoder, input_encoder, scope = "layer",dtype="float32")
        
        x2 = tf.placeholder("float",[None,self.sliding_decoder/self.time_step,self.time_step])
        y = tf.placeholder("float", [None, self.n_output])
        input_decoder=tf.unstack(x2 ,self.sliding_decoder/self.time_step,self.time_step)
        outputs_decoder,state_decoder=rnn.static_rnn(encoder, input_decoder, scope = "layer",dtype="float32", initial_state=state_encoder)

        out_weights=tf.Variable(tf.random_normal([self.num_units,self.n_output]))
        out_bias=tf.Variable(tf.random_normal([self.n_output]))

        prediction=tf.nn.sigmoid(tf.matmul(outputs_decoder[-1],out_weights)+out_bias)
        prediction_inverse = prediction * (self.max_y - self.min_y) + self.min_y 
        y_inverse = y * (self.max_y - self.min_y) + self.min_y 
        error = tf.reduce_sum(tf.abs(tf.subtract(prediction_inverse,y_inverse)))/len(self.test_y)
        # loss_function
        loss=tf.reduce_mean(tf.square(y-prediction))
        #optimization
        optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        
        avg_set = []
        epoch_set=[]
        init=tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            iter=1
            for epoch in range(self.epochs):
                # Train with each example
                total_batch = int(len(self.train_x_encoder)/self.batch_size)
                print total_batch
                # sess.run(updates)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs_encoder,batch_xs_decoder ,batch_ys = self.train_x_encoder[i*self.batch_size:(i+1)*self.batch_size], self.train_x_decoder[i*self.batch_size:(i+1)*self.batch_size],self.train_y[i*self.batch_size:(i+1)*self.batch_size]
                    sess.run(optimizer,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder, y:batch_ys})
                    avg_cost += sess.run(loss,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder,y: batch_ys})/total_batch
                    if(i == total_batch -1):
                        print sess.run(state_encoder,feed_dict={x1: batch_xs_encoder})
                    # print sess.run(outputs_decoder,feed_dict={x: batch_xs_encoder,cell_state: c,hidden_state:h})
                    # print "outputs"
                    # print sess.run(outputs[-1])
                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print "Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost)
                    avg_set.append(avg_cost)
                    epoch_set.append(epoch+1)
                    # print sess.run(state_encoder,feed_dict={x1: batch_xs_encoder})
                    print "Training phase finished"
                    # print sess.run(error,feed_dict={x1: self.val_x_encoder,x2: self.val_x_decoder,y: self.val_y})
            print sess.run(state_encoder,feed_dict={x1: batch_xs_encoder})
            print "out_weights"
            print sess.run(out_weights)
            
            # out = rnn.static_rnn(lstm_layer,test_X,dtype="float32")
            prediction_inverse = sess.run([prediction_inverse], feed_dict={x1:self.test_x_encoder,x2:self.test_x_decoder, y: self.test_y})
            error = sess.run(error, feed_dict={x1:self.test_x_encoder,x2:self.test_x_decoder, y: self.test_y})
            print error

