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
from model.utils.preprocessing_data_forBNN import TimeseriesBNN
class Model:
    def __init__(self, original_data = None, train_size = None, valid_size = None, 
    sliding_encoder = None, sliding_decoder = None, sliding_inference = None,
    batch_size = None, num_units = None, num_layers = None,
    activation_decoder = None, activation_inference = None, 
    # n_input = None, n_output = None,
    learning_rate = None, epochs = None, time_step = None, display_step = None ):
        self.original_data = original_data
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder
        self.sliding_inference = sliding_inference
        self.batch_size = batch_size
        self.num_units = num_units
        self.num_layers = num_layers
        self.activation_decoder = activation_decoder
        self.activation_inference = activation_inference
        # self.n_input = n_input
        # self.n_output = n_output
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.time_step = time_step
        self.display_step = display_step
    def preprocessing_data(self):
        print ('timeseries1')
        timeseries = TimeseriesBNN(self.original_data, self.train_size, self.valid_size, self.sliding_encoder, self.sliding_decoder, self.sliding_inference, self.time_step)
        print ('timeseries')
        self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.min_y, self.max_y, self.train_x_inference, self.valid_x_inference, self.test_x_inference, self.train_y_inference, self.valid_y_inference, self.test_y_inference = timeseries.prepare_data()
    def init_RNN(self, num_units, num_layers):
        cell = tf.contrib.rnn.LSTMCell(num_units,state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
        rnn_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple = True)
        return rnn_cells
    def fit(self):
        self.preprocessing_data()
        print ('self.train_x_encoder')
        print (self.train_x_encoder)
        self.train_x_encoder = np.array(self.train_x_encoder)
        self.train_x_decoder = np.array(self.train_x_decoder)
        self.test_x_encoder = np.array(self.test_x_encoder)
        self.test_x_decoder = np.array(self.test_x_decoder)
        self.test_y_decoder = np.array(self.test_y_decoder)
        self.train_x_inference = np.array(self.train_x_inference)
        print ('self.train_x_inference')
        print (self.train_x_inference)
        self.test_x_inference = np.array(self.test_x_inference)
        self.n_input_encoder = self.train_x_encoder.shape[1]
        self.n_input_decoder = self.train_x_decoder.shape[1]
        self.n_output = self.train_y_inference.shape[1]
        print ('self.test_y')
        print (self.train_x_encoder[0])
        print (self.train_x_decoder[0])
        # print (self.train_y[0])
        print (self.max_y)
        print (self.min_y)
        # print self.test_y
        encoder = self.init_RNN(self.num_units, self.num_layers)
        x1 = tf.placeholder("float",[None, self.sliding_encoder/self.time_step, self.time_step])
        # input_encoder=tf.unstack(x1 ,[None,self.sliding_encoder/self.time_step,self.time_step])
        outputs_encoder,state_encoder=tf.nn.dynamic_rnn(encoder, x1,dtype="float32")
        
        x2 = tf.placeholder("float",[None, self.sliding_decoder/self.time_step, self.time_step])
        y1 = tf.placeholder("float", [None, self.n_output])
        x3 = tf.placeholder("float",[None, int(self.sliding_inference/self.time_step), self.time_step])
        y2 = tf.placeholder("float", [None, self.n_output])
        # input_decoder=tf.unstack(x2 ,self.sliding_decoder/self.time_step,self.time_step)
        outputs_decoder,state_decoder=tf.nn.dynamic_rnn(encoder, x2,dtype="float32", initial_state=state_encoder)
        out_weights=tf.Variable(tf.random_normal([int(self.sliding_decoder/self.time_step), self.n_output]))
        out_bias=tf.Variable(tf.random_normal([self.n_output]))
        # prediction = outputs_decoder[:,:,-1]
        prediction=tf.nn.sigmoid(tf.matmul(outputs_decoder[:,:,-1],out_weights)+out_bias)
        prediction_inverse = prediction * (self.max_y - self.min_y) + self.min_y 
        y1_inverse = y1 * (self.max_y - self.min_y) + self.min_y 
        # loss_function
        loss_encoder_decoder = tf.reduce_mean(tf.square(y1-prediction))
        #optimization
        optimizer_encoder_decoder = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_encoder_decoder)
        # state_encoder = np.reshape(state_encoder[-1].h, [4])
        state_encoder = tf.reshape(state_encoder[-1].h,  [None, 1, self.num_units])
        input_inference = tf.concat([x3, state_encoder[-1].h],0)
        # hidden_value = tf.layers.dense(input_inference, 3, activation=tf.nn.sigmoid)
        # output_inference = tf.layers.dense(hidden_value,1, activation=tf.nn.sigmoid)
        # loss_inference = tf.reduce_mean(tf.square(y2-output_inference))
        
        # #optimization
        # optimizer_inference = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss_inference)
        # output_inference_inverse = output_inference * (self.max_y - self.min_y) + self.min_y 
        # y2_inverse = y2 * (self.max_y - self.min_y) + self.min_y
        # error = tf.reduce_sum(tf.abs(tf.subtract(output_inference_inverse,y2_inverse)))/len(self.test_y_inference)

        cost_train_set = []
        cost_valid_set = []
        epoch_set=[]
        init=tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            iter=1
            for epoch in range(self.epochs):
                # Train with each example
                total_batch = int(len(self.train_x_encoder)/self.batch_size)
                print (total_batch)
                # sess.run(updates)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs_encoder,batch_xs_decoder ,batch_ys = self.train_x_encoder[i*self.batch_size:(i+1)*self.batch_size], self.train_x_decoder[i*self.batch_size:(i+1)*self.batch_size],self.train_y_inference[i*self.batch_size:(i+1)*self.batch_size]
                    sess.run(optimizer_encoder_decoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder, y1:batch_ys})
                    # print('state_encoder')
                    # print (sess.run(state_encoder,feed_dict={x1: batch_xs_encoder}))
                    avg_cost += sess.run(loss_encoder_decoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder,y1: batch_ys})/total_batch
                    if(i == total_batch -1):
                        a = sess.run(state_encoder,feed_dict={x1: batch_xs_encoder})
                        print ('......')
                        print(a)
                        print(a[0])
                        # print (a.c)
                    # print sess.run(outputs_decoder,feed_dict={x: batch_xs_encoder,cell_state: c,hidden_state:h})
                    # print "outputs"
                    # print sess.run(outputs[-1])
                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print ("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost))
                    cost_train_set.append(avg_cost)
                    epoch_set.append(epoch+1)
                    val_cost = sess.run(loss_encoder_decoder, feed_dict={x1:self.valid_x_encoder,x2:self.valid_x_decoder, y1: self.valid_y_decoder})
                    # print sess.run(state_encoder,feed_dict={x1: batch_xs_encoder})
                    print (val_cost)
                    cost_valid_set.append(val_cost)
                    print ("Training phase finished")
                    # print sess.run(error,feed_dict={x1: self.val_x_encoder,x2: self.val_x_decoder,y: self.val_y})
            print ('sess.run')    
            print (type(state_encoder))        
            print (sess.run(state_encoder,feed_dict={x1: batch_xs_encoder}))
            print (sess.run(state_encoder[-1].h,feed_dict={x1: batch_xs_encoder}))
            print ('training encoder-decoder ok!!!')

            # for epoch in range(self.epochs):
            #     total_batch = int(len(self.train_x_inference)/self.batch_size)
            #     print (total_batch)
            #     # sess.run(updates)
            #     avg_cost = 0
            #     for i in range(total_batch):
            #         batch_xs_encoder,batch_xs_inference ,batch_ys = self.train_x_encoder[i*self.batch_size:(i+1)*self.batch_size], self.train_x_inference[i*self.batch_size:(i+1)*self.batch_size],self.train_y_inference[i*self.batch_size:(i+1)*self.batch_size]
            #         print ('input_inference')
            #         s_e = sess.run(state_encoder,feed_dict={x1: batch_xs_encoder})
            #         # print (s_e[0]['h'])
            #         # lstm_c, lstm_h = s_e
            #         # print ('....')
            #         # print(lstm_c)
            #         # print(lstm_h)
            #         # for se in s_e:
            #         #     print ('...')
            #         #     print(se)
            #         # print (s_e[1])
            #         print (batch_xs_inference)
            #         print (sess.run(input_inference,feed_dict={x1: batch_xs_encoder,x3: batch_xs_decoder, y2:batch_ys}))
            #         sess.run(optimizer_inference,feed_dict={x1: batch_xs_encoder,x3: batch_xs_decoder, y2:batch_ys})
            #         avg_cost += sess.run(loss_inference,feed_dict={x1: batch_xs_encoder,x3: batch_xs_decoder,y2: batch_ys})/total_batch
                    # if(i == total_batch -1):
                        # print sess.run(state_encoder,feed_dict={x1: batch_xs_encoder})

            # print "out_weights"
            # print sess.run(out_weights)
            # outputs_decoder = sess.run(outputs_decoder, feed_dict={x1:self.test_x_encoder,x2:self.test_x_decoder, y: self.test_y})
            # print outputs_decoder
            # print outputs_decoder[:,:,-1]
            prediction_inverse = sess.run(output_inference_inverse, feed_dict={x1:self.test_x_encoder,x3:self.test_x_inference, y2: self.test_y_inference})
            error = sess.run(error, feed_dict={x1:self.test_x_encoder, x3:self.test_x_inference, y2: self.test_y_inference})
            print (error)
            
            
            
            
            
            plt.plot(cost_train_set)
            plt.plot(cost_valid_set)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
            plt.savefig('/home/nguyenthang/Hust-Learn/lab/machinelearninghandling/history/history_mem.png')
					
            # print outputs_decoder 
            print (len(outputs_decoder),len(outputs_decoder[0]),len(outputs_decoder[0][0]))
            # print prediction_inverse
            print (sess.run(state_encoder, feed_dict={x1:self.train_x_encoder}))
            # return state_decoder
            predictionDf = pd.DataFrame(np.array(prediction_inverse))
            predictionDf.to_csv('/home/nguyenthang/Hust-Learn/lab/machinelearninghandling/results/result_mem.csv', index=False, header=None)
            # errorDf = pd.DataFrame(np.array(error))
            # errorDf.to_csv('/home/nguyenthang/Hust-Learn/lab/machinelearninghandling/results/error.csv', index=False, header=None)
