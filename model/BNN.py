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

"""This class build the model BNN with initial function and train function"""
class Model:
    def __init__(self, original_data = None, train_size = None, valid_size = None, 
    sliding_encoder = None, sliding_decoder = None, sliding_inference = None,
    batch_size = None, num_units_LSTM = None, num_layers = None,
    activation_decoder = None, activation_inference = None, 
    # n_input = None, n_output = None,
    learning_rate = None, epochs_encoder_decoder = None, epochs_inference = None , input_dim = None, 
    num_units_inference = None ):
        self.original_data = original_data
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder
        self.sliding_inference = sliding_inference
        self.batch_size = batch_size
        self.num_units_LSTM = num_units_LSTM
        self.num_layers = num_layers
        self.activation_decoder = activation_decoder
        self.activation_inference = activation_inference
        # self.n_input = n_input
        # self.n_output = n_output
        self.learning_rate = learning_rate
        self.epochs_encoder_decoder = epochs_encoder_decoder
        self.epochs_inference = epochs_inference
        self.input_dim = input_dim

        self.num_units_inference = num_units_inference


    def preprocessing_data(self):
        print ('timeseries1')
        timeseries = TimeseriesBNN(self.original_data, self.train_size, self.valid_size, self.sliding_encoder, self.sliding_decoder, self.sliding_inference, self.input_dim)
        print ('timeseries')
        self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.min_y, self.max_y, self.train_x_inference, self.valid_x_inference, self.test_x_inference, self.train_y_inference, self.valid_y_inference, self.test_y_inference = timeseries.prepare_data()
    def init_RNN(self, num_units, num_layers):
        cell = tf.contrib.rnn.LSTMCell(num_units,state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
        rnn_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple = True)
        return rnn_cells
    def early_stopping(self, array, patience):
        value = array[len(array) - patience - 1]
        arr = array[len(array)-patience:]
        check = 0
        for val in arr:
            if(val > value):
                check += 1
        if(check == patience):
            return False
        else:
            return True
    def fit(self):
        self.preprocessing_data()
        # print ('self.train_x_encoder')
        # print (self.train_x_encoder)
        self.train_x_encoder = np.array(self.train_x_encoder)
        self.train_x_decoder = np.array(self.train_x_decoder)
        self.test_x_encoder = np.array(self.test_x_encoder)
        self.test_x_decoder = np.array(self.test_x_decoder)
        self.test_y_decoder = np.array(self.test_y_decoder)
        self.train_x_inference = np.array(self.train_x_inference)
        # print ('self.train_x_inference')
        # print (self.train_x_inference)
        self.test_x_inference = np.array(self.test_x_inference)
        self.n_input_encoder = self.train_x_encoder.shape[1]
        self.n_input_decoder = self.train_x_decoder.shape[1]
        self.n_output_inference = self.train_y_inference.shape[1]
        self.n_output_encoder_decoder = self.train_y_decoder.shape[1]
        # print ('self.test_y')
        # print (self.train_x_encoder[0])
        # print (self.train_x_decoder[0])
        # print (self.train_y[0])
        # print (self.max_y)
        # print (self.min_y)
        # print self.test_y
        tf.reset_default_graph()
        encoder = self.init_RNN(self.num_units_LSTM, self.num_layers)
        x1 = tf.placeholder("float",[None, self.sliding_encoder/self.input_dim, self.input_dim])
        # input_encoder=tf.unstack(x1 ,[None,self.sliding_encoder/self.time_step,self.time_step])
        outputs_encoder,state_encoder=tf.nn.dynamic_rnn(encoder, x1, dtype="float32")
        
        x2 = tf.placeholder("float",[None, self.sliding_decoder/self.input_dim, self.input_dim])
        y1 = tf.placeholder("float", [None, self.n_output_encoder_decoder])
        x3 = tf.placeholder("float",[None, 1, int(self.sliding_inference)])
        y2 = tf.placeholder("float", [None, self.n_output_inference])
        # input_decoder=tf.unstack(x2 ,self.sliding_decoder/self.time_step,self.time_step)
        outputs_decoder,state_decoder=tf.nn.dynamic_rnn(encoder, x2,dtype="float32", initial_state=state_encoder)
        out_weights=tf.Variable(tf.random_normal([int(self.sliding_decoder/self.input_dim), self.n_output_encoder_decoder]))
        out_bias=tf.Variable(tf.random_normal([self.n_output_encoder_decoder]))
        # prediction = outputs_decoder[:,:,-1]
        prediction=tf.nn.sigmoid(tf.matmul(outputs_decoder[:,:,-1],out_weights)+out_bias)
        prediction_inverse = prediction * (self.max_y - self.min_y) + self.min_y 
        y1_inverse = y1 * (self.max_y - self.min_y) + self.min_y 
        # loss_function
        loss_encoder_decoder = tf.reduce_mean(tf.square(y1-prediction))
        #optimization
        optimizer_encoder_decoder = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_encoder_decoder)
        # state = tf.tile(state_encoder[-1].h)
        # state = tf.shape(x3)[0]
        state = tf.reshape(state_encoder[-1].h, [tf.shape(x3)[0], 1, self.num_units_LSTM])
        input_inference = tf.concat([x3,state],2)
        input_inference = tf.reshape(input_inference,[tf.shape(x3)[0], self.sliding_inference + self.num_units_LSTM])
        # state_encoder = np.reshape(state_encoder[-1].h, [4])
        # state_encoder = tf.reshape(state_encoder[-1].h,  [None, 1, self.num_units])
        # input_inference = tf.concat([x3, state_encoder[-1].h],0)
        if(self.activation_inference == 1):
            activation = tf.nn.sigmoid
        elif(self.activation_inference == 2):
            activation = tf.nn.relu 
        hidden_value = tf.layers.dense(input_inference, self.num_units_inference, activation=activation)
        output_inference = tf.layers.dense(hidden_value,self.n_output_inference, activation=activation)
        # loss
        loss_inference = tf.reduce_mean(tf.square(y2-output_inference))
        #optimization
        optimizer_inference = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss_inference)
        output_inference_inverse = output_inference * (self.max_y - self.min_y) + self.min_y 
        y2_inverse = y2 * (self.max_y - self.min_y) + self.min_y
        error = tf.reduce_mean(tf.abs(tf.subtract(output_inference_inverse,y2_inverse)) )

        cost_train_encoder_decoder_set = []
        cost_valid_encoder_decoder_set = []
        cost_train_inference_set = []
        cost_valid_inference_set = []
        # epoch_set=[]
        init=tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            # training encoder_decoder
            print ("start training encoder_decoder")
            for epoch in range(self.epochs_encoder_decoder):
                # Train with each example
                print ('epoch encoder_decoder: ', epoch)
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
                # Display logs per epoch step
                print ("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost))
                cost_train_encoder_decoder_set.append(avg_cost)
                val_cost = sess.run(loss_encoder_decoder, feed_dict={x1:self.valid_x_encoder,x2:self.valid_x_decoder, y1: self.valid_y_decoder})
                cost_valid_encoder_decoder_set.append(val_cost)
                if (epoch>5):
                    if (self.early_stopping(cost_valid_encoder_decoder_set, 5) == False):
                        print ("early stopping encoder-decoder training")
                        break
                print ("Epoch encoder-decoder finished")
            print ('training encoder-decoder ok!!!')
            # training inferences
            print ('start training inference')
            for epoch in range(self.epochs_inference):
                print ("epoch inference: ", epoch)
                total_batch = int(len(self.train_x_inference)/self.batch_size)
                print (total_batch)
                # sess.run(updates)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs_encoder,batch_xs_inference ,batch_ys = self.train_x_encoder[i*self.batch_size:(i+1)*self.batch_size], self.train_x_inference[i*self.batch_size:(i+1)*self.batch_size],self.train_y_inference[i*self.batch_size:(i+1)*self.batch_size]
                    # print ('input_inference')
                    s_e = sess.run(state_encoder,feed_dict={x1: batch_xs_encoder})
                    sess.run(optimizer_inference,feed_dict={x1: batch_xs_encoder,x3: batch_xs_inference, y2:batch_ys})
                    avg_cost += sess.run(loss_inference,feed_dict={x1: batch_xs_encoder,x3: batch_xs_inference,y2: batch_ys})/total_batch
                    if(i == total_batch -1):
                        print (sess.run(state_encoder,feed_dict={x1: batch_xs_encoder}))
                print ("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost))
                cost_train_inference_set.append(avg_cost)
                # epoch_set.append(epoch+1)
                val_cost = sess.run(loss_inference, feed_dict={x1:self.valid_x_encoder,x3:self.valid_x_inference, y2: self.valid_y_inference})
                cost_valid_inference_set.append(val_cost)
                if (epoch>5):
                    if (self.early_stopping(cost_valid_inference_set,5) == False):
                        print ("early stopping encoder-decoder training")
                        break
            output_inference_inverse = sess.run(output_inference_inverse, feed_dict={x1:self.test_x_encoder,x3:self.test_x_inference, y2: self.test_y_inference})
            output_inference = sess.run(output_inference, feed_dict={x1:self.test_x_encoder,x3:self.test_x_inference, y2: self.test_y_inference})
            # print (output_inference)
            error = sess.run(error, feed_dict={x1:self.test_x_encoder,x3:self.test_x_inference, y2: self.test_y_inference})
            print (error)
            # print (output_inference_inverse)
            # print (self.test_y_inference)
            # print (len(output_inference_inverse))
            # print (len(self.test_y_inference))
            
        
            
            plt.plot(cost_train_inference_set)
            plt.plot(cost_valid_inference_set)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            # plt.show()
            plt.savefig('/home/thangnguyen/hust/lab/machine_learning_handling/history/history_mem.png')
					
            predictionDf = pd.DataFrame(np.array(output_inference_inverse))
            predictionDf.to_csv('/home/thangnguyen/hust/lab/machine_learning_handling/results/result_mem.csv', index=False, header=None)
            # errorDf = pd.DataFrame(np.array(error))
            # errorDf.to_csv('/home/thangnguyen/hust/lab/machine_learning_handling/results/error.csv', index=False, header=None)
            sess.close()