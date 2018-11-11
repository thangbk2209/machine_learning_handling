import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd 
from tensorflow.contrib import rnn
# from utils.preprocessing_data import Timeseries
from model.utils.preprocessing_data_forBNN import LSTM
import time
"""This class build the model BNN with initial function and train function"""
class Model:
    def __init__(self, original_data = None, prediction_data = None, 
    train_size = None, valid_size = None, sliding = None,
    batch_size = None, num_units_LSTM = None, 
    activation = None, optimizer = None,
    learning_rate = None, epochs = None,  
    input_dim = None, patience = None, dropout_rate = 0.8):
        self.original_data = original_data
        self.prediction_data = prediction_data
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding = sliding
        self.batch_size = batch_size
        self.num_units_LSTM = num_units_LSTM
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_dim = input_dim
        self.patience = patience
        self.dropout_rate = dropout_rate
    def preprocessing_data(self):
        timeseries = LSTM(self.original_data, self.prediction_data, self.train_size, self.valid_size, self.sliding, self.input_dim)
        self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y, self.min_y, self.max_y = timeseries.prepare_data()
    def init_RNN(self, num_units, activation):
        num_layers = len(num_units)
        hidden_layers = []
        for i in range(num_layers):
            if(i==0):
                cell = tf.contrib.rnn.LSTMCell(num_units[i],activation = activation)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                        input_keep_prob = 1.0, 
                                        output_keep_prob = self.dropout_rate,
                                        state_keep_prob = self.dropout_rate,
                                        variational_recurrent = True,
                                        input_size = self.input_dim,
                                        dtype=tf.float32)
                hidden_layers.append(cell)
            else:
                cell = tf.contrib.rnn.LSTMCell(num_units[i],activation = activation)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                        input_keep_prob = self.dropout_rate, 
                                        output_keep_prob = self.dropout_rate,
                                        state_keep_prob = self.dropout_rate,
                                        variational_recurrent = True,
                                        input_size = self.num_units_LSTM[i-1],
                                        dtype=tf.float32)
                hidden_layers.append(cell)
        rnn_cells = tf.contrib.rnn.MultiRNNCell(hidden_layers, state_is_tuple = True)
        return rnn_cells
    def mlp(self, input, num_units, activation):
        num_layers = len(num_units)
        prev_layer = input
        for i in range(num_layers):
            prev_layer = tf.layers.dense(prev_layer,
                                         num_units[i],
                                         activation = activation,
                                         name = 'layer'+str(i))
            drop_rate = 1-self.dropout_rate
            prev_layer = tf.layers.dropout(prev_layer , rate = drop_rate)
        
        prediction = tf.layers.dense(inputs=prev_layer,
                               units=1, 
                               activation = activation,
                               name = 'output_layer')
        return prediction
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
        print (self.max_y)
        print (self.min_y)
        # lol
        if(self.activation == 1):
            activation = tf.nn.sigmoid
        elif(self.activation == 2):
            activation= tf.nn.relu 
        elif(self.activation== 3):
            activation = tf.nn.tanh
        elif(self.activation == 4):
            activation = tf.nn.elu

        if(self.optimizer == 1):
            optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.9)
        elif(self.optimizer == 2):
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        else:
            optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
        
        tf.reset_default_graph()
        x = tf.placeholder("float",[None, self.sliding*len(self.original_data)/self.input_dim, self.input_dim])
        y = tf.placeholder("float", [None, self.train_y.shape[1]])
        with tf.variable_scope('LSTM'):
            
            lstm_layer = self.init_RNN(self.num_units_LSTM,activation)
            outputs, new_state = tf.nn.dynamic_rnn(lstm_layer, x, dtype="float32")
            outputs = tf.identity(outputs, name='outputs')

        prediction = tf.layers.dense(outputs[:,:,-1],self.train_y.shape[1],activation = activation,use_bias = True)
        loss = tf.reduce_mean(tf.square(y-prediction))
        optimize = optimizer.minimize(loss)

        cost_train_set = []
        cost_valid_set = []
        epoch_set=[]
        init=tf.global_variables_initializer()
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(init)
            # training encoder_decoder
            print ("start training ")
            for epoch in range(self.epochs):
                start_time = time.time()
                # Train with each example
                print ('epoch : ', epoch+1)
                total_batch = int(len(self.train_x)/self.batch_size)
                # print (total_batch)
                # sess.run(updates)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs = self.train_x[i*self.batch_size:(i+1)*self.batch_size]
                    batch_ys = self.train_y[i*self.batch_size:(i+1)*self.batch_size]
                    # print (sess.run(outputs_encoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder, y1:batch_ys}))
                    # print (sess.run(new_state_encoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder, y1:batch_ys}))
                    sess.run(optimize,feed_dict={x: batch_xs, y:batch_ys})
                    avg_cost += sess.run(loss,feed_dict={x: batch_xs, y:batch_ys})/total_batch
                # Display logs per epoch step
                print ("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost))
                cost_train_set.append(avg_cost)
                val_cost = sess.run(loss, feed_dict={x:self.valid_x, y: self.valid_y})
                cost_valid_set.append(val_cost)
                if (epoch > self.patience):
                    if (self.early_stopping(cost_train_set, self.patience) == False):
                        print ("early stopping training")
                        break
                print ("Epoch finished")
                print ('time for epoch: ', epoch + 1 , time.time()-start_time)
            print ('training ok!!!')
            
            
            prediction = sess.run(prediction, feed_dict={x:self.test_x, y: self.test_y})
            prediction = prediction * (self.max_y[0] - self.min_y[0]) +self.min_y[0] 
            prediction = np.asarray(prediction)
            MAE_err = MAE(prediction,self.test_y)
            RMSE_err = np.sqrt(MSE(prediction,self.test_y))
            error_model = np.asarray([MAE_err,RMSE_err])
            print (error_model)
            print (self.test_y.shape)
            print (prediction.shape)
            name_LSTM = ""
            for i in range(len(self.num_units_LSTM)):
                
                if (i == len(self.num_units_LSTM) - 1):
                    name_LSTM += str(self.num_units_LSTM[i])
                else:
                    name_LSTM += str(self.num_units_LSTM[i]) +'_'
            folder_to_save_result = 'results/LSTM/univariate/cpu/5minutes/ver1/'
            file_name = str(self.sliding) + '-' + str(self.batch_size) + '-' + name_LSTM + '-' + str(self.activation)+ '-' + str(self.optimizer) + '-' + str(self.input_dim) +'-'+str(self.dropout_rate)
            history_file = folder_to_save_result + 'history/' + file_name + '.png'
            prediction_file = folder_to_save_result + 'prediction/' + file_name + '.csv'
            save_path = saver.save(sess, folder_to_save_result + 'model_saved/' +  file_name)
            
            plt.plot(cost_train_set)
            plt.plot(cost_valid_set)

            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            # plt.show()
            # plt.savefig('/home/thangnguyen/hust/lab/machine_learning_handling/history/history_mem.png')
            plt.savefig(history_file)
            plt.close()
            predictionDf = pd.DataFrame(np.array(prediction))
            # predictionDf.to_csv('/home/thangnguyen/hust/lab/machine_learning_handling/results/result_mem.csv', index=False, header=None)
            predictionDf.to_csv(prediction_file, index=False, header=None)
            sess.close()
            return error_model
