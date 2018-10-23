import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd 
from tensorflow.contrib import rnn
# from utils.preprocessing_data import Timeseries
from model.utils.preprocessing_data_forBNN import FuzzyMultivariateTimeseriesBNNUber
import time
"""This class build the model BNN with initial function and train function"""
class Model:
    def __init__(self, original_data = None, prediction_data = None, external_feature = None, train_size = None, valid_size = None, 
    sliding_encoder = None, sliding_decoder = None, sliding_inference = None,
    batch_size = None, num_units_LSTM = None, num_layers = None,
    activation = None, optimizer = None,
    # n_input = None, n_output = None,
    learning_rate = None, epochs_encoder_decoder = None, epochs_inference = None, 
    input_dim = None, num_units_inference = None, patience = None, number_out_decoder = 1, dropout_rate = 0.8):
        self.original_data = original_data
        self.prediction_data = prediction_data
        self.external_feature = external_feature
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder
        self.sliding_inference = sliding_inference
        self.batch_size = batch_size
        self.num_units_LSTM = num_units_LSTM
        self.activation = activation
        self.optimizer = optimizer
        # self.n_input = n_input
        # self.n_output = n_output
        self.learning_rate = learning_rate
        self.epochs_encoder_decoder = epochs_encoder_decoder
        self.epochs_inference = epochs_inference
        self.input_dim = input_dim

        self.num_units_inference = num_units_inference
        self.patience = patience
        self.number_out_decoder = number_out_decoder
        self.dropout_rate = dropout_rate
    def preprocessing_data(self):
        timeseries = FuzzyMultivariateTimeseriesBNNUber(self.original_data, self.prediction_data, self.external_feature, self.train_size, self.valid_size, self.sliding_encoder, self.sliding_decoder, self.sliding_inference, self.input_dim,self.number_out_decoder)
        self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.min_y, self.max_y, self.train_x_inference, self.valid_x_inference, self.test_x_inference, self.train_y_inference, self.valid_y_inference, self.test_y_inference = timeseries.prepare_data()
    def init_RNN(self, num_units, activation):
        print (len(self.test_y_inference))
        print (self.test_x_encoder[-1])
        print (self.test_x_inference[-1])
        print (self.test_y_inference[-1])
        print (self.train_x_encoder[0])
        print (self.train_x_inference[0])
        print (self.train_y_inference[0])
        # lol
        print(num_units)
        num_layers = len(num_units)
        print (num_layers)
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
        print ("================check preprocessing data ok==================")
        print ('self.train_x_encoder')
        print (self.train_x_encoder[0])
        print ('self.train_x_decoder')
        print (self.train_x_decoder[0])
        print ('self.train_y_decoder')
        print (self.train_y_decoder[0])
        print (self.train_y_decoder.shape)
        print ('self.train_x_inference')
        print (self.train_x_inference[0])
        print ('self.train_y_inference')
        print (self.train_y_inference[0])
        print ('test y')
        print (self.test_y_inference)
        print (self.min_y)
        print (self.max_y)
        print (len(self.train_x_encoder))
        # lol111
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
        print (self.sliding_encoder)
        print (len(self.original_data))
        tf.reset_default_graph()
        x1 = tf.placeholder("float",[None, self.sliding_encoder*len(self.original_data)/self.input_dim, self.input_dim])
        x2 = tf.placeholder("float",shape = (None, self.sliding_decoder*len(self.original_data)/self.input_dim, self.input_dim))
        if(self.number_out_decoder == 1):
            y1 = tf.placeholder("float", [None, self.sliding_decoder])
            with tf.variable_scope('encoder'):
                
                encoder = self.init_RNN(self.num_units_LSTM,activation)
                # input_encoder=tf.unstack(x1 ,[None,self.sliding_encoder/self.time_step,self.time_step])
                outputs_encoder, new_state_encoder=tf.nn.dynamic_rnn(encoder, x1, dtype="float32")
                outputs_encoder = tf.identity(outputs_encoder, name='outputs_encoder')
            with tf.variable_scope('decoder'):
                decoder = self.init_RNN(self.num_units_LSTM,activation)
                outputs_decoder, new_state_decoder=tf.nn.dynamic_rnn(decoder, x2,dtype="float32", initial_state=new_state_encoder)
            
            prediction = outputs_decoder[:,:,-1]
            loss_encoder_decoder = tf.reduce_mean(tf.square(y1-prediction))
            optimizer_encoder_decoder = optimizer.minimize(loss_encoder_decoder)
        else:
            y11 = tf.placeholder("float", [None, self.sliding_decoder])
            y12 = tf.placeholder("float", [None, self.sliding_decoder])
            with tf.variable_scope('encoder'):
                encoder = self.init_RNN(self.num_units_LSTM,activation)
                # input_encoder=tf.unstack(x1 ,[None,self.sliding_encoder/self.time_step,self.time_step])
                outputs_encoder,new_state_encoder=tf.nn.dynamic_rnn(encoder, x1, dtype="float32")
                # with tf.control_dependencies([state.assign(state_encoder)]):
                #     outputs_encoder = tf.identity(outputs_encoder, name='outputs_encoder')
            with tf.variable_scope('decoder1'):
                decoder = self.init_RNN(self.num_units_LSTM,activation)
                outputs_decoder1,new_state_decoder1=tf.nn.dynamic_rnn(decoder, x2,dtype="float32", initial_state = new_state_encoder)
            with tf.variable_scope('decoder2'):
                decoder = self.init_RNN(self.num_units_LSTM,activation)
                outputs_decoder2,new_state_decoder2=tf.nn.dynamic_rnn(decoder, x2,dtype="float32", initial_state = new_state_encoder)
            prediction1 = outputs_decoder1[:,:,-1]
            prediction2 = outputs_decoder2[:,:,-1]
            loss_encoder_decoder = tf.reduce_mean(tf.square(y11-prediction1) + tf.square(y12-prediction2))
            optimizer_encoder_decoder = optimizer.minimize(loss_encoder_decoder)
        cost_train_encoder_decoder_set = []
        cost_valid_encoder_decoder_set = []
        # cost_train_inference_set = []
        # cost_valid_inference_set = []
        epoch_set=[]
        init=tf.global_variables_initializer()
        name_LSTM = ""
        for i in range(len(self.num_units_LSTM)):   
            if (i == len(self.num_units_LSTM) - 1):
                name_LSTM += str(self.num_units_LSTM[i])
            else:
                name_LSTM += str(self.num_units_LSTM[i]) +'_'
        folder_to_save_result = 'results/fuzzy/encoder_decoder/5minutes/cpu/bnn_multivariate_uber/'
        file_name = str(self.sliding_encoder) + '-' + str(self.sliding_decoder) + '-' + str(self.batch_size) + '-' + name_LSTM + '-' + str(self.activation)+ '-' + str(self.optimizer) + '-' + str(self.input_dim) +'-'+str(self.number_out_decoder) +'-'+str(self.dropout_rate)
        save_path = 'results/fuzzy/encoder_decoder/5minutes/cpu/bnn_multivariate_uber/model_saved/' +  file_name
            
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(init)
            # training encoder_decoder
            print ("start training encoder_decoder")
            builder = tf.saved_model.builder.SavedModelBuilder(save_path)
            if (self.number_out_decoder == 1):
                for epoch in range(self.epochs_encoder_decoder):
                    start_time = time.time()
                    # Train with each example
                    print ('epoch encoder_decoder: ', epoch+1)
                    total_batch = int(len(self.train_x_encoder)/self.batch_size)
                    # print (total_batch)
                    # sess.run(updates)
                    avg_cost = 0
                    for i in range(total_batch):
                        batch_xs_encoder,batch_xs_decoder = self.train_x_encoder[i*self.batch_size:(i+1)*self.batch_size], self.train_x_decoder[i*self.batch_size:(i+1)*self.batch_size]
                        batch_ys = self.train_y_decoder[i*self.batch_size:(i+1)*self.batch_size]
                        # print (sess.run(outputs_encoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder, y1:batch_ys}))
                        # print (sess.run(new_state_encoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder, y1:batch_ys}))
                        sess.run(optimizer_encoder_decoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder, y1:batch_ys})
                        avg_cost += sess.run(loss_encoder_decoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder, y1:batch_ys})/total_batch
                        if(i == total_batch -1):
                            a = sess.run(new_state_encoder,feed_dict={x1: batch_xs_encoder})
                    # Display logs per epoch step
                    print ("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost))
                    cost_train_encoder_decoder_set.append(avg_cost)
                    val_cost = sess.run(loss_encoder_decoder, feed_dict={x1:self.valid_x_encoder,x2:self.valid_x_decoder, y1: self.valid_y_decoder})
                    cost_valid_encoder_decoder_set.append(val_cost)
                    if (epoch > self.patience):
                        if (self.early_stopping(cost_train_encoder_decoder_set, self.patience) == False):
                            print ("early stopping encoder-decoder training")
                            break
                    print ("Epoch encoder-decoder finished")
                    print ('time for epoch encoder-decoder: ', epoch + 1 , time.time()-start_time)
                print ('training encoder-decoder ok!!!')
                builder.add_meta_graph_and_variables(sess, ["tag"], signature_def_map= {
                "model": tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs= {"x1": x1,"x2":x2},
                    outputs= {"y1": y1,"state": new_state_encoder[-1].h})
                })
                builder.save()
            else:
                for epoch in range(self.epochs_encoder_decoder):
                    start_time = time.time()
                    # Train with each example
                    print ('epoch encoder_decoder: ', epoch+1)
                    total_batch = int(len(self.train_x_encoder)/self.batch_size)
                    # print (total_batch)
                    # sess.run(updates)
                    avg_cost = 0
                    for i in range(total_batch):
                        batch_xs_encoder,batch_xs_decoder = self.train_x_encoder[i*self.batch_size:(i+1)*self.batch_size], self.train_x_decoder[i*self.batch_size:(i+1)*self.batch_size]
                        batch_ys1, batch_ys2 = self.train_y_decoder[0][i*self.batch_size:(i+1)*self.batch_size], self.train_y_decoder[1][i*self.batch_size:(i+1)*self.batch_size]
                        sess.run(optimizer_encoder_decoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder, y11:batch_ys1, y12:batch_ys2})
                        avg_cost += sess.run(loss_encoder_decoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder,y11:batch_ys1, y12:batch_ys2})/total_batch
                        if(i == total_batch -1):
                            a = sess.run(new_state_encoder,feed_dict={x1: batch_xs_encoder})
                    # Display logs per epoch step
                    print ("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost))
                    cost_train_encoder_decoder_set.append(avg_cost)
                    val_cost = sess.run(loss_encoder_decoder, feed_dict={x1:self.valid_x_encoder,x2:self.valid_x_decoder, y11: self.valid_y_decoder[0],y12: self.valid_y_decoder[1]})
                    cost_valid_encoder_decoder_set.append(val_cost)
                    if (epoch > self.patience):
                        if (self.early_stopping(cost_train_encoder_decoder_set, self.patience) == False):
                            print ("early stopping encoder-decoder training")
                            break
                    print ("Epoch encoder-decoder finished")
                    print ('time for epoch encoder-decoder: ', epoch + 1 , time.time()-start_time)

                print ('training encoder-decoder ok!!!')
                builder.add_meta_graph_and_variables(sess, ["tag"], signature_def_map= {
                "model": tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs= {"x1": x1,"x2":x2},
                    outputs= {"y11": y11,"y12": y12,"state": new_state_encoder[-1].h})
                })
                builder.save()
            vector_state_test = sess.run(new_state_encoder[-1].h,feed_dict={x1:self.test_x_encoder})
            vector_state_train = sess.run(new_state_encoder[-1].h,feed_dict={x1:self.train_x_encoder})
            print (vector_state_test.shape)
            print (vector_state_train.shape)
            vector_state = np.concatenate((vector_state_test,vector_state_train))
            history_file = folder_to_save_result + 'history/' + file_name + '.png'
            # prediction_file = folder_to_save_result + 'prediction/' + file_name + '.csv'
            vector_state_file = folder_to_save_result + 'vector_representation/' + file_name + '.csv'
            # uncertainty_file = folder_to_save_result + 'uncertainty/' + file_name + '.csv'
            
            plt.plot(cost_train_encoder_decoder_set)
            plt.plot(cost_valid_encoder_decoder_set)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train_encoder_decoder', 'validation_encoder_decoder'], loc='upper left')
            # plt.show()
            # plt.savefig('/home/thangnguyen/hust/lab/machine_learning_handling/history/history_mem.png')
            plt.savefig(history_file)
            plt.close()
            
            vector_stateDf = pd.DataFrame(np.array(vector_state))
            vector_stateDf.to_csv(vector_state_file, index=False, header=None)
            sess.close()