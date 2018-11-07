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
from model.utils.preprocessing_data_forBNN import MultivariateTimeseriesBNN
import time
from Graph import *

"""This class build the model BNN with initial function and train function"""
class Model:
    def __init__(self, original_data = None, prediction_data = None, external_feature = None, train_size = None, valid_size = None, 
    sliding_encoder = None, sliding_decoder = None, sliding_inference = None,
    batch_size = None, num_units_LSTM = None, num_layers = None, activation = None, optimizer = None, 
    learning_rate = None, epochs_encoder_decoder = None, epochs_inference = None,  input_dim = None, 
    num_units_inference = None, patience = None, number_out_decoder = None, dropout_rate = None, range_normalize = None):
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
        self.range_normalize = range_normalize
    def preprocessing_data(self):
        timeseries = MultivariateTimeseriesBNN(self.original_data, self.prediction_data, self.external_feature, self.train_size, self.valid_size, self.sliding_encoder, self.sliding_decoder, self.sliding_inference, self.input_dim, self.number_out_decoder, self.range_normalize)
        self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.min_prediction_arr, self.max_prediction_arr, self.train_x_inference, self.valid_x_inference, self.test_x_inference, self.train_y_inference, self.valid_y_inference, self.test_y_inference = timeseries.prepare_data()
        
    def init_RNN(self, num_units, activation):
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
                                        input_keep_prob = 1.0, 
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
            prev_layer = tf.layers.dropout(prev_layer , rate = self.dropout_rate)
        
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
        self.train_x_encoder = np.array(self.train_x_encoder)
        self.train_x_decoder = np.array(self.train_x_decoder)
        self.test_x_encoder = np.array(self.test_x_encoder)
        self.test_x_decoder = np.array(self.test_x_decoder)
        self.test_y_decoder = np.array(self.test_y_decoder)
        self.train_x_inference = np.array(self.train_x_inference)
        self.test_x_inference = np.array(self.test_x_inference)
        self.n_input_encoder = self.train_x_encoder.shape[1]
        self.n_input_decoder = self.train_x_decoder.shape[1]
        self.n_output_inference = self.train_y_inference.shape[1]

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
        x1 = tf.placeholder("float",[None, self.sliding_encoder*len(self.original_data)/self.input_dim, self.input_dim])
        x2 = tf.placeholder("float",shape = (None, self.sliding_decoder*len(self.original_data)/self.input_dim, self.input_dim))
        
        if(self.number_out_decoder == 1):
            y1 = tf.placeholder("float", [None, 1])
            with tf.variable_scope('encoder'):
                
                encoder = self.init_RNN(self.num_units_LSTM,activation)
                outputs_encoder, new_state_encoder=tf.nn.dynamic_rnn(encoder, x1, dtype="float32")
                outputs_encoder = tf.identity(outputs_encoder, name='outputs_encoder')
            with tf.variable_scope('decoder'):
                decoder = self.init_RNN(self.num_units_LSTM,activation)
                outputs_decoder, new_state_decoder=tf.nn.dynamic_rnn(decoder, x2,dtype="float32", initial_state=new_state_encoder)
            out_weights=tf.Variable(tf.random_normal([int(self.num_units_LSTM[-1]), 1]))
            out_bias=tf.Variable(tf.random_normal([1]))
            
            prediction = activation(tf.matmul(outputs_decoder[:,-1,:],out_weights)+out_bias)
            loss_encoder_decoder = tf.reduce_mean(tf.square(y1-prediction))
            optimizer_encoder_decoder = optimizer.minimize(loss_encoder_decoder)
        else:
            y11 = tf.placeholder("float", [None, 1])
            y12 = tf.placeholder("float", [None, 1])
            with tf.variable_scope('encoder'):
                encoder = self.init_RNN(self.num_units_LSTM,activation)
                outputs_encoder,new_state_encoder=tf.nn.dynamic_rnn(encoder, x1, dtype="float32")
            with tf.variable_scope('decoder1'):
                decoder = self.init_RNN(self.num_units_LSTM,activation)
                outputs_decoder1,new_state_decoder1=tf.nn.dynamic_rnn(decoder, x2,dtype="float32", initial_state = new_state_encoder)
            with tf.variable_scope('decoder2'):
                decoder = self.init_RNN(self.num_units_LSTM,activation)
                outputs_decoder2,new_state_decoder2=tf.nn.dynamic_rnn(decoder, x2,dtype="float32", initial_state = new_state_encoder)
            
            out_weights1=tf.Variable(tf.random_normal([int(self.num_units_LSTM[-1]), 1]))
            out_bias1=tf.Variable(tf.random_normal([1]))
            prediction1 = activation(tf.matmul(outputs_decoder1[:,-1,:],out_weights1)+out_bias1)

            out_weights2=tf.Variable(tf.random_normal([int(self.num_units_LSTM[-1]), 1]))
            out_bias2=tf.Variable(tf.random_normal([1]))
            prediction2 = activation(tf.matmul(outputs_decoder2[:,-1,:],out_weights2)+out_bias2)
            
            loss_encoder_decoder = tf.reduce_mean(tf.square(y11-prediction1) + tf.square(y12-prediction2))
            optimizer_encoder_decoder = optimizer.minimize(loss_encoder_decoder)
        x3 = tf.placeholder("float",[None, 1, int(self.sliding_inference*len(self.external_feature))])
        y2 = tf.placeholder("float", [None, self.n_output_inference])
       
        state = tf.reshape(new_state_encoder[-1].h, [tf.shape(x3)[0], 1, self.num_units_LSTM[-1]])
        input_inference = tf.concat([x3,state],2)
        input_inference = tf.reshape(input_inference,[tf.shape(x3)[0], self.sliding_inference*len(self.external_feature) + self.num_units_LSTM[-1]])
        output_inference = self.mlp(input_inference, self.num_units_inference, activation)
        loss_inference = tf.reduce_mean(tf.square(y2-output_inference))
        #optimization
        optimizer_inference = optimizer.minimize(loss_inference)
        output_inference_inverse = output_inference * (self.max_prediction_arr[0] - self.min_prediction_arr[0]) + self.min_prediction_arr[0]
        y2_inverse = y2 
        MAE = tf.reduce_mean(tf.abs(tf.subtract(output_inference_inverse,y2_inverse)) )
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(output_inference_inverse,y2_inverse))))
        cost_train_encoder_decoder_set = []
        cost_valid_encoder_decoder_set = []
        cost_train_inference_set = []
        cost_valid_inference_set = []
        epoch_set=[]
        init=tf.global_variables_initializer()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(init)
            # training encoder_decoder
            print ("start training encoder_decoder")
            if (self.number_out_decoder == 1):
                for epoch in range(self.epochs_encoder_decoder):
                    start_time = time.time()
                    # Train with each example
                    print ('epoch encoder_decoder: ', epoch+1)
                    total_batch = int(len(self.train_x_encoder)/self.batch_size)
                    avg_cost = 0
                    for i in range(total_batch):
                        batch_xs_encoder,batch_xs_decoder = self.train_x_encoder[i*self.batch_size:(i+1)*self.batch_size], self.train_x_decoder[i*self.batch_size:(i+1)*self.batch_size]
                        batch_ys = self.train_y_decoder[i*self.batch_size:(i+1)*self.batch_size]
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
            else:
                for epoch in range(self.epochs_encoder_decoder):
                    start_time = time.time()
                    # Train with each example
                    print ('epoch encoder_decoder: ', epoch+1)
                    total_batch = int(len(self.train_x_encoder)/self.batch_size)
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
                   
                    if (epoch > self.patience):
                        if (self.early_stopping(cost_train_encoder_decoder_set, self.patience) == False):
                            print ("early stopping encoder-decoder training")
                            break
                    print ("Epoch encoder-decoder finished")
                    print ('time for epoch encoder-decoder: ', epoch + 1 , time.time()-start_time)

                print ('training encoder-decoder ok!!!')

            # training inferences
            print ('start training inference')
            for epoch in range(self.epochs_inference):
                start_time = time.time()
                print ("epoch inference: ", epoch+1)
                total_batch = int(len(self.train_x_inference)/self.batch_size)
                # print (total_batch)
                # sess.run(updates)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs_encoder,batch_xs_inference ,batch_ys = self.train_x_encoder[i*self.batch_size:(i+1)*self.batch_size], self.train_x_inference[i*self.batch_size:(i+1)*self.batch_size],self.train_y_inference[i*self.batch_size:(i+1)*self.batch_size]
                  
                    sess.run(optimizer_inference,feed_dict={x1: batch_xs_encoder,x3: batch_xs_inference, y2:batch_ys})
                    avg_cost += sess.run(loss_inference,feed_dict={x1: batch_xs_encoder,x3: batch_xs_inference,y2: batch_ys})/total_batch
                    
                print ("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost))
                cost_train_inference_set.append(avg_cost)
                
                val_cost = sess.run(loss_inference, feed_dict={x1:self.valid_x_encoder,x3:self.valid_x_inference, y2: self.valid_y_inference})
                cost_valid_inference_set.append(val_cost)
                if (epoch > self.patience):
                    if (self.early_stopping(cost_train_inference_set , self.patience) == False):
                        print ("early stopping inference training")
                        break
                print ('time for epoch inference: ', epoch + 1 , time.time()-start_time)

         
            vector_state = sess.run(new_state_encoder[-1].h,feed_dict={x1:self.test_x_encoder})
            outputs = []
            MSE = []
            error_model = []
            B = 20
            for i in range(B):
                
                MAEi = sess.run(MAE, feed_dict={x1:self.test_x_encoder,x3:self.test_x_inference, y2: self.test_y_inference})
                RMSEi = sess.run(RMSE, feed_dict={x1:self.test_x_encoder,x3:self.test_x_inference, y2: self.test_y_inference})
                output_inference_inversei = sess.run(output_inference_inverse, feed_dict={x1:self.test_x_encoder,x3:self.test_x_inference, y2: self.test_y_inference})
                print (i,'MAE: ', MAEi,'RMSE: ', RMSEi)
                
                errori = [MAEi, RMSEi]
                error_model.append(errori)
                outputs.append(output_inference_inversei)
            output_inference_inverse_valid = sess.run(output_inference_inverse, feed_dict={x1:self.valid_x_encoder,x3:self.valid_x_inference, y2: self.valid_y_inference})
            err_valid = 0
            error_model = np.average(error_model,axis = 0)
            for i in range(len(output_inference_inverse_valid)):
                test_valid = self.valid_y_inference * (self.max_prediction_arr[0] - self.min_prediction_arr[0]) + self.min_prediction_arr[0]
                err_valid += np.square(output_inference_inverse_valid[i][0]-test_valid[i])/len(output_inference_inverse_valid)
            y_pre = []
            error = []
            for k in range(len(self.test_y_inference)):
                errork = 0
                outk = 0
                y_prei = []
                errori = []
                for t in range(B):
                    outk += outputs[t][k][0]/B
                    errork += np.square(self.test_y_inference[k] - outputs[t][k][0])
                errori.append(errork)
                y_prei.append(outk)
                y_pre.append(y_prei)
                error.append(errori)
            
            uncertainty = []
            for i in range(len(error)):
                uncertainty_i = np.sqrt(error[i][0] + err_valid[0])
                uncertainty.append(uncertainty_i)
            name_LSTM = ""
            for i in range(len(self.num_units_LSTM)):
                
                if (i == len(self.num_units_LSTM) - 1):
                    name_LSTM += str(self.num_units_LSTM[i])
                else:
                    name_LSTM += str(self.num_units_LSTM[i]) +'_'
            name_inference = ""
            for i in range(len(self.num_units_inference)):
                
                if (i == len(self.num_units_inference) - 1):
                    name_inference += str(self.num_units_inference[i])
                else:
                    name_inference += str(self.num_units_inference[i]) +'_'
            folder_to_save_result = 'results/fuzzy/multivariate/mem/5minutes/bnn_multivariate/'
            file_name = str(self.sliding_encoder) + '-' + str(self.sliding_decoder) + '-' + str(self.sliding_inference) + '-' + str(self.batch_size) + '-' + name_LSTM + '-' + str(self.activation)+'-'+str(self.optimizer) + '-' + str(self.input_dim) + '-' + name_inference+'-'+str(self.dropout_rate)
            history_file = folder_to_save_result + 'history/' + file_name + '.png'
            prediction_file = folder_to_save_result + 'prediction/' + file_name + '.csv'
            vector_state_file = folder_to_save_result + 'vector_representation/' + file_name + '.csv'
            uncertainty_file = folder_to_save_result + 'uncertainty/' + file_name + '.csv'
            save_path = saver.save(sess, 'results/fuzzy/multivariate/mem/5minutes/bnn_multivariate/model_saved/' +  file_name) +'/model'
            plt.plot(cost_train_inference_set)
            plt.plot(cost_valid_inference_set)
            plt.plot(cost_train_encoder_decoder_set)
            plt.plot(cost_valid_encoder_decoder_set)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train_inference', 'validation_inference','train_encoder_decoder', 'validation_encoder_decoder'], loc='upper left')
            # plt.show()
            plt.close()
            # plt.savefig('/home/thangnguyen/hust/lab/machine_learning_handling/history/history_mem.png')
            plt.savefig(history_file)
            print ('done')
            predictionDf = pd.DataFrame(np.array(y_pre))
            # predictionDf.to_csv('/home/thangnguyen/hust/lab/machine_learning_handling/results/result_mem.csv', index=False, header=None)
            predictionDf.to_csv(prediction_file, index=False, header=None)
            # draw_predict(self.test_y_inference, predictionDf,file_name + '.png',folder_to_save_result +'plot/')
            uncertaintyDf = pd.DataFrame(np.array(uncertainty))
            uncertaintyDf.to_csv(uncertainty_file, index=False, header=None)
            # errorDf.to_csv(prediction_file, index=False, header=None)
            vector_stateDf = pd.DataFrame(np.array(vector_state))
            vector_stateDf.to_csv(vector_state_file, index=False, header=None)
            sess.close()
            return error_model