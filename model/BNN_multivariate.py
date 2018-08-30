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

"""This class build the model BNN with initial function and train function"""
class Model:
    def __init__(self, original_data = None, external_feature = None, train_size = None, valid_size = None, 
    sliding_encoder = None, sliding_decoder = None, sliding_inference = None,
    batch_size = None, num_units_LSTM = None, num_layers = None,
    activation = None, optimizer = None,
    # n_input = None, n_output = None,
    learning_rate = None, epochs_encoder_decoder = None, epochs_inference = None, 
    input_dim = None, num_units_inference = None, patience = None):
        self.original_data = original_data
        self.external_feature = external_feature
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder
        self.sliding_inference = sliding_inference
        self.batch_size = batch_size
        self.num_units_LSTM = num_units_LSTM
        self.num_layers = num_layers
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
    def preprocessing_data(self):
        # print ('timeseries')
        # print (self.original_data)
        # lol47
        timeseries = MultivariateTimeseriesBNN(self.original_data, self.external_feature, self.train_size, self.valid_size, self.sliding_encoder, self.sliding_decoder, self.sliding_inference, self.input_dim)
        # print ('timeseries')
        self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.min_y, self.max_y, self.train_x_inference, self.valid_x_inference, self.test_x_inference, self.train_y_inference, self.valid_y_inference, self.test_y_inference = timeseries.prepare_data()
    def init_RNN(self, num_units, num_layers, activation):
        cell = tf.contrib.rnn.LSTMCell(num_units,activation = activation)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                 input_keep_prob = 0.9, 
                                 output_keep_prob = 0.9,
                                 state_keep_prob = 0.9,
                                #  variational_recurrent = variational_recurrent,
                                 dtype=tf.float32)
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
        print ("================check preprocessing data ok==================")
        print ('self.train_x_encoder')
        print (self.train_x_encoder[0])
        print ('self.train_x_decoder')
        print (self.train_x_decoder[0])
        print ('self.train_x_inference')
        print (self.train_x_inference[0])
        print (self.min_y)
        print (self.max_y)
        # lol75
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

        tf.reset_default_graph()
        encoder = self.init_RNN(self.num_units_LSTM, self.num_layers,activation)
        x1 = tf.placeholder("float",[None, self.sliding_encoder*len(self.original_data)/self.input_dim, self.input_dim])
        # input_encoder=tf.unstack(x1 ,[None,self.sliding_encoder/self.time_step,self.time_step])
        outputs_encoder,state_encoder=tf.nn.dynamic_rnn(encoder, x1, dtype="float32")
        
        x2 = tf.placeholder("float",[None, self.sliding_decoder*len(self.original_data)/self.input_dim, self.input_dim])
        y1 = tf.placeholder("float", [None, self.n_output_encoder_decoder])
        x3 = tf.placeholder("float",[None, 1, int(self.sliding_inference*len(self.external_feature))])
        y2 = tf.placeholder("float", [None, self.n_output_inference])
        # input_decoder=tf.unstack(x2 ,self.sliding_decoder/self.time_step,self.time_step)
        outputs_decoder,state_decoder=tf.nn.dynamic_rnn(encoder, x2,dtype="float32", initial_state=state_encoder)
        out_weights=tf.Variable(tf.random_normal([int(self.sliding_decoder*len(self.original_data)/self.input_dim), self.n_output_encoder_decoder]))
        out_bias=tf.Variable(tf.random_normal([self.n_output_encoder_decoder]))
        # prediction = outputs_decoder[:,:,-1]
        
        prediction=activation(tf.matmul(outputs_decoder[:,:,-1],out_weights)+out_bias)
        prediction_inverse = prediction * (self.max_y[0] - self.min_y[0]) + self.min_y[0] 
        y1_inverse = y1 * (self.max_y[0] - self.min_y[0]) + self.min_y[0]
        # loss_function
        loss_encoder_decoder = tf.reduce_mean(tf.square(y1-prediction))
        #optimization
        optimizer_encoder_decoder = optimizer.minimize(loss_encoder_decoder)
        # state = tf.tile(state_encoder[-1].h)
        # state = tf.shape(x3)[0]
        state = tf.reshape(state_encoder[-1].h, [tf.shape(x3)[0], 1, self.num_units_LSTM])
        input_inference = tf.concat([x3,state],2)
        input_inference = tf.reshape(input_inference,[tf.shape(x3)[0], self.sliding_inference*len(self.external_feature) + self.num_units_LSTM])
        # state_encoder = np.reshape(state_encoder[-1].h, [4])
        # state_encoder = tf.reshape(state_encoder[-1].h,  [None, 1, self.num_units])
        # input_inference = tf.concat([x3, state_encoder[-1].h],0)
        hidden_value1 = tf.layers.dense(input_inference, self.num_units_inference, activation=activation)
        hidden_value2 = tf.layers.dense(hidden_value1, 4, activation=activation)
        output_inference = tf.layers.dense(hidden_value1,self.n_output_inference, activation=activation)
        # loss
        loss_inference = tf.reduce_mean(tf.square(y2-output_inference))
        #optimization
        optimizer_inference = optimizer.minimize(loss_inference)
        output_inference_inverse = output_inference * (self.max_y[0] - self.min_y[0]) + self.min_y[0]
        y2_inverse = y2 
        MAE = tf.reduce_mean(tf.abs(tf.subtract(output_inference_inverse,y2_inverse)) )
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(output_inference_inverse,y2_inverse))))
        cost_train_encoder_decoder_set = []
        cost_valid_encoder_decoder_set = []
        cost_train_inference_set = []
        cost_valid_inference_set = []
        # epoch_set=[]
        init=tf.global_variables_initializer()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(init)
            # training encoder_decoder
            print ("start training encoder_decoder")
            for epoch in range(self.epochs_encoder_decoder):
                # Train with each example
                print ('epoch encoder_decoder: ', epoch+1)
                total_batch = int(len(self.train_x_encoder)/self.batch_size)
                # print (total_batch)
                # sess.run(updates)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs_encoder,batch_xs_decoder ,batch_ys = self.train_x_encoder[i*self.batch_size:(i+1)*self.batch_size], self.train_x_decoder[i*self.batch_size:(i+1)*self.batch_size],self.train_y_inference[i*self.batch_size:(i+1)*self.batch_size]
                    sess.run(optimizer_encoder_decoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder, y1:batch_ys})
                    avg_cost += sess.run(loss_encoder_decoder,feed_dict={x1: batch_xs_encoder,x2: batch_xs_decoder,y1: batch_ys})/total_batch
                    if(i == total_batch -1):
                        a = sess.run(state_encoder,feed_dict={x1: batch_xs_encoder})
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
            print ('training encoder-decoder ok!!!')
            # training inferences
            print ('start training inference')
            for epoch in range(self.epochs_inference):
                print ("epoch inference: ", epoch+1)
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
                    # if(i == total_batch -1):
                    #     print (sess.run(state_encoder,feed_dict={x1: batch_xs_encoder}))
                print ("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost))
                cost_train_inference_set.append(avg_cost)
                # epoch_set.append(epoch+1)
                val_cost = sess.run(loss_inference, feed_dict={x1:self.valid_x_encoder,x3:self.valid_x_inference, y2: self.valid_y_inference})
                cost_valid_inference_set.append(val_cost)
                if (epoch > self.patience):
                    if (self.early_stopping(cost_train_inference_set , self.patience) == False):
                        print ("early stopping inference training")
                        break
                
            output_inference_inverse = sess.run(output_inference_inverse, feed_dict={x1:self.test_x_encoder,x3:self.test_x_inference, y2: self.test_y_inference})
            output_inference = sess.run(output_inference, feed_dict={x1:self.test_x_encoder,x3:self.test_x_inference, y2: self.test_y_inference})
            # print (output_inference)
            vector_state = sess.run(state_encoder[-1].h,feed_dict={x1:self.test_x_encoder})
            MAE = sess.run(MAE, feed_dict={x1:self.test_x_encoder,x3:self.test_x_inference, y2: self.test_y_inference})
            RMSE = sess.run(RMSE, feed_dict={x1:self.test_x_encoder,x3:self.test_x_inference, y2: self.test_y_inference})
            print ('MAE: ', MAE)
            print ('RMSE: ', RMSE)
            error = [MAE,RMSE]
            # print (output_inference_inverse)
            # print (self.test_y_inference)
            # print (len(output_inference_inverse))
            # print (len(self.test_y_inference))
            
    
            folder_to_save_result = 'results/mem/5minutes/'
            history_file = folder_to_save_result + 'history/' + str(self.sliding_encoder) + '-' + str(self.sliding_decoder) + '-' + str(self.sliding_inference) + '-' + str(self.batch_size) + '-' + str(self.num_units_LSTM) + '-' + str(self.num_layers) + '-' + str(self.activation) + '-' + str(self.input_dim) + '-' + str(self.num_units_inference) + '.png'
            prediction_file = folder_to_save_result + 'prediction/' + str(self.sliding_encoder) + '-' + str(self.sliding_decoder) + '-' + str(self.sliding_inference) + '-' + str(self.batch_size) + '-' + str(self.num_units_LSTM) + '-' + str(self.num_layers) + '-' + str(self.activation) + '-' + str(self.input_dim) + '-' + str(self.num_units_inference) + '.csv'
            vector_state_file = folder_to_save_result + 'vector_representation/' + str(self.sliding_encoder) + '-' + str(self.sliding_decoder) + '-' + str(self.sliding_inference) + '-' + str(self.batch_size) + '-' + str(self.num_units_LSTM) + '-' + str(self.num_layers) + '-' + str(self.activation) + '-' + str(self.input_dim) + '-' + str(self.num_units_inference) + '.csv'
            
            save_path = saver.save(sess, 'results/mem/5minutes/model_saved/' +  str(self.sliding_encoder) + '-' + str(self.sliding_decoder) + '-' + str(self.sliding_inference) + '-' + str(self.batch_size) + '-' + str(self.num_units_LSTM) + '-' + str(self.num_layers) + '-' + str(self.activation) + '-' + str(self.input_dim) + '-' + str(self.num_units_inference))
            
            plt.plot(cost_train_inference_set)
            plt.plot(cost_valid_inference_set)
            plt.plot(cost_train_encoder_decoder_set)
            plt.plot(cost_valid_encoder_decoder_set)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train_inference', 'validation_inference','train_encoder_decoder', 'validation_encoder_decoder'], loc='upper left')
            # plt.show()
            # plt.savefig('/home/thangnguyen/hust/lab/machine_learning_handling/history/history_mem.png')
            plt.savefig(history_file)
					
            predictionDf = pd.DataFrame(np.array(output_inference_inverse))
            # predictionDf.to_csv('/home/thangnguyen/hust/lab/machine_learning_handling/results/result_mem.csv', index=False, header=None)
            predictionDf.to_csv(prediction_file, index=False, header=None)
            # errorDf = pd.DataFrame(np.array(error))
            # errorDf.to_csv(error_file, index=False, header=None)
            # errorDf.to_csv(prediction_file, index=False, header=None)
            vector_stateDf = pd.DataFrame(np.array(vector_state))
            vector_stateDf.to_csv(vector_state_file, index=False, header=None)
            sess.close()
            return error