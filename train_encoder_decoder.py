import tensorflow as tf
import numpy as np

from multiprocessing import Pool
# from queue import Queue
from sklearn.model_selection import ParameterGrid

from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 
from tensorflow.contrib import rnn
# from model.utils.preprocessing_data import Timeseries
# preprocessing_data_forBNN
from model.encoder_decoder import Model as encoder_decoder
from model.BNN import Model as BNN 

# queue = Queue()

link = './data/google_trace_timeseries/data_resource_usage_5Minutes_6176858948.csv'

colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv(link, header=None, index_col=False, names=colnames, usecols=[3,4,9,10], engine='python')
scaler = MinMaxScaler(feature_range=(0, 1))
cpu = df['cpu_rate'].values.reshape(-1,1)
mem = df['mem_usage'].values.reshape(-1,1)
disk_io_time = df['disk_io_time'].values
disk_space = df['disk_space'].values

# define constant
train_size = int(0.6 * len(cpu))
print (train_size)
valid_size = int(0.2 * len(cpu))

sliding_encoders = [18,24,30]
slding_decoders = [2,3,4,5,6]
sliding_inferences = [1,2,3,4,5]
# activation for inference layer : - 1 is sigmoid
#                                  - 2 is relu
#                                  - 3 is softmax 
activation_inferences = [1,2]
# num_units_LSTM_arr - array number units lstm for encoder and decoder
num_units_LSTM_arr = [4, 8, 16, 32]

batch_size_arr = [4, 8, 16, 32, 64, 128]
num_units_inference_arr = [4, 8, 16, 64]
metrics = [1,2]

activation_decoder = 1
num_layers = 1
learning_rate = 0.01
epochs_encoder_decoder = 2000
epochs_inference = 2000
input_dim = [1,2]
n_output_encoder_decoder = 1

for metric in metrics:
    for sliding_encoder in sliding_encoders:
        for sliding_decoder in slding_decoders:
            for sliding_inference in sliding_inferences:
                for activation_inference in activation_inferences:
                    for num_units_LSTM in num_units_LSTM_arr:
                        for batch_size in batch_size_arr:
                            for num_units_inference in num_units_inference_arr:
                                if(metric==1):
                                    print('mem1')
                                    model = BNN(mem, train_size, valid_size, 
                                        sliding_encoder, sliding_decoder, sliding_inference, batch_size,
                                        num_units_LSTM, num_layers, activation_decoder, activation_inference,
                                        # n_input = None, n_output = None,
                                        learning_rate, epochs_encoder_decoder,epochs_inference, input_dim, num_units_inference )
                                    model.fit()
                                else:
                                    model = BNN(cpu, train_size, valid_size, 
                                        sliding_encoder, sliding_decoder, sliding_inference, batch_size,
                                        num_units, num_layers, activation_decoder, activation_inference,
                                        # n_input = None, n_output = None,
                                        learning_rate, epochs_encoder_decoder,epochs_inference, input_dim, num_units_inference )
                                    model.fit()

# sliding_encoder = 20
# sliding_decoder = 4
# sliding_inference = 10
# activation_decoder = 1
# activation_inference = 1
# num_units = 4
# num_layers = 1
# learning_rate = 0.01
# epochs_encoder_decoder = 20
# epochs_inference = 20
# input_dim = 1
# n_output_encoder_decoder = 1
# batch_size = 4
# num_units_inference = 20

# model = BNN(mem, train_size, valid_size, 
#     sliding_encoder, sliding_decoder, sliding_inference, batch_size,
#     num_units, num_layers, activation_decoder, activation_inference,
#     # n_input = None, n_output = None,
#     learning_rate, epochs_encoder_decoder,epochs_inference, input_dim, num_units_inference )
# model.fit()