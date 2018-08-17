import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from multiprocessing import Pool
from queue import Queue
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
import traceback

queue = Queue()

def train_model(item):
    # try:
    sliding_encoder = item["sliding_encoder"]
    sliding_decoder = item["sliding_decoder"]
    sliding_inference = item["sliding_inference"]
    batch_size = item["batch_size"]
    num_units_LSTM = item["num_unit_LSTM"]
    num_layer = item["num_layer"]
    activation_decoder = item["activation_decoder"]
    activation_inference = item["activation_inference"]
    input_dim = item["input_dim"]
    num_units_inference = item["num_units_inference"]    

    model = BNN(dataset_original, train_size, valid_size, 
        sliding_encoder =  sliding_encoder, sliding_decoder = sliding_decoder,
        sliding_inference = sliding_inference, batch_size = batch_size,
        num_units_LSTM = num_units_LSTM, num_layers = num_layer, 
        activation_decoder = activation_decoder, activation_inference = activation_inference, 
        learning_rate = learning_rate, epochs_encoder_decoder = epochs_encoder_decoder,
        epochs_inference = epochs_inference,
        input_dim = input_dim, num_units_inference = num_units_inference, patience = patience )
    model.fit()
    # except:
    #     traceback.print_stack()
# producer
# define constant
link = './data/google_trace_timeseries/data_resource_usage_5Minutes_6176858948.csv'
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv(link, header=None, index_col=False, names=colnames, usecols=[3,4,9,10], engine='python')
scaler = MinMaxScaler(feature_range=(0, 1))
cpu = df['cpu_rate'].values.reshape(-1,1)
mem = df['mem_usage'].values.reshape(-1,1)
disk_io_time = df['disk_io_time'].values
disk_space = df['disk_space'].values

dataset_original = mem
train_size = int(0.6 * len(cpu))
# print (train_size)
valid_size = int(0.2 * len(cpu))


sliding_encoders = [12,15,18,21,24]
sliding_decoders = [2,3,4,5]
sliding_inferences = [1,2,3,4,5,6,7,8,9,10,11,12]
batch_size_arr = [4,8,16,32,64,128]
num_units_LSTM_arr = [2,4,8]
num_layers = [1]
# activation for inference and decoder layer : - 1 is sigmoid
#                                              - 2 is relu
#                                              - 3 is tanh
#                                              - 4 is elu
activation_decoder = [1,2,3,4]
activation_inferences = [1,2,3,4]
learning_rate = 0.01
epochs_encoder_decoder = 2000
epochs_inference = 2000
patience = 20  #number of epoch checking for early stopping
# num_units_LSTM_arr - array number units lstm for encoder and decoder
input_dim = [1]
num_units_inference_arr = [4,6,8,10,12,14,16]

# n_output_encoder_decoder = 1
param_grid = {
        'sliding_encoder': sliding_encoders,
        'sliding_decoder': sliding_decoders,
        'sliding_inference': sliding_inferences,
        'batch_size': batch_size_arr,
        'num_unit_LSTM': num_units_LSTM_arr,
        'num_layer': num_layers,
        'activation_decoder': activation_decoder,
        'activation_inference': activation_inferences,
        'input_dim': input_dim,
        'num_units_inference': num_units_inference_arr
    }
# Create combination of params.
print ("check")
a = ParameterGrid(param_grid)
print(type(a))
# print (a.__len__())
for item in list(ParameterGrid(param_grid)) :
    queue.put_nowait(item)
# Consumer
if __name__ == '__main__':
    pool = Pool(4)
    pool.map(train_model, list(queue.queue))
    pool.close()
    pool.join()
    pool.terminate()

# for metric in metrics:
#     for sliding_encoder in sliding_encoders:
#         for sliding_decoder in slding_decoders:
#             for sliding_inference in sliding_inferences:
#                 for activation_inference in activation_inferences:
#                     for num_units_LSTM in num_units_LSTM_arr:
#                         for batch_size in batch_size_arr:
#                             for num_units_inference in num_units_inference_arr:
#                                 if(metric==1):
#                                     print('mem1')
#                                     model = BNN(mem, train_size, valid_size, 
#                                         sliding_encoder, sliding_decoder, sliding_inference, batch_size,
#                                         num_units_LSTM, num_layers, activation_decoder, activation_inference,
#                                         # n_input = None, n_output = None,
#                                         learning_rate, epochs_encoder_decoder,epochs_inference, input_dim, num_units_inference )
#                                     model.fit()
#                                 else:
#                                     model = BNN(cpu, train_size, valid_size, 
#                                         sliding_encoder, sliding_decoder, sliding_inference, batch_size,
#                                         num_units, num_layers, activation_decoder, activation_inference,
#                                         # n_input = None, n_output = None,
#                                         learning_rate, epochs_encoder_decoder,epochs_inference, input_dim, num_units_inference )
#                                     model.fit()
# sliding_encoder = 12
# sliding_decoder = 4
# sliding_inference = 6
# activation_decoder = 1
# activation_inference = 1
# num_units = 2
# num_layers = 1
# learning_rate = 0.01
# epochs_encoder_decoder = 200
# epochs_inference = 200
# input_dim = 1
# n_output_encoder_decoder = 1
# batch_size = 4
# num_units_inference = 20
# patience = 20
# model = BNN(cpu, train_size, valid_size, 
#     sliding_encoder, sliding_decoder, sliding_inference, batch_size,
#     num_units, num_layers, activation_decoder, activation_inference,
#     # n_input = None, n_output = None,
#     learning_rate, epochs_encoder_decoder,epochs_inference, input_dim, num_units_inference, patience)
# model.fit() 