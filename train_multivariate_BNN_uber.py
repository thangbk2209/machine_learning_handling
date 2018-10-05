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
from model.BNN_multivariate_uber import Model as BNN_multivariate
import traceback
   
queue = Queue()

def train_model(item):
    # try:
    sliding_encoder = item["sliding_encoder"]
    sliding_decoder = item["sliding_decoder"]
    sliding_inference = item["sliding_inference"]
    batch_size = item["batch_size"]
    num_units_LSTM = item["num_unit_LSTM"]
    activation= item["activation"]
    input_dim = item["input_dim"]
    num_units_inference = item["num_units_inference"]    
    optimizer = item["optimizer"]
    number_out_decoder = item["number_out_decoder"]
    dropout_rate = item["dropout_rate"]
    model = BNN_multivariate(dataset_original, external_feature, train_size, valid_size, 
        sliding_encoder =  sliding_encoder, sliding_decoder = sliding_decoder,
        sliding_inference = sliding_inference, batch_size = batch_size,
        num_units_LSTM = num_units_LSTM, 
        activation = activation,optimizer = optimizer,
        learning_rate = learning_rate, epochs_encoder_decoder = epochs_encoder_decoder,
        epochs_inference = epochs_inference,
        input_dim = input_dim, num_units_inference = num_units_inference, patience = patience, 
        number_out_decoder = number_out_decoder, dropout_rate = dropout_rate)
    error = model.fit()
    name_LSTM = ""
    for i in range(len(num_units_LSTM)):
        
        if (i == len(num_units_LSTM) - 1):
            name_LSTM += str(num_units_LSTM[i])
        else:
            name_LSTM += str(num_units_LSTM[i]) +'_'
    name_inference = ""
    for i in range(len(num_units_inference)):
        
        if (i == len(num_units_inference) - 1):
            name_inference += str(num_units_inference[i])
        else:
            name_inference += str(num_units_inference[i]) +'_'
    file_name = str(sliding_encoder) + '-' + str(sliding_decoder) + '-' + str(sliding_inference) + '-' + str(batch_size) + '-' + name_LSTM + '-' + str(activation)+ '-' + str(optimizer) + '-' + str(input_dim) + '-' + name_inference +'-'+str(number_out_decoder) +'-'+str(dropout_rate)
            
    summary = open("results/multivariate/cpu/5minutes/evaluate_bnn_multivariate_uber.csv",'a+')
    summary.write(file_name +','+str(error[0])+','+str(error[1])+'\n')
    print (error)
    # except:
    #     traceback.print_stack()
# producer
# define constant
# data google trace
link = './data/google_trace_timeseries/data_resource_usage_5Minutes_6176858948.csv'
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv(link, header=None, index_col=False, names=colnames, usecols=[3,4,9,10], engine='python')
scaler = MinMaxScaler(feature_range=(0, 1))
cpu = df['cpu_rate'].values.reshape(-1,1)
mem = df['mem_usage'].values.reshape(-1,1)
disk_io_time = df['disk_io_time'].values.reshape(-1,1)
disk_space = df['disk_space'].values.reshape(-1,1)
dataset_original = [mem,cpu]
external_feature = [mem]

# data vn30
# link = './data/VNM_03092008_10092018.csv'
# colnames = ['date','CLOSE','OPEN','HIGH','LOW','VOLUME'] 
# df = read_csv(link, header=None, index_col=False, names=colnames, engine='python')
# scaler = MinMaxScaler(feature_range=(0, 1))
# date = df['date'].values.reshape(-1,1)
# CLOSE = df['CLOSE'].values.reshape(-1,1)
# OPEN = df['OPEN'].values.reshape(-1,1)
# HIGH = df['HIGH'].values.reshape(-1,1)
# LOW = df['LOW'].values.reshape(-1,1)
# VOLUME = df['VOLUME'].values.reshape(-1,1)
# print (VOLUME)
# # lol
# dataset_original = [VOLUME]
# external_feature = [VOLUME]

# dataset_original = np.concatenate((cpu,mem), axis = 1)
# print (dataset_original)
# lol61
train_size = int(0.6 * len(cpu))
# print (train_size)
valid_size = int(0.2 * len(cpu))


sliding_encoders = [18,24]
sliding_decoders = [2,4]
sliding_inferences = [8,10]
batch_size_arr = [8]
input_dim = [len(dataset_original)]
num_units_LSTM_arr = [[16,4],[32,4],[64,4]]
dropout_rate = [0.95]
# activation for inference and decoder layer : - 1 is sigmoid
#                                              - 2 is relu
#                                              - 3 is tanh
#                                              - 4 is elu
activation= [1,2]
# 1: momentum
# 2: adam
# 3: rmsprop

optimizers = [2,3]

learning_rate = 0.005
epochs_encoder_decoder = 1000
epochs_inference = 1000
patience = 20  #number of epoch checking for early stopping
# num_units_LSTM_arr - array number units lstm for encoder and decoder

num_units_inference_arr = [[64],[128]]
number_out_decoder = [1,2]
n_output_encoder_decoder = 1
param_grid = {
        'sliding_encoder': sliding_encoders,
        'sliding_decoder': sliding_decoders,
        'sliding_inference': sliding_inferences,
        'batch_size': batch_size_arr,
        'num_unit_LSTM': num_units_LSTM_arr,
        'activation': activation,
        'input_dim': input_dim,
        'num_units_inference': num_units_inference_arr,
        'optimizer':optimizers,
        'number_out_decoder':number_out_decoder,
        'dropout_rate':dropout_rate
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
    summary = open("results/multivariate/cpu/5minutes/evaluate_bnn_multivariate_uber.csv",'a+')
    summary.write("model,MAE,RMSE\n")
 
    pool = Pool(8)
    pool.map(train_model, list(queue.queue))
    pool.close()
    pool.join()
    pool.terminate()
