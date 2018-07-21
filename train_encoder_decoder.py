import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 
from tensorflow.contrib import rnn
# from model.utils.preprocessing_data import Timeseries
# preprocessing_data_forBNN
from model.encoder_decoder import Model

link = './data/google_trace_timeseries/data_resource_usage_10Minutes_6176858948.csv'

colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv(link, header=None, index_col=False, names=colnames, usecols=[3,4,9,10], engine='python')
scaler = MinMaxScaler(feature_range=(0, 1))
cpu = df['cpu_rate'].values.reshape(-1,1)
mem = df['mem_usage'].values.reshape(-1,1)
disk_io_time = df['disk_io_time'].values
disk_space = df['disk_space'].values

# define constant
train_size = int(0.6 * len(cpu))
valid_size = int(0.2 * len(cpu))
sliding_encoder = 12
sliding_decoder = 2
display_step = 1
activation = 1
num_units = 2
num_layers = 1
learning_rate = 0.01
epochs = 200
time_step = 1
n_output = 1
batch_size = 4


model = Model(cpu, train_size, valid_size, 
    sliding_encoder, sliding_decoder, batch_size,
    num_units, activation, num_layers, 
    # n_input = None, n_output = None,
    learning_rate, epochs, time_step, display_step )
model.fit()