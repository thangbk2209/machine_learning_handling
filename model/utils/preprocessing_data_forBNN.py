import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 


class TimeseriesEncoderDecoder:
    def __init__(self, original_data = None, train_size = None, valid_size = None, 
    sliding_encoder = None, sliding_decoder = None, input_dim = None):
        self.original_data = original_data
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder
        self.input_dim = input_dim
    def scaling_data(self, X):
        min = np.amin(X)
        max = np.amax(X)
        mean = np.mean(X)
        scale = (X-min)/(max-min)
        return scale, min, max
    def create_x(self, timeseries, sliding):
        dataX = []
        for i in range(len(timeseries)-sliding):
            datai = []
            for j in range(sliding):
                datai.append(timeseries[i+j])
            dataX.append(datai)
        return dataX
    def prepare_data(self):
        self.original_scaled, self.minTimeseries, self.maxTimeSeries = self.scaling_data(self.original_data)
        dataX_encoder = self.create_x(self.original_scaled, self.sliding_encoder)
        dataX_decoder = self.create_x(self.original_scaled, self.sliding_decoder)
        print ('dataX_decoder')
        # print dataX_encoder[0]
        # print dataX_decoder[0]
        print (self.minTimeseries)
        print (self.maxTimeSeries)
        
        self.train_x_encoder = dataX_encoder[0:self.train_size - self.sliding_encoder]
        self.train_x_encoder = np.array(self.train_x_encoder)
        self.train_x_encoder = np.reshape(self.train_x_encoder, (self.train_x_encoder.shape[0], int(self.train_x_encoder.shape[1]/self.input_dim), self.input_dim))
        print ('self.train_x_encoder')
        print (self.train_x_encoder)
        self.valid_x_encoder = np.array(dataX_encoder[self.train_size - self.sliding_encoder: self.train_size + self.valid_size - self.sliding_encoder])
        self.valid_x_encoder = np.reshape(self.valid_x_encoder, (self.valid_x_encoder.shape[0], int(self.valid_x_encoder.shape[1]/self.input_dim), self.input_dim))
       
        self.test_x_encoder = np.array(dataX_encoder[self.train_size + self.valid_size - self.sliding_encoder:])
        self.test_x_encoder = np.reshape(self.test_x_encoder, (self.test_x_encoder.shape[0], int(self.test_x_encoder.shape[1]/self.input_dim), self.input_dim))
        
        self.train_x_decoder = np.array(dataX_decoder[self.sliding_encoder - self.sliding_decoder:self.train_size - self.sliding_decoder])
        self.train_x_decoder = np.reshape(self.train_x_decoder, (self.train_x_decoder.shape[0], int(self.train_x_decoder.shape[1]/self.input_dim), self.input_dim))
        
        self.valid_x_decoder = np.array(dataX_decoder[self.train_size - self.sliding_decoder: self.train_size + self.valid_size - self.sliding_decoder])
        self.valid_x_decoder = np.reshape(self.valid_x_decoder, (self.valid_x_decoder.shape[0], int(self.valid_x_decoder.shape[1]/self.input_dim), self.input_dim))
        self.test_x_decoder = np.array(dataX_decoder[self.train_size + self.valid_size - self.sliding_decoder:])
        self.test_x_decoder = np.reshape(self.test_x_decoder, (self.test_x_decoder.shape[0], int(self.test_x_decoder.shape[1]/self.input_dim), self.input_dim))

        self.train_y = self.original_scaled[self.sliding_encoder : self.train_size]
        self.valid_y = self.original_scaled[self.train_size : self.train_size + self.valid_size]
        self.test_y = self.original_scaled[self.train_size + self.valid_size:]
        return self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y, self.valid_y, self.test_y, self.minTimeseries, self.maxTimeSeries

class TimeseriesBNN:
    def __init__(self, original_data = None, train_size = None, valid_size = None, 
        sliding_encoder = None, sliding_decoder = None, sliding_inference = None ,input_dim = None):
        self.original_data = original_data
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder
        self.sliding_inference = sliding_inference
        self.input_dim = input_dim 
    def prepare_data(self):
        timeseries_encoder_decoder = TimeseriesEncoderDecoder(self.original_data, self.train_size, self.valid_size, self.sliding_encoder, self.sliding_decoder, self.input_dim)
        self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.minTimeseries, self.maxTimeSeries  = timeseries_encoder_decoder.prepare_data()
        self.original_scaled = timeseries_encoder_decoder.original_scaled
        data_x_inference = timeseries_encoder_decoder.create_x(timeseries_encoder_decoder.original_scaled, self.sliding_inference)
        self.train_x_inference = np.array(data_x_inference[0:self.train_size - self.sliding_inference])
        self.train_x_inference = np.reshape(self.train_x_inference, (self.train_x_inference.shape[0], int(self.train_x_inference.shape[1]/self.input_dim), self.input_dim))
        self.valid_x_inference = np.array(data_x_inference[self.train_size - self.sliding_inference: self.train_size + self.valid_size - self.sliding_inference])
        self.test_x_inference = np.array(data_x_inference[self.train_size + self.valid_size - self.sliding_inference:])
        
        self.train_y_inference = self.original_scaled[self.sliding_inference : self.train_size]
        self.valid_y_inference = self.original_scaled[self.train_size : self.train_size + self.valid_size]
        self.test_y_inference = self.original_scaled[self.train_size + self.valid_size:]
        return self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.minTimeseries, self.maxTimeSeries, self.train_x_inference, self.valid_x_inference, self.test_x_inference, self.train_y_inference, self.valid_y_inference, self.test_y_inference