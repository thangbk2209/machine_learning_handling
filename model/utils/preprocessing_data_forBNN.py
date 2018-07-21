import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 


class TimeseriesBNN:
    def __init__(self, original_data = None, train_size = None, valid_size = None, 
    sliding_encoder = None, sliding_decoder = None, time_steps = None):
        self.original_data = original_data
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder
        self.time_steps = time_steps
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
        print 'dataX_decoder'
        # print dataX_encoder[0]
        # print dataX_decoder[0]
        print self.minTimeseries
        print self.maxTimeSeries
        
        self.train_x_encoder = dataX_encoder[0:self.train_size - self.sliding_encoder]
        self.train_x_encoder = np.array(self.train_x_encoder)
        self.train_x_encoder = np.reshape(self.train_x_encoder, (self.train_x_encoder.shape[0], self.train_x_encoder.shape[1]/self.time_steps, self.time_steps))
        print 'self.train_x_encoder'
        print self.train_x_encoder
        self.valid_x_encoder = np.array(dataX_encoder[self.train_size - self.sliding_encoder: self.train_size + self.valid_size - self.sliding_encoder])
        self.valid_x_encoder = np.reshape(self.valid_x_encoder, (self.valid_x_encoder.shape[0], self.valid_x_encoder.shape[1]/self.time_steps, self.time_steps))
       
        self.test_x_encoder = np.array(dataX_encoder[self.train_size + self.valid_size - self.sliding_encoder:])
        self.test_x_encoder = np.reshape(self.test_x_encoder, (self.test_x_encoder.shape[0], self.test_x_encoder.shape[1]/self.time_steps, self.time_steps))
        
        self.train_x_decoder = np.array(dataX_decoder[self.sliding_encoder - self.sliding_decoder:self.train_size - self.sliding_decoder])
        self.train_x_decoder = np.reshape(self.train_x_decoder, (self.train_x_decoder.shape[0], self.train_x_decoder.shape[1]/self.time_steps, self.time_steps))
        
        self.valid_x_decoder = np.array(dataX_decoder[self.train_size - self.sliding_decoder: self.train_size + self.valid_size - self.sliding_decoder])
        self.valid_x_decoder = np.reshape(self.valid_x_decoder, (self.valid_x_decoder.shape[0], self.valid_x_decoder.shape[1]/self.time_steps, self.time_steps))
        self.test_x_decoder = np.array(dataX_decoder[self.train_size + self.valid_size - self.sliding_decoder:])
        self.test_x_decoder = np.reshape(self.test_x_decoder, (self.test_x_decoder.shape[0], self.test_x_decoder.shape[1]/self.time_steps, self.time_steps))

        self.train_y = self.original_scaled[self.sliding_encoder : self.train_size]
        self.valid_y = self.original_scaled[self.train_size : self.train_size + self.valid_size]
        self.test_y = self.original_scaled[self.train_size + self.valid_size:]
        return self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y, self.valid_y, self.test_y, self.minTimeseries, self.maxTimeSeries