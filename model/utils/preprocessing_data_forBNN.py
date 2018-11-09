import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 

"""This class create input-output pair for encoder-decoder models"""
class TimeseriesEncoderDecoder:
    def __init__(self, original_data = None, train_size = None, valid_size = None, 
    sliding_encoder = None, sliding_decoder = None, input_dim = None):
        self.original_data = original_data
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder
        self.input_dim = input_dim
    # scaling data to interval [0,1] 
    def scaling_data(self, X):
        minX = np.amin(X)
        maxX = np.amax(X)
        mean = np.mean(X)
        scale = (X-minX)/(maxX-minX)
        return scale, minX, maxX
    
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
        
        self.train_x_encoder = dataX_encoder[0:self.train_size - self.sliding_encoder]
        self.train_x_encoder = np.array(self.train_x_encoder)
        self.train_x_encoder = np.reshape(self.train_x_encoder, (self.train_x_encoder.shape[0], int(self.train_x_encoder.shape[1]/self.input_dim), self.input_dim))

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
"""This class create input-ouput pair for BNN model with univariate input
include:
    input_encoder, input_decoder, output of encoder_decoder model
    input inference -- like external features and ouput of inference - like output of BNN model
"""
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
        self.train_x_inference = np.array(data_x_inference[self.sliding_encoder - self.sliding_inference:self.train_size - self.sliding_inference])
        self.train_x_inference = np.reshape(self.train_x_inference, (self.train_x_inference.shape[0], 1, int(self.train_x_inference.shape[1])))
        
        self.valid_x_inference = np.array(data_x_inference[self.train_size - self.sliding_inference: self.train_size + self.valid_size - self.sliding_inference])
        self.valid_x_inference = np.reshape(self.valid_x_inference, (self.valid_x_inference.shape[0], 1, int(self.valid_x_inference.shape[1])))

        self.test_x_inference = np.array(data_x_inference[self.train_size + self.valid_size - self.sliding_inference:])
        self.test_x_inference = np.reshape(self.test_x_inference, (self.test_x_inference.shape[0], 1, int(self.test_x_inference.shape[1])))

        self.train_y_inference = self.original_scaled[self.sliding_encoder: self.train_size]
        self.valid_y_inference = self.original_scaled[self.train_size: self.train_size + self.valid_size]
        self.test_y_inference = self.original_scaled[self.train_size + self.valid_size:]
        return self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.minTimeseries, self.maxTimeSeries, self.train_x_inference, self.valid_x_inference, self.test_x_inference, self.train_y_inference, self.valid_y_inference, self.test_y_inference

"""
This class prepare multivariate input and output for BNN

"""
class MultivariateTimeseriesBNN:
    def __init__(self, original_data = None, prediction_data = None, external_feature = None, train_size = None, valid_size = None, 
        sliding_encoder = None, sliding_decoder = None, sliding_inference = None ,input_dim = None, number_out_decoder = None, range_normalize = None):
        self.original_data = original_data
        self.prediction_data = prediction_data
        self.external_feature = external_feature
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder
        self.sliding_inference = sliding_inference
        self.input_dim = input_dim 
        self.number_out_decoder = number_out_decoder
        self.range_normalize = range_normalize
    def prepare_data(self):
        if(self.range_normalize == False):
            print ('==================false======================')
            self.scaled_original_data, self.min_original_arr, self.max_original_arr = self.scale_timeseries(self.original_data)
            self.scaled_prediction_data, self.min_prediction_data, self.max_prediction_data = self.scale_timeseries(self.prediction_data)
            print ("==============check scale data============")
            print (self.scaled_original_data)
            print (self.min_original_arr)
            print (self.max_original_arr)
            # lol121
            self.scaled_external_feature, self.min_ext, self.max_ext = self.scale_timeseries(self.external_feature)
            self.multivariate_original_timeseries = self.create_multivariate_timeseries(self.scaled_original_data)
            # check create multivariate timeseries for encoder,decoder
            print ("=================check create timeseries for encoder-decoder====================")
            print (self.multivariate_original_timeseries)
            # print (self.scaled_external_feature)
            print ('================scaled external feature===============')
            self.ext_timeseries = self.create_multivariate_timeseries(self.scaled_external_feature)
            # print (self.multivariate_timeseries)
            print ('=============check external timeseries===============')
            print (self.ext_timeseries)
            # lol135
            dataX_encoder = self.create_x(self.multivariate_original_timeseries, self.sliding_encoder)
            dataX_decoder = self.create_x(self.multivariate_original_timeseries, self.sliding_decoder)
            dataX_ext = self.create_x(self.ext_timeseries, self.sliding_inference)
            # check create input-output pair for training
            # print ('min_arr')
            # print (self.min_arr)
            # print ('max_arr')
            # print (self.max_arr)
            # print ('dataX_encoder')
            # print (dataX_encoder[0])
            # print ('dataX_decoder')
            # print (dataX_decoder[0])
            # print ('dataX_ext')
            # print (dataX_ext[0])
            # lol150
            # print (dataX_encoder[0])
            # print (dataX_encoder[1])
            # # lol124
            # print (dataX_encoder[1])
            # print (dataX_decoder[0])
            # print (dataX_decoder[1])
            print ("==============check train x==============")
            self.train_x_encoder = dataX_encoder[0:self.train_size - self.sliding_encoder]
            print (self.train_x_encoder[0])
            # lol131
            self.train_x_encoder = np.array(self.train_x_encoder)
            
            # lol134
            self.train_x_encoder = np.reshape(self.train_x_encoder, (self.train_x_encoder.shape[0], int(self.train_x_encoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
            
            self.valid_x_encoder = np.array(dataX_encoder[self.train_size - self.sliding_encoder: self.train_size + self.valid_size - self.sliding_encoder])
            self.valid_x_encoder = np.reshape(self.valid_x_encoder, (self.valid_x_encoder.shape[0], int(self.valid_x_encoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        
            self.test_x_encoder = np.array(dataX_encoder[self.train_size + self.valid_size - self.sliding_encoder:])
            self.test_x_encoder = np.reshape(self.test_x_encoder, (self.test_x_encoder.shape[0], int(self.test_x_encoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
            
            self.train_x_decoder = np.array(dataX_decoder[self.sliding_encoder - self.sliding_decoder:self.train_size - self.sliding_decoder])
            print (self.train_x_decoder[0])
            self.train_x_decoder = np.reshape(self.train_x_decoder, (self.train_x_decoder.shape[0], int(self.train_x_decoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
            
            self.valid_x_decoder = np.array(dataX_decoder[self.train_size - self.sliding_decoder: self.train_size + self.valid_size - self.sliding_decoder])
            self.valid_x_decoder = np.reshape(self.valid_x_decoder, (self.valid_x_decoder.shape[0], int(self.valid_x_decoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
            self.test_x_decoder = np.array(dataX_decoder[self.train_size + self.valid_size - self.sliding_decoder:])
            self.test_x_decoder = np.reshape(self.test_x_decoder, (self.test_x_decoder.shape[0], int(self.test_x_decoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))

            # self.train_y_decoder = self.scaled_original_data[0][self.sliding_encoder : self.train_size]
            # self.valid_y_decoder = self.scaled_original_data[0][self.train_size : self.train_size + self.valid_size]
            # self.test_y_decoder = self.scaled_original_data[0][self.train_size + self.valid_size:]
            self.train_y_decoder = []
            self.valid_y_decoder = []
            self.test_y_decoder = []
            if (self.number_out_decoder == 1):
                # print (train_y_decoder_raw.shape)
                # for i in range(len(train_y_decoder_raw)):
                #     train_y_decoderi = []
                #     for j in range(len(train_y_decoder_raw[i])):
                #         train_y_decoderi.append(train_y_decoder_raw[i][j][0])
                #     self.train_y_decoder.append(train_y_decoderi)
                self.train_y_decoder = self.scaled_original_data[0][self.sliding_encoder : self.train_size]
                self.valid_y_decoder = self.scaled_original_data[0][self.train_size : self.train_size + self.valid_size]
                self.test_y_decoder = self.scaled_original_data[0][self.train_size + self.valid_size:]
            else:
                # print (train_y_decoder_raw.shape)
                # train_y_decoder1 = []
                # train_y_decoder2 = []
                # for i in range(len(train_y_decoder_raw)):
                #     train_y_decoderi1 = []
                #     train_y_decoderi2 = []
                #     for j in range(len(train_y_decoder_raw[i])):
                #         train_y_decoderi1.append(train_y_decoder_raw[i][j][0])
                #         train_y_decoderi2.append(train_y_decoder_raw[i][j][1])
                #     train_y_decoder1.append(train_y_decoderi1)
                #     train_y_decoder2.append(train_y_decoderi2)
                    # print (train_y_decoder1)
                    # print (train_y_decoder2)
                train_y_decoder1 = self.scaled_original_data[0][self.sliding_encoder : self.train_size]
                valid_y_decoder1 = self.scaled_original_data[0][self.train_size : self.train_size + self.valid_size]
                test_y_decoder1 = self.scaled_original_data[0][self.train_size + self.valid_size:]

                train_y_decoder2 = self.scaled_original_data[1][self.sliding_encoder : self.train_size]
                valid_y_decoder2 = self.scaled_original_data[1][self.train_size : self.train_size + self.valid_size]
                test_y_decoder2 = self.scaled_original_data[1][self.train_size + self.valid_size:]

                self.train_y_decoder.append(train_y_decoder1)
                self.train_y_decoder.append(train_y_decoder2)

                self.valid_y_decoder.append(valid_y_decoder1)
                self.valid_y_decoder.append(valid_y_decoder2)

                self.test_y_decoder.append(test_y_decoder1)
                self.test_y_decoder.append(test_y_decoder2)
            self.train_y_decoder = np.array(self.train_y_decoder)
            self.valid_y_decoder = np.array(self.valid_y_decoder)
            self.test_y_decoder = np.array(self.test_y_decoder)
            
            self.train_x_inference = np.array(dataX_ext[self.sliding_encoder - self.sliding_inference:self.train_size - self.sliding_inference])
            
            self.train_x_inference = np.reshape(self.train_x_inference, (self.train_x_inference.shape[0], 1, int(self.train_x_inference.shape[1]*len(self.external_feature))))
            print (self.train_x_inference[0])
            self.valid_x_inference = np.array(dataX_ext[self.train_size - self.sliding_inference: self.train_size + self.valid_size - self.sliding_inference])
            self.valid_x_inference = np.reshape(self.valid_x_inference, (self.valid_x_inference.shape[0], 1, int(self.valid_x_inference.shape[1]*len(self.external_feature))))

            self.test_x_inference = np.array(dataX_ext[self.train_size + self.valid_size - self.sliding_inference:])
            self.test_x_inference = np.reshape(self.test_x_inference, (self.test_x_inference.shape[0], 1, int(self.test_x_inference.shape[1]*len(self.external_feature))))

            self.train_y_inference = self.scaled_prediction_data[0][self.sliding_encoder: self.train_size]
            self.valid_y_inference = self.scaled_prediction_data[0][self.train_size: self.train_size + self.valid_size]
            self.test_y_inference = self.prediction_data[0][self.train_size + self.valid_size:]
            # lol199
            return self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.min_prediction_data, self.max_prediction_data, self.train_x_inference, self.valid_x_inference, self.test_x_inference, self.train_y_inference, self.valid_y_inference, self.test_y_inference
        else:
            print ('==================true======================')
            print (self.original_data )
            print (self.prediction_data )
            print (self.external_feature)
            print (self.train_size )
            print (self.valid_size)
            print (self.sliding_encoder)
            print (self.sliding_decoder)
            print (self.sliding_inference)
            print (self.input_dim)
            print (self.number_out_decoder)
            # lol
            self.scaled_original_data, self.min_original_arr, self.max_original_arr = self.scale_timeseries(self.original_data)
            self.scaled_prediction_data, self.min_prediction_data, self.max_prediction_data = self.scale_timeseries(self.prediction_data)
            print ("==============check scale data============")
            # self.scaled_original_data = np.asarray(self.scaled_original_data)
            self.original_data = np.asarray(self.original_data)
            # print (self.scaled_original_data.shape)
            print (self.original_data.shape)
            # lol121
            # self.scaled_external_feature, self.min_ext, self.max_ext = self.scale_timeseries(self.external_feature)
            self.multivariate_original_timeseries = self.create_multivariate_timeseries(self.original_data)
            # check create multivariate timeseries for encoder,decoder
            print ("=================check create timeseries for encoder-decoder====================")
            print (self.multivariate_original_timeseries)
            # print (self.scaled_external_feature)
            print ('================scaled external feature===============')
            self.ext_timeseries = self.create_multivariate_timeseries(self.external_feature)
            # print (self.multivariate_timeseries)
            print ('=============check external timeseries===============')
            print (self.ext_timeseries)
            # lol135
            dataX_encoder = self.create_x(self.multivariate_original_timeseries, self.sliding_encoder)
            dataX_decoder = self.create_x(self.multivariate_original_timeseries, self.sliding_decoder)
            dataX_ext = self.create_x(self.ext_timeseries, self.sliding_inference)
            dataX_decoder = np.asarray(dataX_decoder)
            print (dataX_decoder.shape)
            print (dataX_decoder[0])
            print (dataX_encoder[1])
            # lol
            print ("==============check train x==============")
            self.train_x_encoder = dataX_encoder[0:self.train_size - self.sliding_encoder]
            print (self.train_x_encoder[0])
            # lol131
            self.train_x_encoder = np.array(self.train_x_encoder)
            
            # lol134
            self.train_x_encoder = np.reshape(self.train_x_encoder, (self.train_x_encoder.shape[0], int(self.train_x_encoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
            self.valid_x_encoder = np.array(dataX_encoder[self.train_size - self.sliding_encoder: self.train_size + self.valid_size - self.sliding_encoder])
            self.valid_x_encoder = np.reshape(self.valid_x_encoder, (self.valid_x_encoder.shape[0], int(self.valid_x_encoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        
            self.test_x_encoder = np.array(dataX_encoder[self.train_size + self.valid_size - self.sliding_encoder:])
            self.test_x_encoder = np.reshape(self.test_x_encoder, (self.test_x_encoder.shape[0], int(self.test_x_encoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
            
            self.train_x_decoder = np.array(dataX_decoder[self.sliding_encoder - self.sliding_decoder:self.train_size - self.sliding_decoder])
            print (self.train_x_decoder[0])
            self.train_x_decoder = np.reshape(self.train_x_decoder, (self.train_x_decoder.shape[0], int(self.train_x_decoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
            
            self.valid_x_decoder = np.array(dataX_decoder[self.train_size - self.sliding_decoder: self.train_size + self.valid_size - self.sliding_decoder])
            self.valid_x_decoder = np.reshape(self.valid_x_decoder, (self.valid_x_decoder.shape[0], int(self.valid_x_decoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
            self.test_x_decoder = np.array(dataX_decoder[self.train_size + self.valid_size - self.sliding_decoder:])
            self.test_x_decoder = np.reshape(self.test_x_decoder, (self.test_x_decoder.shape[0], int(self.test_x_decoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
            self.train_x_inference = np.array(dataX_ext[self.sliding_encoder - self.sliding_inference:self.train_size - self.sliding_inference])
            
            self.train_x_inference = np.reshape(self.train_x_inference, (self.train_x_inference.shape[0], 1, int(self.train_x_inference.shape[1]*len(self.external_feature))))
            print (self.train_x_inference[0])
            self.valid_x_inference = np.array(dataX_ext[self.train_size - self.sliding_inference: self.train_size + self.valid_size - self.sliding_inference])
            self.valid_x_inference = np.reshape(self.valid_x_inference, (self.valid_x_inference.shape[0], 1, int(self.valid_x_inference.shape[1]*len(self.external_feature))))

            self.test_x_inference = np.array(dataX_ext[self.train_size + self.valid_size - self.sliding_inference:])
            self.test_x_inference = np.reshape(self.test_x_inference, (self.test_x_inference.shape[0], 1, int(self.test_x_inference.shape[1]*len(self.external_feature))))

            self.train_y_inference = self.scaled_prediction_data[0][self.sliding_encoder: self.train_size]
            self.valid_y_inference = self.scaled_prediction_data[0][self.train_size: self.train_size + self.valid_size]
            self.test_y_inference = self.prediction_data[0][self.train_size + self.valid_size:]

            print (self.train_x_encoder.shape)
            print (self.train_x_decoder.shape)
            print (self.train_x_inference.shape)
            normalize_train_x_encoder = []
            min_train_x_encoder = []
            max_train_x_encoder = []
            normalize_train_x_decoder = self.train_x_decoder
            for i in range(len(self.train_x_encoder)):
                arr = []
                for t in range(len(self.train_x_encoder[i][0])):
                    arr.append([])
                for j in range(len(self.train_x_encoder[i])):
                    for k in range(len(self.train_x_encoder[i][j])):
                        arr[k].append(self.train_x_encoder[i][j][k])
                scaled_arr, min_arr, max_arr = self.scale_timeseries(arr)
                scaled_arr = np.asarray(scaled_arr)
                scaled_arr = np.transpose(scaled_arr)
                min_train_x_encoder.append(min_arr)
                max_train_x_encoder.append(max_arr)
                for j in range(len(self.train_x_decoder[i])):
                    for k in range(len(self.train_x_decoder[i][j])):
                        normalize_train_x_decoder[i][j][k] = (self.train_x_decoder[i][j][k] - min_arr[k]) / (max_arr[k] - min_arr[k])
                normalize_train_x_encoder.append(scaled_arr)
            normalize_train_x_decoder = np.asarray(normalize_train_x_decoder)
            normalize_train_x_encoder = np.asarray(normalize_train_x_encoder)
            print (normalize_train_x_encoder.shape)
            print (normalize_train_x_decoder.shape)

            normalize_valid_x_encoder = []
            min_valid_x_encoder = []
            max_valid_x_encoder = []
            normalize_valid_x_decoder = self.valid_x_decoder
            for i in range(len(self.valid_x_encoder)):
                arr = []
                for t in range(len(self.valid_x_encoder[i][0])):
                    arr.append([])
                for j in range(len(self.valid_x_encoder[i])):
                    for k in range(len(self.valid_x_encoder[i][j])):
                        arr[k].append(self.valid_x_encoder[i][j][k])
                # print (arr)
                scaled_arr, min_arr, max_arr = self.scale_timeseries(arr)
                scaled_arr = np.asarray(scaled_arr)
                scaled_arr = np.transpose(scaled_arr)
                min_valid_x_encoder.append(min_arr)
                max_valid_x_encoder.append(max_arr)
                for j in range(len(self.valid_x_decoder[i])):
                    for k in range(len(self.valid_x_decoder[i][j])):
                        normalize_valid_x_decoder[i][j][k] = (self.valid_x_decoder[i][j][k] - min_arr[k]) / (max_arr[k] - min_arr[k])
                normalize_valid_x_encoder.append(scaled_arr)
            normalize_valid_x_decoder = np.asarray(normalize_valid_x_decoder)
            normalize_valid_x_encoder = np.asarray(normalize_valid_x_encoder)
            print (normalize_valid_x_encoder.shape)
            print (normalize_valid_x_decoder.shape)

            normalize_test_x_encoder = []
            min_test_x_encoder = []
            max_test_x_encoder = []
            normalize_test_x_decoder = self.test_x_decoder
            for i in range(len(self.test_x_encoder)):
                arr = []
                for t in range(len(self.test_x_encoder[i][0])):
                    arr.append([])
                for j in range(len(self.test_x_encoder[i])):
                    for k in range(len(self.test_x_encoder[i][j])):
                        arr[k].append(self.test_x_encoder[i][j][k])
                scaled_arr, min_arr, max_arr = self.scale_timeseries(arr)
                scaled_arr = np.asarray(scaled_arr)
                scaled_arr = np.transpose(scaled_arr)
                min_test_x_encoder.append(min_arr)
                max_test_x_encoder.append(max_arr)
                for j in range(len(self.test_x_decoder[i])):
                    for k in range(len(self.test_x_decoder[i][j])):
                        normalize_test_x_decoder[i][j][k] = (self.test_x_decoder[i][j][k] - min_arr[k]) / (max_arr[k] - min_arr[k])
                normalize_test_x_encoder.append(scaled_arr)
            normalize_test_x_decoder = np.asarray(normalize_test_x_decoder)
            normalize_test_x_encoder = np.asarray(normalize_test_x_encoder)
            print (normalize_test_x_encoder.shape)
            print (normalize_test_x_decoder.shape)
            self.train_y_decoder = []
            self.valid_y_decoder = []
            self.test_y_decoder = []
            print (min_train_x_encoder[0])
            print (max_train_x_encoder[0])
            if (self.number_out_decoder == 1):
                
                self.train_y_decoder = self.scaled_original_data[0][self.sliding_encoder : self.train_size]
                # print (self.train_y_decoder[0])
                # for i in range(len(self.train_y_decoder)):
                #     self.train_y_decoder[i][0] = (self.train_y_decoder[i][0] - min_train_x_encoder[i][0])/(max_train_x_encoder[i][0]-min_train_x_encoder[i][0])
                # self.train_y_decoder = np.array(self.train_y_decoder)
                # print (self.train_y_decoder[0])
                self.valid_y_decoder = self.scaled_original_data[0][self.train_size : self.train_size + self.valid_size]
                self.test_y_decoder = self.scaled_original_data[0][self.train_size + self.valid_size:]
            else:
                train_y_decoder1 = self.scaled_original_data[0][self.sliding_encoder : self.train_size]
                valid_y_decoder1 = self.scaled_original_data[0][self.train_size : self.train_size + self.valid_size]
                test_y_decoder1 = self.scaled_original_data[0][self.train_size + self.valid_size:]

                train_y_decoder2 = self.scaled_original_data[1][self.sliding_encoder : self.train_size]
                valid_y_decoder2 = self.scaled_original_data[1][self.train_size : self.train_size + self.valid_size]
                test_y_decoder2 = self.scaled_original_data[1][self.train_size + self.valid_size:]

                self.train_y_decoder.append(train_y_decoder1)
                self.train_y_decoder.append(train_y_decoder2)

                self.valid_y_decoder.append(valid_y_decoder1)
                self.valid_y_decoder.append(valid_y_decoder2)

                self.test_y_decoder.append(test_y_decoder1)
                self.test_y_decoder.append(test_y_decoder2)
            self.train_y_decoder = np.array(self.train_y_decoder)
            self.valid_y_decoder = np.array(self.valid_y_decoder)
            self.test_y_decoder = np.array(self.test_y_decoder)
            
            
            # lol199
            # return self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.min_prediction_data, self.max_prediction_data, self.train_x_inference, self.valid_x_inference, self.test_x_inference, self.train_y_inference, self.valid_y_inference, self.test_y_inference
            return normalize_train_x_encoder, normalize_valid_x_encoder, normalize_test_x_encoder, normalize_train_x_decoder, normalize_valid_x_decoder, normalize_test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.min_prediction_data, self.max_prediction_data, self.train_x_inference, self.valid_x_inference, self.test_x_inference, self.train_y_inference, self.valid_y_inference, self.test_y_inference
    def scaling_data(self, X):
        minX = np.amin(X)
        maxX = np.amax(X)
        mean = np.mean(X)
        scale = (X-minX)/(maxX-minX)
        return scale, minX, maxX
    def scale_timeseries(self, X):
        scale = []
        min_arr = []
        max_arr = []
        for i in range(len(X)):
            scalei, mini, maxi = self.scaling_data(X[i])
            scale.append(scalei)
            min_arr.append(mini)
            max_arr.append(maxi)
        return scale, min_arr, max_arr
    """
    This function concatenate multi timeseries into a multivariate timeseries
    """
    def create_multivariate_timeseries(self, X):
        # print ('===============<>===============')
        # print (X[0])
        # print (len(X))
        # print (X[1])
        # data = []
        if(len(X)>1):
            data = np.concatenate((X[0],X[1]), axis=1)
            if(len(X) > 2):
                for i in range(2,len(X),1):
                    # print (i)
                    data = np.column_stack((data,X[i]))
        else:
            data = []
            for i in range(len(X[0])):
                # print(X[0][i])
                data.append(X[0][i])
            data = np.array(data)
        return data
    """
    This function create samples with sliding and timseries
    example: timeseries : [1,2,3,4]
             sliding = 2
    output: [[1,2],[2,3],[3,4]]
    """
    def create_x(self, timeseries, sliding):
        # print (len(timeseries))
        dataX = []
        for i in range(len(timeseries)-sliding):
            datai = []
            for j in range(sliding):
                datai.append(timeseries[i+j])
            dataX.append(datai)
        return dataX
class MultivariateTimeseriesBNNUber:
    def __init__(self, original_data = None, external_feature = None, train_size = None, valid_size = None, 
        sliding_encoder = None, sliding_decoder = None, sliding_inference = None ,input_dim = None, number_out_decoder = 1):
        self.original_data = original_data
        self.external_feature = external_feature
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder
        self.sliding_inference = sliding_inference
        self.input_dim = input_dim 
        self.number_out_decoder = number_out_decoder
    def prepare_data(self):
        # check input
        # print ("==============check input=============")
        # print (self.original_data[0])
        # print (self.external_feature[0])
        # lol117
        self.scaled_data, self.min_arr, self.max_arr = self.scale_timeseries(self.original_data)
        print ("==============check scale data============")
        print (self.scaled_data)
        print (self.min_arr)
        print (self.max_arr)
        # lol121
        self.scaled_external_feature, self.min_ext, self.max_ext = self.scale_timeseries(self.external_feature)
        self.multivariate_timeseries = self.create_multivariate_timeseries(self.scaled_data)
        # check create multivariate timeseries for encoder,decoder
        print ("=================check create timeseries for encoder-decoder====================")
        print (self.multivariate_timeseries)
        # print (self.scaled_external_feature)
        print ('================scaled external feature===============')
        self.ext_timeseries = self.create_multivariate_timeseries(self.scaled_external_feature)
        # print (self.multivariate_timeseries)
        print ('=============check external timeseries===============')
        print (self.ext_timeseries)
        # lol135
        dataX_encoder = self.create_x(self.multivariate_timeseries, self.sliding_encoder)
        dataX_decoder = self.create_x(self.multivariate_timeseries, self.sliding_decoder)
        dataX_ext = self.create_x(self.ext_timeseries, self.sliding_inference)
        # check create input-output pair for training
        print ('min_arr')
        print (self.min_arr)
        print ('max_arr')
        print (self.max_arr)
        print ('dataX_encoder')
        print (dataX_encoder[0])
        print ('dataX_decoder')
        print (dataX_decoder[0])
        print ('dataX_ext')
        print (dataX_ext[0])
        # lol150
        # print (dataX_encoder[0])
        # print (dataX_encoder[1])
        # # lol124
        # print (dataX_encoder[1])
        # print (dataX_decoder[0])
        # print (dataX_decoder[1])
        print ("==============check train x==============")
        self.train_x_encoder = dataX_encoder[0:self.train_size - self.sliding_encoder]
        print (self.train_x_encoder[0])
        # lol131
        self.train_x_encoder = np.array(self.train_x_encoder)
        
        # lol134
        self.train_x_encoder = np.reshape(self.train_x_encoder, (self.train_x_encoder.shape[0], int(self.train_x_encoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        
        self.valid_x_encoder = np.array(dataX_encoder[self.train_size - self.sliding_encoder: self.train_size + self.valid_size - self.sliding_encoder])
        self.valid_x_encoder = np.reshape(self.valid_x_encoder, (self.valid_x_encoder.shape[0], int(self.valid_x_encoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
       
        self.test_x_encoder = np.array(dataX_encoder[self.train_size + self.valid_size - self.sliding_encoder:])
        self.test_x_encoder = np.reshape(self.test_x_encoder, (self.test_x_encoder.shape[0], int(self.test_x_encoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        
        self.train_x_decoder = np.array(dataX_decoder[self.sliding_encoder - self.sliding_decoder:self.train_size - self.sliding_decoder])
        print (self.train_x_decoder[0])
        self.train_x_decoder = np.reshape(self.train_x_decoder, (self.train_x_decoder.shape[0], int(self.train_x_decoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        
        self.valid_x_decoder = np.array(dataX_decoder[self.train_size - self.sliding_decoder: self.train_size + self.valid_size - self.sliding_decoder])
        self.valid_x_decoder = np.reshape(self.valid_x_decoder, (self.valid_x_decoder.shape[0], int(self.valid_x_decoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        self.test_x_decoder = np.array(dataX_decoder[self.train_size + self.valid_size - self.sliding_decoder:])
        self.test_x_decoder = np.reshape(self.test_x_decoder, (self.test_x_decoder.shape[0], int(self.test_x_decoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        train_y_decoder_raw = np.array(dataX_decoder[self.sliding_encoder - self.sliding_decoder+1:self.train_size - self.sliding_decoder+1])
        # self.train_y_decoder = self.scaled_data[0][self.sliding_encoder : self.train_size]
        self.train_y_decoder = []
        if (self.number_out_decoder == 1):
            print (train_y_decoder_raw.shape)
            for i in range(len(train_y_decoder_raw)):
                train_y_decoderi = []
                for j in range(len(train_y_decoder_raw[i])):
                    train_y_decoderi.append(train_y_decoder_raw[i][j][0])
                self.train_y_decoder.append(train_y_decoderi)
        else:
            print (train_y_decoder_raw.shape)
            train_y_decoder1 = []
            train_y_decoder2 = []
            for i in range(len(train_y_decoder_raw)):
                train_y_decoderi1 = []
                train_y_decoderi2 = []
                for j in range(len(train_y_decoder_raw[i])):
                    train_y_decoderi1.append(train_y_decoder_raw[i][j][0])
                    train_y_decoderi2.append(train_y_decoder_raw[i][j][1])
                train_y_decoder1.append(train_y_decoderi1)
                train_y_decoder2.append(train_y_decoderi2)
                # print (train_y_decoder1)
                # print (train_y_decoder2)
            self.train_y_decoder.append(train_y_decoder1)
            self.train_y_decoder.append(train_y_decoder2)
        self.train_y_decoder = np.array(self.train_y_decoder)

        valid_y_decoder_raw = np.array(dataX_decoder[self.train_size : self.train_size + self.valid_size])
        # self.train_y_decoder = self.scaled_data[0][self.sliding_encoder : self.train_size]
        self.valid_y_decoder = []
        if (self.number_out_decoder == 1):
            print (valid_y_decoder_raw.shape)
            for i in range(len(valid_y_decoder_raw)):
                valid_y_decoderi = []
                for j in range(len(valid_y_decoder_raw[i])):
                    valid_y_decoderi.append(valid_y_decoder_raw[i][j][0])
                self.valid_y_decoder.append(valid_y_decoderi)
        else:
            print (valid_y_decoder_raw.shape)
            valid_y_decoder1 = []
            valid_y_decoder2 = []
            for i in range(len(valid_y_decoder_raw)):
                valid_y_decoderi1 = []
                valid_y_decoderi2 = []
                for j in range(len(valid_y_decoder_raw[i])):
                    valid_y_decoderi1.append(valid_y_decoder_raw[i][j][0])
                    valid_y_decoderi2.append(valid_y_decoder_raw[i][j][1])
                valid_y_decoder1.append(valid_y_decoderi1)
                valid_y_decoder2.append(valid_y_decoderi2)
                # print (train_y_decoder1)
                # print (train_y_decoder2)
            self.valid_y_decoder.append(valid_y_decoder1)
            self.valid_y_decoder.append(valid_y_decoder2)
        self.valid_y_decoder = np.array(self.valid_y_decoder)
        # self.valid_y_decoder = self.scaled_data[0][self.train_size : self.train_size + self.valid_size]
        self.test_y_decoder = self.scaled_data[0][self.train_size + self.valid_size:]
        
        self.train_x_inference = np.array(dataX_ext[self.sliding_encoder - self.sliding_inference:self.train_size - self.sliding_inference])
        
        self.train_x_inference = np.reshape(self.train_x_inference, (self.train_x_inference.shape[0], 1, int(self.train_x_inference.shape[1]*len(self.external_feature))))
        print (self.train_x_inference[0])
        self.valid_x_inference = np.array(dataX_ext[self.train_size - self.sliding_inference: self.train_size + self.valid_size - self.sliding_inference])
        self.valid_x_inference = np.reshape(self.valid_x_inference, (self.valid_x_inference.shape[0], 1, int(self.valid_x_inference.shape[1]*len(self.external_feature))))

        self.test_x_inference = np.array(dataX_ext[self.train_size + self.valid_size - self.sliding_inference:])
        self.test_x_inference = np.reshape(self.test_x_inference, (self.test_x_inference.shape[0], 1, int(self.test_x_inference.shape[1]*len(self.external_feature))))

        self.train_y_inference = self.scaled_data[0][self.sliding_encoder: self.train_size]
        self.valid_y_inference = self.scaled_data[0][self.train_size: self.train_size + self.valid_size]
        self.test_y_inference = self.original_data[0][self.train_size + self.valid_size:]
        # lol199
        # print ('self.train_x_encoder[0]')
        # print (self.train_x_encoder[0])
        # print ('x decoder')
        # print (self.train_x_decoder[0])
        # print ('x inference')
        # print (self.train_x_inference[0])
        print ('train y decoder')
        print (self.train_y_decoder[0])
        # # print (self.train_y_decoder[1][0])
        # print ('y inference')
        # print (self.train_y_inference[0])
        # lol
        return self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.min_arr, self.max_arr, self.train_x_inference, self.valid_x_inference, self.test_x_inference, self.train_y_inference, self.valid_y_inference, self.test_y_inference
    def scaling_data(self, X):
        minX = np.amin(X)
        maxX = np.amax(X)
        mean = np.mean(X)
        scale = (X-minX)/(maxX-minX)
        return scale, minX, maxX
    def scale_timeseries(self, X):
        scale = []
        min_arr = []
        max_arr = []
        for i in range(len(X)):
            scalei, mini, maxi = self.scaling_data(X[i])
            scale.append(scalei)
            min_arr.append(mini)
            max_arr.append(maxi)
        return scale, min_arr, max_arr
    """
    This function concatenate multi timeseries into a multivariate timeseries
    """
    def create_multivariate_timeseries(self, X):
        # print ('===============<>===============')
        # print (X[0])
        # print (len(X))
        # print (X[1])
        # data = []
        if(len(X)>1):
            data = np.concatenate((X[0],X[1]), axis=1)
            if(len(X) > 2):
                for i in range(2,len(X),1):
                    # print (i)
                    data = np.column_stack((data,X[i]))
        else:
            data = []
            for i in range(len(X[0])):
                # print(X[0][i])
                data.append(X[0][i])
            data = np.array(data)
        return data
    """
    This function create samples with sliding and timseries
    example: timeseries : [1,2,3,4]
             sliding = 2
    output: [[1,2],[2,3],[3,4]]
    """
    def create_x(self, timeseries, sliding):
        # print (len(timeseries))
        dataX = []
        for i in range(len(timeseries)-sliding):
            datai = []
            for j in range(sliding):
                datai.append(timeseries[i+j])
            dataX.append(datai)
        return dataX
class FuzzyMultivariateTimeseriesBNNUber:
    def __init__(self, original_data = None, prediction_data = None, external_feature = None, train_size = None, valid_size = None, 
        sliding_encoder = None, sliding_decoder = None, sliding_inference = None ,input_dim = None, number_out_decoder = 1):
        self.original_data = original_data
        self.prediction_data = prediction_data
        self.external_feature = external_feature
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding_encoder = sliding_encoder
        self.sliding_decoder = sliding_decoder
        self.sliding_inference = sliding_inference
        self.input_dim = input_dim 
        self.number_out_decoder = number_out_decoder
    def prepare_data(self):
        # check input
        # print ("==============check input=============")
        # print (self.original_data[0])
        # print (self.external_feature[0])
        # lol117
        self.scaled_data, self.min_arr, self.max_arr = self.scale_timeseries(self.original_data)
        # print ("==============check scale data============")
        # print (self.scaled_data)
        # print (self.min_arr)
        # print (self.max_arr)
        # lol121
        # print (self.prediction_data)
        # print (self.original_data)
        self.scale_prediction_data, self.min_prediction_data, self.max_prediction_data = self.scale_timeseries(self.prediction_data)
        # print (self.scale_prediction_data)
        # print (self.scaled_data)
        # lol/
        self.scaled_external_feature, self.min_ext, self.max_ext = self.scale_timeseries(self.external_feature)
        self.multivariate_timeseries = self.create_multivariate_timeseries(self.scaled_data)
        # check create multivariate timeseries for encoder,decoder
        # print ("=================check create timeseries for encoder-decoder====================")
        # print (self.multivariate_timeseries)
        # # print (self.scaled_external_feature)
        # print ('================scaled external feature===============')
        self.ext_timeseries = self.create_multivariate_timeseries(self.scaled_external_feature)
        # print (self.multivariate_timeseries)
        # print ('=============check external timeseries===============')
        # print (self.ext_timeseries)
        # lol135
        dataX_encoder = self.create_x(self.multivariate_timeseries, self.sliding_encoder)
        dataX_decoder = self.create_x(self.multivariate_timeseries, self.sliding_decoder)
        dataX_ext = self.create_x(self.ext_timeseries, self.sliding_inference)
        # check create input-output pair for training
        # print ('min_arr')
        # print (self.min_arr)
        # print ('max_arr')
        # print (self.max_arr)
        # print ('dataX_encoder')
        # print (dataX_encoder[0])
        # print ('dataX_decoder')
        # print (dataX_decoder[0])
        # print ('dataX_ext')
        # print (dataX_ext[0])
        # lol150
        # print (dataX_encoder[0])
        # print (dataX_encoder[1])
        # # lol124
        # print (dataX_encoder[1])
        # print (dataX_decoder[0])
        # print (dataX_decoder[1])
        # print ("==============check train x==============")
        self.train_x_encoder = dataX_encoder[0:self.train_size - self.sliding_encoder]
        # print (self.train_x_encoder[0])
        # lol131
        self.train_x_encoder = np.array(self.train_x_encoder)
        
        # lol134
        self.train_x_encoder = np.reshape(self.train_x_encoder, (self.train_x_encoder.shape[0], int(self.train_x_encoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        
        self.valid_x_encoder = np.array(dataX_encoder[self.train_size - self.sliding_encoder: self.train_size + self.valid_size - self.sliding_encoder])
        self.valid_x_encoder = np.reshape(self.valid_x_encoder, (self.valid_x_encoder.shape[0], int(self.valid_x_encoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
       
        self.test_x_encoder = np.array(dataX_encoder[self.train_size + self.valid_size - self.sliding_encoder:])
        self.test_x_encoder = np.reshape(self.test_x_encoder, (self.test_x_encoder.shape[0], int(self.test_x_encoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        
        self.train_x_decoder = np.array(dataX_decoder[self.sliding_encoder - self.sliding_decoder:self.train_size - self.sliding_decoder])
        # print (self.train_x_decoder[0])
        self.train_x_decoder = np.reshape(self.train_x_decoder, (self.train_x_decoder.shape[0], int(self.train_x_decoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        
        self.valid_x_decoder = np.array(dataX_decoder[self.train_size - self.sliding_decoder: self.train_size + self.valid_size - self.sliding_decoder])
        self.valid_x_decoder = np.reshape(self.valid_x_decoder, (self.valid_x_decoder.shape[0], int(self.valid_x_decoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        self.test_x_decoder = np.array(dataX_decoder[self.train_size + self.valid_size - self.sliding_decoder:])
        self.test_x_decoder = np.reshape(self.test_x_decoder, (self.test_x_decoder.shape[0], int(self.test_x_decoder.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        train_y_decoder_raw = np.array(dataX_decoder[self.sliding_encoder - self.sliding_decoder+1:self.train_size - self.sliding_decoder+1])
        # self.train_y_decoder = self.scaled_data[0][self.sliding_encoder : self.train_size]
        self.train_y_decoder = []
        if (self.number_out_decoder == 1):
            # print (train_y_decoder_raw.shape)
            for i in range(len(train_y_decoder_raw)):
                train_y_decoderi = []
                for j in range(len(train_y_decoder_raw[i])):
                    train_y_decoderi.append(train_y_decoder_raw[i][j][0])
                self.train_y_decoder.append(train_y_decoderi)
        else:
            # print (train_y_decoder_raw.shape)
            train_y_decoder1 = []
            train_y_decoder2 = []
            for i in range(len(train_y_decoder_raw)):
                train_y_decoderi1 = []
                train_y_decoderi2 = []
                for j in range(len(train_y_decoder_raw[i])):
                    train_y_decoderi1.append(train_y_decoder_raw[i][j][0])
                    train_y_decoderi2.append(train_y_decoder_raw[i][j][1])
                train_y_decoder1.append(train_y_decoderi1)
                train_y_decoder2.append(train_y_decoderi2)
                # print (train_y_decoder1)
                # print (train_y_decoder2)
            self.train_y_decoder.append(train_y_decoder1)
            self.train_y_decoder.append(train_y_decoder2)
        self.train_y_decoder = np.array(self.train_y_decoder)

        valid_y_decoder_raw = np.array(dataX_decoder[self.train_size : self.train_size + self.valid_size])
        # self.train_y_decoder = self.scaled_data[0][self.sliding_encoder : self.train_size]
        self.valid_y_decoder = []
        if (self.number_out_decoder == 1):
            # print (valid_y_decoder_raw.shape)
            for i in range(len(valid_y_decoder_raw)):
                valid_y_decoderi = []
                for j in range(len(valid_y_decoder_raw[i])):
                    valid_y_decoderi.append(valid_y_decoder_raw[i][j][0])
                self.valid_y_decoder.append(valid_y_decoderi)
        else:
            # print (valid_y_decoder_raw.shape)
            valid_y_decoder1 = []
            valid_y_decoder2 = []
            for i in range(len(valid_y_decoder_raw)):
                valid_y_decoderi1 = []
                valid_y_decoderi2 = []
                for j in range(len(valid_y_decoder_raw[i])):
                    valid_y_decoderi1.append(valid_y_decoder_raw[i][j][0])
                    valid_y_decoderi2.append(valid_y_decoder_raw[i][j][1])
                valid_y_decoder1.append(valid_y_decoderi1)
                valid_y_decoder2.append(valid_y_decoderi2)
                # print (train_y_decoder1)
                # print (train_y_decoder2)
            self.valid_y_decoder.append(valid_y_decoder1)
            self.valid_y_decoder.append(valid_y_decoder2)
        self.valid_y_decoder = np.array(self.valid_y_decoder)
        # self.valid_y_decoder = self.scaled_data[0][self.train_size : self.train_size + self.valid_size]
        self.test_y_decoder = self.scaled_data[0][self.train_size + self.valid_size:]
        
        self.train_x_inference = np.array(dataX_ext[self.sliding_encoder - self.sliding_inference:self.train_size - self.sliding_inference])
        
        self.train_x_inference = np.reshape(self.train_x_inference, (self.train_x_inference.shape[0], 1, int(self.train_x_inference.shape[1]*len(self.external_feature))))
        print (self.train_x_inference[0])
        self.valid_x_inference = np.array(dataX_ext[self.train_size - self.sliding_inference: self.train_size + self.valid_size - self.sliding_inference])
        self.valid_x_inference = np.reshape(self.valid_x_inference, (self.valid_x_inference.shape[0], 1, int(self.valid_x_inference.shape[1]*len(self.external_feature))))

        self.test_x_inference = np.array(dataX_ext[self.train_size + self.valid_size - self.sliding_inference:])
        self.test_x_inference = np.reshape(self.test_x_inference, (self.test_x_inference.shape[0], 1, int(self.test_x_inference.shape[1]*len(self.external_feature))))

        self.train_y_inference = self.scale_prediction_data[0][self.sliding_encoder: self.train_size]
        self.valid_y_inference = self.scale_prediction_data[0][self.train_size: self.train_size + self.valid_size]
        self.test_y_inference = self.prediction_data[0][self.train_size + self.valid_size:]
        # lol199
        # print ('self.train_x_encoder[0]')
        # print (self.train_x_encoder[0])
        # print ('x decoder')
        # print (self.train_x_decoder[0])
        # print ('x inference')
        # print (self.train_x_inference[0])
        # print ('train y decoder')
        # print (self.train_y_decoder[0])
        # # print (self.train_y_decoder[1][0])
        # print ('y inference')
        # print (self.train_y_inference[0])
        # lol
        return self.train_x_encoder, self.valid_x_encoder, self.test_x_encoder, self.train_x_decoder, self.valid_x_decoder, self.test_x_decoder, self.train_y_decoder, self.valid_y_decoder, self.test_y_decoder, self.min_prediction_data, self.max_prediction_data, self.train_x_inference, self.valid_x_inference, self.test_x_inference, self.train_y_inference, self.valid_y_inference, self.test_y_inference
    def scaling_data(self, X):
        minX = np.amin(X)
        maxX = np.amax(X)
        mean = np.mean(X)
        scale = (X-minX)/(maxX-minX)
        return scale, minX, maxX
    def scale_timeseries(self, X):
        scale = []
        min_arr = []
        max_arr = []
        for i in range(len(X)):
            scalei, mini, maxi = self.scaling_data(X[i])
            scale.append(scalei)
            min_arr.append(mini)
            max_arr.append(maxi)
        return scale, min_arr, max_arr
    """
    This function concatenate multi timeseries into a multivariate timeseries
    """
    def create_multivariate_timeseries(self, X):
        # print ('===============<>===============')
        # print (X[0])
        # print (len(X))
        # print (X[1])
        # data = []
        if(len(X)>1):
            data = np.concatenate((X[0],X[1]), axis=1)
            if(len(X) > 2):
                for i in range(2,len(X),1):
                    # print (i)
                    data = np.column_stack((data,X[i]))
        else:
            data = []
            for i in range(len(X[0])):
                # print(X[0][i])
                data.append(X[0][i])
            data = np.array(data)
        return data
    """
    This function create samples with sliding and timseries
    example: timeseries : [1,2,3,4]
             sliding = 2
    output: [[1,2],[2,3],[3,4]]
    """
    def create_x(self, timeseries, sliding):
        # print (len(timeseries))
        dataX = []
        for i in range(len(timeseries)-sliding):
            datai = []
            for j in range(sliding):
                datai.append(timeseries[i+j])
            dataX.append(datai)
        return dataX

class LSTM:
    def __init__(self, original_data = None, prediction_data = None, train_size = None, valid_size = None, 
        sliding = None,input_dim = None):
        self.original_data = original_data
        self.prediction_data = prediction_data
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding = sliding
        self.input_dim = input_dim
    def prepare_data(self):
        self.scaled_data, self.min_arr, self.max_arr = self.scale_timeseries(self.original_data)
        self.scale_prediction_data, self.min_prediction_data, self.max_prediction_data = self.scale_timeseries(self.prediction_data)

        self.multivariate_timeseries = self.create_multivariate_timeseries(self.scaled_data)
        
        dataX = self.create_x(self.multivariate_timeseries, self.sliding)

        self.train_x = dataX[0:self.train_size - self.sliding]
        self.train_x = np.array(self.train_x)
        self.train_x = np.reshape(self.train_x, (self.train_x.shape[0], int(self.train_x.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        
        self.valid_x = np.array(dataX[self.train_size - self.sliding: self.train_size + self.valid_size - self.sliding])
        self.valid_x = np.reshape(self.valid_x, (self.valid_x.shape[0], int(self.valid_x.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
       
        self.test_x = np.array(dataX[self.train_size + self.valid_size - self.sliding:])
        self.test_x = np.reshape(self.test_x, (self.test_x.shape[0], int(self.test_x.shape[1]*len(self.original_data)/self.input_dim), self.input_dim))
        
        self.train_y = self.scale_prediction_data[0][self.sliding: self.train_size]
        self.valid_y = self.scale_prediction_data[0][self.train_size: self.train_size + self.valid_size]
        self.test_y = self.prediction_data[0][self.train_size + self.valid_size:]
        self.train_y = np.asarray(self.train_y)
        self.valid_y = np.asarray(self.valid_y)
        self.test_y = np.asarray(self.test_y)

        return self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y, self.min_prediction_data, self.max_prediction_data
    def scaling_data(self, X):
        minX = np.amin(X)
        maxX = np.amax(X)
        mean = np.mean(X)
        scale = (X-minX)/(maxX-minX)
        return scale, minX, maxX
    def scale_timeseries(self, X):
        scale = []
        min_arr = []
        max_arr = []
        for i in range(len(X)):
            scalei, mini, maxi = self.scaling_data(X[i])
            scale.append(scalei)
            min_arr.append(mini)
            max_arr.append(maxi)
        return scale, min_arr, max_arr
    """
    This function concatenate multi timeseries into a multivariate timeseries
    """
    def create_multivariate_timeseries(self, X):
        # print ('===============<>===============')
        # print (X[0])
        # print (len(X))
        # print (X[1])
        # data = []
        if(len(X)>1):
            data = np.concatenate((X[0],X[1]), axis=1)
            if(len(X) > 2):
                for i in range(2,len(X),1):
                    # print (i)
                    data = np.column_stack((data,X[i]))
        else:
            data = []
            for i in range(len(X[0])):
                # print(X[0][i])
                data.append(X[0][i])
            data = np.array(data)
        return data
    """
    This function create samples with sliding and timseries
    example: timeseries : [1,2,3,4]
             sliding = 2
    output: [[1,2],[2,3],[3,4]]
    """
    def create_x(self, timeseries, sliding):
        # print (len(timeseries))
        dataX = []
        for i in range(len(timeseries)-sliding):
            datai = []
            for j in range(sliding):
                datai.append(timeseries[i+j])
            dataX.append(datai)
        return dataX
        