import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd 

class Timeseries:
    def _init_(original_data = None, train_size = None, 
    valid_size = None, sliding = None):
        self.original_data = original_data
        self.train_size = train_size
        self.valid_size = valid_size
        self.sliding = sliding
    def prepare_data(self):
        dataX = []
        for i in range(len(self.original_data)-self.sliding):
            datai = []
            for j in range(sliding):
                datai.append(self.original_data[i+j])
            dataX.append(datai)
        self.train_x = dataX[0:self.train_size]
        self.valid_x = dataX[self.train_size: self.train_size + self.valid_size]
        self.test_x = dataX[self.train_size+ self.valid_size:]
        self.train_y = self.original_data[self.sliding:self.sliding+self.train_size]
        self.valid_y = self.original_data[self.train_size+self.sliding:self.train_size+self.sliding+self.valid_size]
        self.test_y = self.original_data[self.train_size+self.sliding+self.valid_size:]
        return self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y