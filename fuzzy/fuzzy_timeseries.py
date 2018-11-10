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

class Fuzzification():
    def __init__(self, num_interval = None):
        self.num_interval = num_interval
    def fuzzify(self,timeseries):
        timeseries = np.array(timeseries)
        min_value = min(timeseries)[0]
        max_value = max(timeseries)[0]
        # print (min_value)
        # print (round(max_value))
        u = [min_value, max_value]
        print (u)
        
        self.interval = (u[1] - u[0]) / self.num_interval
        print (self.interval)
        # print (self.number_of_interval)
        arr = []
        for i in range(len(timeseries)):
            # print (self.timeseries[i][0])
            # print ((self.timeseries[i][0] - u[0])/self.interval)
            arr.append(round((timeseries[i][0] - u[0])/self.interval))
        # print (arr)
        return arr