import matplotlib.pyplot as plt
import pandas as pd 
from pandas import read_csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from Graph import *
# colnames=['time_stamp','numberOfTaskIndex','numberOfMachineId','meanCPUUsage','CMU','AssignMem','unmapped_cache_usage','page_cache_usage', 'max_mem_usage','mean_diskIO_time','mean_local_disk_space','max_cpu_usage', 'max_disk_io_time', 'cpi', 'mai','sampled_cpu_usage']
# df = read_csv('/home/nguyen/learnRNNs/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
# colnames = ['cpu','mem','disk_io_time','disk_space'] 

# colnames = ['mem'] 
# # batch_size_array = [8,16,32,64,128]
# realFile = ['results/actualResult.csv']
link = './data/google_trace_timeseries/data_resource_usage_5Minutes_6176858948.csv'
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv(link, header=None, index_col=False, names=colnames, usecols=[3,4,9,10], engine='python')
scaler = MinMaxScaler(feature_range=(0, 1))
cpu = df['cpu_rate'].values.reshape(-1,1)
mem = df['mem_usage'].values.reshape(-1,1)
disk_io_time = df['disk_io_time'].values.reshape(-1,1)
disk_space = df['disk_space'].values.reshape(-1,1)
# sliding_widow = [2]
# optimizerArr=['adam', 'SGD']
# for modelName in modelNameArr: 
# 	for sliding in sliding_widow:
		# for batch_size in batch_size_array:
		# 	if sliding==5 and batch_size ==128:
		# 			break
		# 	for optimize in optimizerArr: 
	
# Real_df = read_csv('%s'%(realFile[0]), header=None, index_col=False, names=colnames, engine='python')
# arr = os.listdir('results/mem/5minutes/bnn_multivariate_uber/prediction/')
# print (arr)
# for i, file in enumerate(arr):
i = 0
# file = '18-8-8-8-[16, 4]-1-2-[16]-2-2.csv'
file = '24-2-10-8-32_4-1-2-2-64-1-0.95.csv'

print (str(file))
file_path = 'testANN.csv'
file_path = 'results/multivariate/mem/5minutes/bnn_multivariate_uber_ver2/prediction/'+ str(file)
Pred_df = read_csv(file_path, header=None, index_col=False, engine='python')
file_name = file.split('.')[0]
# error_df = read_csv('results/cpu/5minutes/bnn_multivariate_uber/prediction/' +file, header=None, index_col=False, engine='python')

RealDataset = mem

train_size = int(len(RealDataset)*0.8)
test_size = len(RealDataset) - train_size
print (RealDataset)
Pred = Pred_df.values

# RMSE = error_df.values[0][0]
# MAE = error_df.values[1][0]
# print RMSE

realTestData = RealDataset[train_size:len(RealDataset)]
print (len(Pred))
print (len(realTestData))
x, y = np.random.randn(2, 100)
print (x.shape)
realTestData = realTestData.reshape(realTestData.shape[0])
Pred = Pred.reshape(Pred.shape[0])
print (realTestData.shape)
# lol
file_path = '/'
# file_path = 'results/fuzzy/multivariate/mem/5minutes/bnn_multivariate_uber/plot/'
# draw_predict(i,realTestData, Pred,file_name,file_path)
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
ax1.xcorr(realTestData, Pred, usevlines=True, maxlags=50, normed=True, lw=2)
ax1.grid(True)
ax1.axhline(0, color='black', lw=2)

ax2.acorr(realTestData, usevlines=True, normed=True, maxlags=50, lw=2)
ax2.grid(True)
ax2.axhline(0, color='black', lw=2)
plt.show()