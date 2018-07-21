
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 
from pandas import read_csv
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
colnames = ['cpu','mem','disk_io_time','disk_space'] 

realFile = ['../data/google_trace_timeseries/data_resource_usage_10Minutes_6176858948.csv']


Real_df = read_csv('%s'%(realFile[0]), header=None, index_col=False, names=colnames, usecols=[3,4,9,10], engine='python')
Pred_TestInverse_df = read_csv('result.csv', header=None, index_col=False, engine='python')

# error_df = read_csv('error_sliding=%s_batchsize=%s_optimize=%s.csv'%(sliding, batch_size,optimize), header=None, index_col=False, engine='python')

RealDataset = Real_df['cpu'].values

train_size = int(len(RealDataset)*0.6)
valid_size = int(len(RealDataset)*0.2)
real_data = RealDataset[train_size + valid_size:]
# test_size = 200
print RealDataset
Pred_TestInverse = Pred_TestInverse_df.values

# testScoreMAE = mean_absolute_error(Pred_TestInverse, realTestData)
# print 'test score', testScoreMAE
ax = plt.subplot()
ax.plot(real_data,label="Actual")
ax.plot(Pred_TestInverse,label="Prediction")
# ax.plrot(TestPred,label="Test")
plt.xlabel("TimeStamp",fontsize=15)
plt.ylabel("CPU",fontsize=15)
plt.title("Encoder-decoder")
ax.text(0,0, 'MAE = 0.39611658', style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':8})
plt.legend()
plt.savefig('result_cpu.pdf')
plt.show()

