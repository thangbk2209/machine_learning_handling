import matplotlib as mpl
mpl.use('Agg')
import math
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression as LR
import numpy as np
import pandas as pd 

from fuzzy.fuzzy_timeseries import Fuzzification
from collections import Counter
# Counter(array1.most_common(1))
# print math.log(2,2)

# a =[1,2,3,4,5,1,2]
# print Counter(a.most_common(1))

def entro(X):
	x = [] # luu lai danh sach cac gia tri X[i] da tinh
	tong_so_lan = 0
	result = 0
	p=[]
	for i in range(len(X)):
		if Counter(x)[X[i]]==0:
			so_lan = Counter(X)[X[i]]
			tong_so_lan += so_lan
			x.append(X[i])
			P = 1.0*so_lan / len(X)
			p.append(P)
			result -= P * math.log(P,2)
		if tong_so_lan == len(X):
			break
	return result
# entropy(X|Y)
def entroXY(X,Y):
	y = []
	result = 0
	pY = []
	tong_so_lan_Y = 0
	for i in range(len(Y)):
		# print Counter(y)[Y[i]]
		if Counter(y)[Y[i]]==0:
			x=[]
			so_lan_Y = Counter(Y)[Y[i]]
			tong_so_lan_Y += so_lan_Y
			y.append(Y[i])
			PY = 1.0* so_lan_Y / len(Y)
			# vi_tri = Y.index(Y[i])
			vi_tri=[]
			for k in range(len(Y)):
				if Y[k] == Y[i]: 
					vi_tri.append(k)
			for j in range(len(vi_tri)):
				x.append(X[vi_tri[j]])
			entro_thanh_phan = entro(x)
			result += PY * entro_thanh_phan
		if tong_so_lan_Y == len(Y):
			break
	return result
def infomation_gain(X,Y):
	return entro(X) - entroXY(X,Y)
def symmetrical_uncertainly(X,Y):
	return 2.0*infomation_gain(X,Y)/(entro(X)+entro(Y))

link = './data/google_trace_timeseries/data_resource_usage_5Minutes_6176858948.csv'
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv(link, header=None, index_col=False, names=colnames, usecols=[3,4,9,10], engine='python')
cpu = df['cpu_rate'].values.reshape(-1,1)
mem = df['mem_usage'].values.reshape(-1,1)
disk_io_time = df['disk_io_time'].values.reshape(-1,1)
disk_space = df['disk_space'].values.reshape(-1,1)

interval_cpu = 0.001
fuzzy_engine_cpu = Fuzzification(interval_cpu)
fuzzied_cpu = fuzzy_engine_cpu.fuzzify(cpu)

interval_mem = 0.00005
fuzzy_engine_mem = Fuzzification(interval_mem)
fuzzied_mem = fuzzy_engine_mem.fuzzify(mem)

interval_disk_io = 0.000001
fuzzy_engine_disk_io = Fuzzification(interval_disk_io)
fuzzied_disk_io = fuzzy_engine_mem.fuzzify(disk_io_time)

interval_disk_space = 0.0000001
fuzzy_engine_disk_space = Fuzzification(interval_disk_space)
fuzzied_disk_space = fuzzy_engine_mem.fuzzify(disk_space)
su=[]
# entropyGGTrace = []
# # numberOfEntropy = 0
print (infomation_gain(fuzzied_cpu,fuzzied_mem))

su = symmetrical_uncertainly(fuzzied_cpu, fuzzied_mem)
print (su)
print (symmetrical_uncertainly(fuzzied_cpu,fuzzied_disk_io))
print (symmetrical_uncertainly(fuzzied_cpu,fuzzied_disk_space))
print (symmetrical_uncertainly(fuzzied_mem,fuzzied_disk_io))
print (symmetrical_uncertainly(fuzzied_mem,fuzzied_disk_space))
print (symmetrical_uncertainly(fuzzied_disk_io,fuzzied_disk_space))
# for i in range(len(colnames)):
# 	print i
# 	sui=[]
# 	for k in range(i+1):
# 		if(k==i):
# 			sui.append(1)
# 		else:
# 			sui.append(symmetrical_uncertainly(df[colnames[i]].values,df[colnames[k]].values))
# 	for j in range(i+1, len(colnames),1):
# 		sui.append(symmetrical_uncertainly(df[colnames[i]].values,df[colnames[j]].values))
# 	su.append(sui)
# print su
# # su=[[1,2,3],[2,3,4]]
# dataFuzzyDf = pd.DataFrame(np.array(su))
# dataFuzzyDf.to_csv('data/SU_4_FuzzytwoMinutes_6176858948.csv', index=False, header=None)
