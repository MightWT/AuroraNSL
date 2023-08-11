import numpy as np
import os

data1_path = ''
data2_path = ''

result_path = ''

data1 = np.load(data1_path)
data2 = np.load(data2_path)


print('data1_shape ',data1.shape)
print('data2_shape ',data2.shape)


result1 = np.concatenate((data1,data2),axis=0)
print('result1_shape ',result1.shape)

np.save(result_path+'perpen',result1)