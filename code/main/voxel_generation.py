import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os



def read_pc(input_data):
    x = input_data[:, 0]
    y = input_data[:, 1]
    z = input_data[:, 2]
    return x, y, z


def save_npy(obj,save_path,name):
    return np.save(save_path+name+'.npy',obj)


def extrema_value(data):
	max_value = np.amax(data)
	min_value = np.amin(data)
	return max_value,min_value


def scale_range(data):
    tmax = 1
    tmin = -1
    scaled_data = []
    rmax = np.max(data)
    rmin = np.min(data)

    for elem in data:
        scaled_elem = ((elem-rmin)/(rmax-rmin))*(tmax-tmin)+tmin
        scaled_data.append(scaled_elem)
    return scaled_data

def normalized_result(obj):
    x,y,z = read_pc(obj)
    obj_scaled = scale_range(obj)
    result = np.asarray(obj_scaled)

    return result
