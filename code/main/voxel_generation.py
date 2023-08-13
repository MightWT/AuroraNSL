import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os



def seperate_xyz(input_data):
    x = input_data[:, 0]
    y = input_data[:, 1]
    z = input_data[:, 2]
    return x, y, z


def transformation(x, y, z, theta, data):
    """

    :param x: distance for x
    :param y: distance for y
    :param z: distance for z
    :param theta: the angle for rotation, in degree
    :return: transformed matrix
    """
    T = np.eye(4)
    theta_rad = np.radians(theta)
    T[0,0] = math.cos(theta_rad)
    T[0,1] = -math.sin(theta_rad)
    T[0,3] = x
    T[1,0] = math.sin(theta_rad)
    T[1,1] = math.cos(theta_rad)
    T[1,3] = y
    T[2,3] = z
    # print(T)
    # expand the original data for matrix computation
    # create an array with ones
    ones_data_arr = np.ones((data.shape[0]+1,data.shape[1]+1))
    # replace the ones_data_arr with data elements
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ones_data_arr[i,j]=data[i,j]


    expanded_data = ones_data_arr
    # multiply with transformation matrix
    # T*D^T
    transformated_data = np.matmul(T, np.transpose(expanded_data))
    transformated_data = np.transpose(transformated_data)
    # reshape transformed_data into (3,3), remove added 1s.
    reshaped_ones = np.ones((data.shape[0],data.shape[1]))
    for i in range(reshaped_ones.shape[0]):
        for j in range(reshaped_ones.shape[1]):
            reshaped_ones[i,j]=transformated_data[i,j]


    result = reshaped_ones
    return result


def get_centroid(path):
	data = np.load(path)

	data1 = data[0:100]
	data2 = data[100:]

	x1 = [p1[0] for p1 in data1]
	y1 = [p1[1] for p1 in data1]
	z1 = [p1[2] for p1 in data1]

	x2 = [p2[0] for p2 in data2]
	y2 = [p2[1] for p2 in data2]
	z2 = [p2[2] for p2 in data2]

	centroid1 = (sum(x1)/len(data1),sum(y1)/len(data1),
				 sum(z1)/len(data1))

	centroid2 = (sum(x2)/len(data2),sum(y2)/len(data2),
				 sum(z2)/len(data2))


	return centroid1,centroid2


def get_centroid_single_object(data):
    # get data in each x,y,z for first and second object
    x,y,z = seperate_xyz(data)
    centroid = (sum(x)/len(x),sum(y)/len(y), sum(z)/len(z))
    return centroid


def initial_position(ix1, iy1, iz1, ix2, iy2, iz2, obj1, obj2):
    """

    :param ix1: the initial x-coordinate of the first object
    :param iy1: the initial y-coordinate of the first object
    :param iz1: the initial z-cooridnate of the first object
    :param ix2: the initial x-coordinate of the second object
    :param iy2: the initial y-coordinate of the second object
    :param iz2: the initial z-coordinate of the second object
    :param obj1:the first object
    :param obj2:the second object
    :return: the coordinates of the first, second object placed at the initial location
    """

    # get the centroid point of the first, and second object
    cx1,cy1,cz1 = get_centroid_single_object(obj1)
    cx2,cy2,cz2 = get_centroid_single_object(obj2)
    # calculate the translation coefficient.
    a1 = ix1 - cx1
    b1 = iy1 - cy1
    c1 = iz1 - cz1

    a2 = ix2 - cx2
    b2 = iy2 - cy2
    c2 = iz2 - cz2
    # using translation function to move the object to initial position
    pobj1 = transformation(a1,b1,c1,0,obj1)
    pobj2 = transformation(a2,b2,c2,0,obj2)

    return pobj1, pobj2


def concatenate_result(obj1,obj2):
    obj = np.concatenate((obj1,obj2))
    return obj


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
    x,y,z = seperate_xyz(obj)
    obj_scaled = scale_range(obj)
    result = np.asarray(obj_scaled)
    # result = np.transpose(result)

    return result
