import Convolution3D as cnv 
import matplotlib.pyplot as plt 
import numpy
import math 
import json
import scipy.fft

import FastStray




#ploting setup
plt.figure
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')

#=====================================================================================================================================
#Parseing
#=====================================================================================================================================
#From a Json file we want to parce the data for the filter
#input Json Example:
#"Trajectory": {"Trans": [100,200,300], "Orient": [0,-0.707,-0.707,0], "TimeMs": 17000,"Grip": true}, ... }
#output array example: 
#[[x1,y1,t1], [x2,y2,t2], ... , [xn,yn,tn]]

file = open('TrajectoryData.json')
rawdata = json.load(file)
traj = rawdata['Trajectory']

filterTraj = []
for i in range(0, len(traj)):
    filterTraj.append([traj[i]['Trans'][0], traj[i]['Trans'][1], traj[i]['Trans'][2], traj[i]['TimeMs']])
#print(filterTraj)

#=====================================================================================================================================
#Filter
#=====================================================================================================================================
#Step 0: Setup
#alpha: size of the moving average filter
#beta: size of the neighborhood to measure the correlation coefficient
#gamma: size of the neighborhood to preform the non-maximum compression

filter = FastStray.FastStray(params={'alpha':10, 'beta': 5, 'gamma': 5}, position= numpy.array(filterTraj))
filter.initalise()
plt.scatter(filter.position[:,0], filter.position[:,1],filter.position[:,2])

filter.moving_average()
plt.scatter(filter.filtering_spatial_position[:,0], filter.filtering_spatial_position[:,1], filter.filtering_spatial_position[:,2])

filter.update_coeff()

filter.update_max_coeff()

filterIndex = filter.simplified_trajectory()
print(filterIndex)
print(f"compression rate is about: {filter.compression_rate()}")
plt.scatter(filter.simplified_spatial_position[:, 0], filter.simplified_spatial_position[:, 1],filter.simplified_spatial_position[:, 2])
plt.show()


#=====================================================================================================================================
#=====================================================================================================================================



