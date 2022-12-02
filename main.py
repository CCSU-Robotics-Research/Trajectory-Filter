import Convolution3D as cnv 
import matplotlib.pyplot as plt 
import numpy as np 
import math 
import json
import scipy.fft

#Filter Properties:
alpha = 3 # window size for moveing average filter. noise / #datapoints tradeoff. needs to be odd 


#import & parce data for filter
f = open('TrajectoryData.Json')
data = json.load(f)
traj = data['Trajectory']

StateVectorSize = 3
trans = np.empty([len(traj),StateVectorSize])

#saves each element to trans 
for i in range(len(traj)):
    for j in range(StateVectorSize):
        trans[i,j] = (traj[i]['Trans'][j])
        
#converts list trans to array trans
Trans = np.array(trans)

#ploting input path
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.scatter(Trans[:,0],Trans[:,1],Trans[:,2])


#convolute array trans
kernal = math.pow(alpha,-3) * np.ones([alpha,alpha])
Trans_AV = cnv.conv3D(Trans,kernal)

ax.scatter(Trans_AV[:,0],Trans_AV[:,1],Trans_AV[:,2])



plt.show()



