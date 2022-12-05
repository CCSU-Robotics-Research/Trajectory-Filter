import numpy as np 
import pandas as pd
from dataclasses import dataclass 
import matplotlib.pyplot as plt 
from  scipy.stats import pearsonr as corr
@dataclass 
class FastStray():
    """A global class that implements faststray algorithms for reduction GPS data point in a given trajectory 
    For more information about the method, you can check the paper on https://arxiv.org/pdf/1608.07338.pdf
    
    attributes
    ----------
    test: used for introducing error during test step, must be set to 0 when position is given as argument
    alpha: size of the moving average filter 
    beta: size of the neighborhood to measure the correaltion coef
    gamma: size of the neighboorhood to perform the non maximum suppression
    position: lon, lat, timestamp data dim = (n_samples, 3)
    spatial_position: lon, lat dim = (n_samples, 2)
    temporal_position: timestamp dim = (n_samples, )
    filtering_spatial_data: moving average on spatial_position
    filtering_temporal_position: moving average on temporal_position data 
    methods:
    --------
    linear_correlation_based_coef: static method, evaluate the correlation between space and time informations 
    ksi_function: more larger is ksi the more information we can extract from the related point 
    moving_average_smooth_trajectory: moving average 
    update_filtering_position: return of MA methods 
    get_params_idx: return intervall which one apply MA algorithm on -- this allows specifying windows for inference
    """
    
    
    params: dict
    position: np.ndarray 
    space: int = 3
    time: int = 1
    test_coeff: int = 0
    error: np.ndarray = np.random.exponential(0.1, (1000, 3)) * test_coeff

    #position: np.ndarray = np.vstack([np.linspace(-3, 3, 1000), np.sin(3*np.linspace(-3, 3, 1000)), np.linspace(0, 1, 1000)]).T + error 
    #sample_size: int = position.shape[0]
    #spatial_dim: tuple = (sample_size, space)
    #temporal_dim: tuple = (sample_size, time)
   
    
    def initalise(self):
        """Sets up object components after trajectory is imported"""
        #gets the height of input array
        self.sample_size = self.position.shape[0]

        #grabs sizes of the two arrays
        self.spatial_dim = (self.sample_size, self.space)
        self.temporal_dim = (self.sample_size, self.time)


    def moving_average(self):
        """Calculate the trajectory 𝑇-1 (composed by a list of points 𝑃1 and time
         stamps 𝑆1) using moving average filter -- alpha param defining the window 
         of the filter -- the filter is computing on the whole trajectory 
        """
        #Splits input array into position and time components
        self.spatial_position = self.position[:,:self.space]
        self.temporal_position = self.position[:,self.space].reshape(-1, 1)

        #creates an array (idx) where each row counts 0 to 16, then once it hits 16 shifts over to 1 - 17... etc.
        #ie: where alpha is 10
        # 00: range(0,9)
        # 01: range(0,10)
        # ...
        # 07: range(0,16) 
        # 08: range(0,16) 
        # 09: range(0,16)
        # 10: range(0,16)
        # 11: range(1,17)
        # 11: range(2,18)
        # ...

        #creates an index per input column element which represents the indicies that the moveing average looks at.   
        idx = self.get_index(param='alpha')

        #maps the mean position function useing the idx array per column of the spatial position array.
        new_spatial_position = np.array([*map(self.mean_position, idx)])

        #returns filtered positions to object wide variables
        self.filtering_spatial_position, self.filtering_temporal_position = new_spatial_position, self.temporal_position

    def mean_position(self, index):
        """return average spatial position"""
        return self.spatial_position[index,:].mean(axis=0)
    
    def sub_spatial_array(self, index):
        return self.filtering_spatial_position[index,:]
    
    def sub_temporal_array(self, index):
        return self.filtering_temporal_position[index,:]

    def sub_coeff(self, index):
        return max(map(self.coeff.__getitem__, index))

    def get_index(self, param):
        return [*map(self.get_params_idx, range(self.sample_size), [self.params[param]]*self.sample_size, [self.sample_size]*self.sample_size)]

    #creates array of linearity coefficients
    def update_coeff(self):
        #creates indicy array per item with beta window size
        idx = self.get_index(param='beta')

        #applies beta window to the diffrent arrays and returns an array of the elements in the window
        p_mu = map(self.sub_spatial_array, idx)
        t_mu = map(self.sub_temporal_array, idx)

        #applies linearity corrolation funciton (ksi_func) to the sub matricies and saves it in the coeff array
        self.coeff = [*map(self.ksi_func, p_mu, t_mu)]

    def update_max_coeff(self):
        idx = self.get_index(param='gamma')
        self.max_coeff = [*map(self.sub_coeff, idx)]

    def simplified_trajectory(self):
        compress_index = np.where(np.array(self.max_coeff) == np.array(self.coeff))[0]
        self.simplified_spatial_position = self.filtering_spatial_position[compress_index,:]
        self.simplified_temporal_position = self.filtering_temporal_position[compress_index,:]
        return(compress_index)

    def run(self):
        self.initalise()
        self.moving_average()
        self.update_coeff()
        self.update_max_coeff()
        self.simplified_trajectory()

    @staticmethod 
    def get_params_idx(idx, value, sample_size):
        ''' gets infinum and supremum of the index'''
        inf, sup = max(0, idx - value), min(idx + value, sample_size)
        return np.arange(inf, sup)

    @staticmethod
    def plot_average_filtering_deviation(init, avg):
        """ init is the noisy trajectory, avg the filter one """
        plt.scatter(avg[:, 0], avg[:, 1])
        plt.scatter(init[:, 0], init[:, 1], color='r')

    @staticmethod
    def ksi_func(pp, tt):
        #parces array elements
        #ppx, ppy, = pp.T
        ppx, ppy, ppz = pp.T

        #calculates linerarity coeficient per element 
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        rat_x = corr(ppx, tt.flatten())[0] ** 2
        rat_y = corr(ppy, tt.flatten())[0] ** 2
        rat_z = corr(ppz, tt.flatten())[0] ** 2
        

        #return 1./ rat_x + 1./ rat_y
        return 1./ rat_x + 1./ rat_y + 1./ rat_z
    
  
    def compression_rate(self):
        return 1. - (len(self.simplified_temporal_position) / len(self.temporal_position))

def main():
    fst = FastStray(params={'alpha':10, 'beta': 10, 'gamma': 10}, position = np.vstack([np.linspace(-3, 3, 1000),np.linspace(-3, 3, 1000), np.sin(3*np.linspace(-3, 3, 1000)), np.linspace(0, 1, 1000)]).T)
    fst.run()
    print(f"compression rate is about: {fst.compression_rate()}")
    plt.scatter(fst.simplified_spatial_position[:, 0], fst.simplified_spatial_position[:, 1])


if __name__ == "__main__":
    main()