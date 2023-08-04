#%% generate channel
from matplotlib import pyplot as plt
import numpy as np
random_seed = 2023
np.random.seed(random_seed)

# fixed system parameters
fc, fs, tau_max, num_subpaths = 100 * 1e9, 10 * 1e9, 20 * 1e-9, 10

# varaibles 
Nr, num_sc, data_num, num_clusters = 256, 32, 200, 3

channel_model = 'cluster'

from scipy import io
channelset_name = 'channel_%s_%dscs_test.mat'%(channel_model,num_sc)
H_list = io.loadmat('./data/'+channelset_name)['H_list']

print(H_list.shape)

#%% generate data, partially connected, FDD. s is an all-one vector, thus resulting in a simple random W matrix
SNR = 20 # SNR
sigma_2 = 1/10**(SNR/10) # noise variance

Mr = 32

# the receive combining matrix, consists of random binary phases  
W = np.random.choice([1,-1],Nr*Mr,replace=True)/np.sqrt(Nr)
W = np.reshape(W, (Nr, Mr))
W = np.matrix(W)

H_list_tmp = np.reshape(H_list,(-1,Nr))

noise_list = np.sqrt(sigma_2/2)*(np.random.randn(Mr,data_num*num_sc)+1j*np.random.randn(Mr,data_num*num_sc))

Y_list = np.transpose(W.H.dot(np.transpose(H_list_tmp))+noise_list)

Y_list = np.array(Y_list)

Y_list = np.reshape(Y_list,(data_num,num_sc,Mr))


dataset_name = 'data_%dBeams_%dSNR_%s_%dscs_test.mat'%(Mr,SNR,channel_model,num_sc)
io.savemat('./data/'+dataset_name,{'H_list':H_list,'Y_list':Y_list,'W':W})
print(dataset_name)
print('Dataset generated!')
