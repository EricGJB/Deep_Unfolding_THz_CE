#%% generate channel
from matplotlib import pyplot as plt
import numpy as np
random_seed = 2023
np.random.seed(random_seed)

# fixed system parameters
fc, fs, tau_max, num_subpaths = 100 * 1e9, 10 * 1e9, 20 * 1e-9, 10

# varaibles 
Nr, num_sc, data_num, num_clusters = 256, 32, 10000, 3

c = 3e8
lambda_c = c/fc
d = lambda_c / 2

# rayleigh distance 
d_r = 2*((Nr-1)*d)**2/lambda_c
print('Rayleigh distance:',d_r)

channel_model = 'cluster'

if channel_model == 'path':
    AS = 0
    RS = 0
else:
    AS = 4 # degree
    RS = 1 # meters

b_angle = np.sqrt(AS**2/2) # the variance of Laplace distribution is 2b^2
b_distance = np.sqrt(RS**2/2)

eta = fs / num_sc
Lp = num_clusters * num_subpaths

H_list = np.zeros((data_num, num_sc, Nr),dtype=np.complex64)

print('Generating near-field channels')

def near_field_manifold(Nt, d, f, r0, theta0):
    c = 3e8
    nn = np.arange(-(Nt-1)/2, (Nt-1)/2+1)
    r = np.sqrt(r0**2 + (nn*d)**2 - 2*r0*nn*d*np.sin(theta0))
    at = np.exp(-1j*2*np.pi*f*(r-r0)/c)#/np.sqrt(Nt)
    return at

# def far_field_manifold(Nt, d, f, r0, theta0):
#     c = 3e8
#     fc = 100 * 1e9
#     nn = np.arange(0,Nt)
#     at = np.exp(-1j*np.pi*np.sin(theta0)*nn*f/fc)#/np.sqrt(Nt)
#     return at

d_min = 5
d_max = 30

for i in range(data_num):
    if i % 500 == 0:
        print('Channel %d/%d' % (i, data_num)) 
    path_gains = np.sqrt(1 / 2) * (np.random.randn(Lp) + 1j * np.random.randn(Lp))
    taus = np.zeros(Lp)
    AoAs = np.zeros(Lp)
    distances = np.zeros(Lp)

    for nc in range(num_clusters):
        # truncated laplacian distribution
        mean_AoA = np.random.uniform(0,360)    
        AoAs_cluster = np.random.laplace(loc=mean_AoA, scale=b_angle, size=num_subpaths) 
        AoAs_cluster = np.maximum(AoAs_cluster, mean_AoA-2*AS)
        AoAs_cluster = np.minimum(AoAs_cluster, mean_AoA + 2 * AS)
        AoAs[nc*num_subpaths:(nc+1)*num_subpaths] = AoAs_cluster / 180 * np.pi
        
        mean_distance = np.random.uniform(d_min,d_max)
        distances_cluster = np.random.laplace(loc=mean_distance, scale=b_distance, size=num_subpaths) 
        distances_cluster = np.maximum(distances_cluster, mean_distance-2*RS)
        distances_cluster = np.minimum(distances_cluster, mean_distance + 2 * RS)
        distances[nc*num_subpaths:(nc+1)*num_subpaths] = distances_cluster

    for n in range(num_sc):
        fn = fc + eta*(n-(num_sc-1)/2)
#        path_loss = 1/fn
        h_sample = 0 
        for l in range(Lp):
#            h_sample = h_sample + near_field_manifold(Nr, d, fc, distances[l], AoAs[l]) # without beam split
#            path_loss = c/(4*np.pi*fn*distances[l])
            path_loss = 1
            h_sample = h_sample + path_loss*path_gains[l]*np.exp(-1j*2*np.pi*fn/c*distances[l])*near_field_manifold(Nr, d, fn, distances[l], AoAs[l])
            # h_sample = h_sample + far_field_manifold(Nr, d, fn, distances[l], AoAs[l]) # far field 
        h_sample = h_sample/np.sqrt(Lp)
        H_list[i,n] = h_sample
        

print(H_list.shape) # (data_num, num_sc, Nr)
print('\n')

plt.plot(np.abs(np.fft.fft(h_sample)))

from scipy import io
channelset_name = 'channel_%s_%dscs.mat'%(channel_model,num_sc)
io.savemat('./data/'+channelset_name,{'H_list':H_list})
print(channelset_name)
print('Channelset generated!')
