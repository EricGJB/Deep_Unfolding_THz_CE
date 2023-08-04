import numpy as np
random_seed = 2023
np.random.seed(random_seed)
from matplotlib import pyplot as plt
from scipy import io
from functions import dictionary,update_mu_Sigma,C2R

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-Mr')
parser.add_argument('-SNR')
args = parser.parse_args()

Mr = int(args.Mr)
SNR = int(args.SNR)

# use partial dataset 
test_num = 20

Nr = 256

sigma_2 = 1/10**(SNR/10) # noise variance

num_sc = 32 # number of subcarriers
fc = 100 * 1e9 # central frequency
fs = 10 * 1e9 # bandwidth
eta = fs / num_sc  # subcarrier spacing
c = 3e8
lamda = c/fc
d = lamda/2 # half wavelength

s = 2
G_angle = s*Nr # angular resolution
A_R = dictionary(Nr, G_angle)
A_R = np.matrix(A_R) 

num_clusters = 3

channel_model = 'cluster'

data = io.loadmat('./data/data_%dBeams_%dSNR_%s_%dscs_test.mat'%(Mr,SNR,channel_model,num_sc))
H_list = data['H_list'][:test_num]
H_list = np.transpose(H_list,(0,2,1))
Y_list = data['Y_list'][:test_num]
Y_real_imag_list = C2R(np.transpose(Y_list,(0,2,1)))

W = np.matrix(data['W'])


#%% 
def polar_domain_manifold(Nt, d, f, r0, theta0):
    c = 3e8
    nn = np.arange(-(Nt-1)/2, (Nt-1)/2+1)
    r = r0 - nn * d * np.sin(theta0) + nn**2 * d**2 * np.cos(theta0)**2 / 2 /r0
    at = np.exp(-1j*2*np.pi*f*(r-r0)/c)/np.sqrt(Nt)
    return np.expand_dims(at,axis=-1)

def dictionary_polar(N, d, lamda, G_angle, fn, beta, rho_min, rho_max):
    dictionary = []

    theta = np.linspace(-1 + 1 / G_angle, 1 - 1 / G_angle, G_angle)
    
    # far-field atoms 
    rr = 1e5
    for idx in range(G_angle):    
        dictionary.append(polar_domain_manifold(N, d, fn, rr, np.arcsin(theta[idx])))   
    
    # near-field atoms 
    Z = (N*d)**2 / 2 / lamda / beta**2
    s = 1
    while Z/s>=rho_min:
        for idx in range(G_angle):    
            rr = Z/s*(1-theta[idx]**2)
            dictionary.append(polar_domain_manifold(N, d, fn, rr, np.arcsin(theta[idx])))    
        s = s + 1
            
    dictionary = np.concatenate(dictionary,axis=-1)
    
    return dictionary

A_list_polar = []

beta = 1.2

rho_min = 3
rho_max = 64

A_n_FID = dictionary_polar(Nr, d, lamda, G_angle, fc, beta, rho_min, rho_max)

G_polar = A_n_FID.shape[-1]
print(G_polar)

for n in range(num_sc):
    fn = fc + (n-(num_sc-1)/2)*eta
    A_n = dictionary_polar(Nr, d, lamda, G_angle, fn, beta, rho_min, rho_max)
    A_list_polar.append(A_n)

Phi_list_polar = np.zeros((num_sc, Mr, G_polar)) + 1j * np.zeros((num_sc, Mr, G_polar))
for i in range(num_sc):
    Phi_list_polar[i] = W.H.dot(A_list_polar[i])
Phi_list_polar = np.tile(np.expand_dims(Phi_list_polar,axis=0),(test_num,1,1,1))
Phi_real_imag_list_polar = C2R(Phi_list_polar)



def dictionary_angle(N, G, sin_value):
    A = np.exp(-1j * np.pi * np.reshape(np.arange(N),(N,1)).dot(np.reshape(sin_value,(1,G)))) / np.sqrt(N)
    return A

sin_value_sc0 = np.linspace(-1 + 1 / G_angle, 1 - 1 / G_angle, G_angle)

A_list = []
for n in range(num_sc):
    fn = fc + (n-(num_sc-1)/2)*eta
    sin_value_scn = sin_value_sc0*(fn/fc) 
    # sin_value_scn = sin_value_sc0 # frequency-independent measurement matrices
    A_list.append(dictionary_angle(Nr, G_angle, sin_value_scn))
A_list = np.array(A_list)

Phi_list = np.zeros((num_sc, Mr, G_angle)) + 1j * np.zeros((num_sc, Mr, G_angle))
for i in range(num_sc):
    Phi_list[i] = W.H.dot(A_list[i])
Phi_list = np.tile(np.expand_dims(Phi_list,axis=0),(test_num,1,1,1))
Phi_real_imag_list = C2R(Phi_list)


#%% SBL frequency 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Lambda

use_gpu = 1

if use_gpu:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0],True)

num_layers = 100

batch_size = 5

plot_sample_index = 0


def FR_MSBL_layer(Mr, num_sc, G, sigma_2):
    Phi_real_imag = Input(shape=(num_sc, Mr, G, 2))
    y_real_imag = Input(shape=(Mr, num_sc, 2))
    alpha_list = Input(shape=(G, num_sc))
    # update mu and Sigma
    mu_real = []
    mu_imag = []
    diag_Sigma_real = []
    for i in range(num_sc):
        mu_real_sc, mu_imag_sc, diag_Sigma_real_sc = Lambda(lambda x: update_mu_Sigma(x,1,sigma_2,Mr))(
            [Phi_real_imag[:,i], y_real_imag[:,:,i:i+1], alpha_list[:,:,i:i+1]])
        mu_real.append(mu_real_sc)
        mu_imag.append(mu_imag_sc)
        diag_Sigma_real.append(diag_Sigma_real_sc)
    mu_real = tf.concat(mu_real,axis=-1)
    mu_imag = tf.concat(mu_imag,axis=-1)
    diag_Sigma_real = tf.concat(diag_Sigma_real,axis=-1)
    
    # update alpha_list
    mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])
    
    # M-SBL
    mu_square_average = Lambda(lambda x:tf.reduce_mean(x,axis=-1,keepdims=True))(mu_square)
    mu_square = tf.tile(mu_square_average,(1,1,num_sc))
    alpha_list_updated = Lambda(lambda x: x[0] + x[1])([mu_square, diag_Sigma_real])
    model = Model(inputs=[Phi_real_imag, y_real_imag, alpha_list], outputs=[alpha_list_updated, mu_real, mu_imag])
    return model

SBL_single_layer = FR_MSBL_layer(Mr, num_sc, G_angle, sigma_2)
SBL_single_layer_polar = FR_MSBL_layer(Mr, num_sc, G_polar, sigma_2)

alpha_list = np.ones((test_num, G_angle, num_sc))  # initialization
alpha_list_polar = np.ones((test_num, G_polar, num_sc))

mse_sbl_list = []
mse_sbl_list_polar = []

for i in range(num_layers):
    if i%10==0:
        print('SBL iteration %d' % i)
    [alpha_list, mu_real, mu_imag] = SBL_single_layer.predict([Phi_real_imag_list, Y_real_imag_list, alpha_list],batch_size=batch_size)
    [alpha_list_polar, mu_real_polar, mu_imag_polar] = SBL_single_layer_polar.predict([Phi_real_imag_list_polar, Y_real_imag_list, alpha_list_polar],batch_size=batch_size)

# final performance
predictions_X = mu_real + 1j * mu_imag
predictions_X_polar = mu_real_polar + 1j * mu_imag_polar

error = 0
error_nmse = 0
error_polar = 0
error_nmse_polar = 0

for i in range(test_num):
    true_H = H_list[i]
    prediction_H = np.zeros(true_H.shape,dtype=np.complex64)
    prediction_H_polar = np.zeros(true_H.shape,dtype=np.complex64)
    for j in range(num_sc):
        prediction_h = A_list[j].dot(predictions_X[i,:,j:j+1])
        prediction_H[:,j:j+1] = prediction_h
        prediction_h_polar = A_list_polar[j].dot(predictions_X_polar[i,:,j:j+1])
        prediction_H_polar[:,j:j+1] = prediction_h_polar
    error = error + np.linalg.norm(prediction_H - true_H) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_H - true_H) / np.linalg.norm(true_H)) ** 2
    error_polar = error_polar + np.linalg.norm(prediction_H_polar - true_H) ** 2
    error_nmse_polar = error_nmse_polar + (np.linalg.norm(prediction_H_polar - true_H) / np.linalg.norm(true_H)) ** 2
mse_sbl = error / (test_num * Nr * num_sc)
nmse_sbl = error_nmse / test_num
print(mse_sbl)
print(nmse_sbl)

mse_sbl_polar = error_polar / (test_num * Nr * num_sc)
nmse_sbl_polar = error_nmse_polar / test_num
print(mse_sbl_polar)
print(nmse_sbl_polar)
#
#plt.figure()
#plt.imshow(np.abs(predictions_X[plot_sample_index]), cmap='gray_r')
#plt.xlabel('$K$')
#plt.ylabel('$G_A$')
#plt.title('AF channel modulus')
#
#plt.figure()
#plt.imshow(np.abs(predictions_X_polar[plot_sample_index]), cmap='gray_r')
#plt.xlabel('$K$')
#plt.ylabel('$G_P$')
#plt.title('PF channel modulus, polar')

plt.figure()
plt.plot(np.abs(predictions_X[plot_sample_index,:,0]))
plt.plot(np.abs(predictions_X[plot_sample_index,:,-1]))

plt.figure()
plt.plot(np.abs(predictions_X_polar[plot_sample_index,:,0]),'b-')
plt.plot(np.abs(predictions_X_polar[plot_sample_index,:,-1]),'r--')
#plt.legend([r'$|\bf{x}^1|$',r'$|\bf{x}^K|$'])
plt.legend(['First Subchannel','Last Subchannel'])
plt.xlabel('Polar Grid Index')
plt.ylabel('Modulus of the Transformed Channels')
plt.ylim(0,6)
#plt.title('Use $K$ Frequency-Dependent Polar Dictionaries')
plt.savefig('./figures/channel2.eps')


#%% SOMP
def SOMP_CE(y_s,A_list,Phi_list,num_sc,num_antenna_bs,num_beams,G_angle,max_iter_count,normalizers):
    residual = np.copy(y_s) # (num_sc, num_beams, 1)

    change_of_residual = 1e4
    
    plot_flag = 0
    
    iter_count = 0
    
    max_angle_indexes = []
    
    while (change_of_residual > 1e-4) & (iter_count < max_iter_count):
        # compute the direction with largest average response energy
        responses = 0
        for n in range(num_sc):
            responses = responses + np.linalg.norm(np.matrix(Phi_list[n]).H.dot(np.matrix(residual[n]))/normalizers[n],axis=-1)**2 # (G_angle, 1) 
        
        max_angle_index = np.argmax(responses)
        max_angle_indexes.append(max_angle_index)
        
        if plot_flag:
            plt.figure()
            plt.plot(responses)
            plot_flag = 0
        
        # update F_RF_n matrices with one vector added 
        if iter_count == 0:
            Phi_n_list = Phi_list[:,:,max_angle_index:max_angle_index+1]           
        else:
            Phi_n_list = np.concatenate([Phi_n_list, Phi_list[:,:,max_angle_index:max_angle_index+1]],axis=-1)
        
        residual_new = np.copy(residual)
        X_hats_tmp = []
        for n in range(num_sc):
            Phi_n = Phi_n_list[n]
            x_hat_n = np.linalg.pinv(Phi_n).dot(y_s[n])
            residual_new[n] = y_s[n] - Phi_n.dot(x_hat_n) 
            X_hats_tmp.append(x_hat_n)

        change_of_residual = np.linalg.norm(residual_new-residual)
#        print(change_of_residual)
        residual = residual_new
        
        iter_count = iter_count + 1
        
    X_hats = np.zeros((num_sc,G_angle,1),dtype=np.complex64)
    # 如果不是因为迭代次数到了而跳出循环，或者存在重复index
    if (iter_count < max_iter_count) or (len(max_angle_indexes)!=len(np.unique(max_angle_indexes))):
        max_angle_indexes = max_angle_indexes[:-1]
        X_hats[:,max_angle_indexes] = np.array(X_hats_tmp)[:,:-1]
    else:
        X_hats[:,max_angle_indexes] = np.array(X_hats_tmp)
        
    assert len(max_angle_indexes)==len(np.unique(max_angle_indexes))    

    H_hats = np.zeros((num_sc,num_antenna_bs,1),dtype=np.complex64)
    for n in range(num_sc):
        x_hat_n = X_hats[n]
        h_hat_n = A_list[n].dot(x_hat_n)
        H_hats[n] = h_hat_n
    
    return H_hats

y_list = np.expand_dims(Y_list,axis=-1) # (test_num,num_sc,Mr,1)
Phi_list = Phi_list[0]
Phi_list_polar = Phi_list_polar[0]

max_iter_count = 2*num_clusters

mse_list = []
mse_list_polar = []
nmse_list = []
nmse_list_polar = []

normalizers = np.linalg.norm(Phi_list,axis=1)
normalizers = np.expand_dims(normalizers,axis=-1)
normalizers_polar = np.linalg.norm(Phi_list_polar,axis=1)
normalizers_polar = np.expand_dims(normalizers_polar,axis=-1)
for i in range(test_num):
    if i % 10 == 0:
        print('Testing sample %d' % i)
    y_s = y_list[i]
    H_trues = H_list[i]
    H_hats = SOMP_CE(y_s, A_list, Phi_list, num_sc, Nr, Mr, G_angle,max_iter_count,normalizers)
    H_hats = np.transpose(H_hats[:,:,0])
    mse = np.linalg.norm(H_hats-H_trues)**2/np.product(H_hats.shape)
    mse_list.append(mse)
    nmse = (np.linalg.norm(H_hats-H_trues)/np.linalg.norm(H_trues))**2
    nmse_list.append(nmse)

    H_hats_polar = SOMP_CE(y_s, A_list_polar, Phi_list_polar, num_sc, Nr, Mr, G_polar,max_iter_count,normalizers_polar)
    H_hats_polar = np.transpose(H_hats_polar[:,:,0])
    mse_polar = np.linalg.norm(H_hats_polar-H_trues)**2/np.product(H_hats.shape)
    mse_list_polar.append(mse_polar)
    nmse_polar = (np.linalg.norm(H_hats_polar-H_trues)/np.linalg.norm(H_trues))**2
    nmse_list_polar.append(nmse_polar)
    
print('SOMP, mse: %.5f, nmse: %.5f'%(np.mean(mse_list),np.mean(nmse_list)))
print('SOMP polar, mse: %.5f, nmse: %.5f'%(np.mean(mse_list_polar),np.mean(nmse_list_polar)))


file_handle=open('./results/sbl_performance',mode='a+')
file_handle.write('Mr=%d, SNR=%d dB:\n'%(Mr,SNR))
file_handle.write(str([nmse_sbl,nmse_sbl_polar,np.mean(nmse_list),np.mean(nmse_list_polar)]))
file_handle.write('\n')
file_handle.write('\n')
file_handle.close()
