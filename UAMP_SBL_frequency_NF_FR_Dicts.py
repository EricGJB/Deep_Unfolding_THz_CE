import numpy as np
random_seed = 2023
np.random.seed(random_seed)
from matplotlib import pyplot as plt
from scipy import io
from functions import dictionary,update_mu_Sigma_delay,update_mu_Sigma,update_mu_Sigma_MSBL,C2R

# use partial dataset 
test_num = 1
Nr = 256
SNR = 10 # SNR
sigma_2 = 1/10**(SNR/10) # noise variance
Mr = 64
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
data = io.loadmat('./data/data_%dBeams_%dSNR_%s_%dscs.mat'%(Mr,SNR,channel_model,num_sc))
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


#def dictionary_polar(N, d, lamda, G_angle, fn, beta, rho_min, rho_max):
#    dictionary = []
#
#    theta = np.linspace(-1 + 1 / G_angle, 1 - 1 / G_angle, G_angle)
#    
#    for idx in range(G_angle):
#        Z = (N*d)**2 * ( 1 - theta[idx]**2) / 2 / lamda / beta**2
#        kmax = int(np.floor(Z/rho_min))
#        kmin = int(np.floor(Z/rho_max)+1)
#
#        r = np.zeros(kmax - kmin + 2)
#        r[0] = (N*d)**2 * 2 / lamda # rayleigh distance
#        r[1:] = Z/np.arange(kmin,kmax+1)
#
#        for rr in r:
#            dictionary.append(polar_domain_manifold(N, d, fn, rr, np.arcsin(theta[idx])))
#            
#    dictionary = np.concatenate(dictionary,axis=-1)
#    
#    return dictionary


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


#%% UAMP-SBL frequency 
def AMP_SBL(R,Phi,G,Mr,sigma_2,num_iter):
    Tau_x = np.ones((G,1))
    X_hat = np.zeros((G,1))
    epsilon = 0.001
    Gamma_hat = np.ones((G,1))
    beta = 1/sigma_2 
    S = np.zeros((Mr,1))
    for i in range(num_iter):
        Tau_p = (np.abs(Phi)**2).dot(Tau_x)
        P = Phi.dot(X_hat)-Tau_p*S
        Tau_s = 1/(Tau_p+1/beta)
        S = Tau_s*(R-P)
        Tau_q = 1/((np.abs(np.transpose(np.conjugate(Phi)))**2).dot(Tau_s))
        Q = X_hat + Tau_q*np.transpose(np.conjugate(Phi)).dot(S)
        Tau_x = Tau_q/(1+Tau_q*Gamma_hat)
        X_hat = Q/(1+Tau_q*Gamma_hat)
        Gamma_hat = (2*epsilon+1)/(np.abs(X_hat)**2+Tau_x)
        epsilon = 0.5*np.sqrt(np.log10(np.mean(Gamma_hat))-np.mean(np.log10(Gamma_hat)))
    return X_hat

num_iter = 100

y_list = np.zeros(Y_list.shape,dtype=np.complex64)
for i in range(num_sc):
    U,Sigma,V = np.linalg.svd(Phi_list[i])
    y_list[:,i] = np.transpose(np.transpose(np.conjugate(U)).dot(np.transpose(Y_list[:,i])))
    Phi_list[i] = np.transpose(np.conjugate(U)).dot(Phi_list[i])
y_list = np.expand_dims(y_list,axis=-1)

error_list = []
error_nmse_list = []
for i in range(test_num): 
    if i%10==0:
        print('Sample %d/%d'%(i,test_num))
    true_H = H_list[i]
    prediction_H = np.zeros(true_H.shape,dtype=np.complex64)
    prediction_X = np.zeros((G_angle,num_sc),dtype=np.complex64)
    for j in range(num_sc):
        r = y_list[i,j]
        x_hat = AMP_SBL(r,Phi_list[j],G_angle,Mr,sigma_2,num_iter)
        prediction_X[:,j] = np.squeeze(x_hat)
        prediction_h = A_list[j].dot(x_hat)
        prediction_H[:,j] = np.squeeze(prediction_h)
    error_list.append(np.linalg.norm(prediction_H - true_H) ** 2)
    error_nmse_list.append((np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2)

mse_amp_sbl = np.mean(error_list)/(Nr * num_sc)
nmse_amp_sbl = np.mean(error_nmse_list)
print('Angular Dicts')
print(mse_amp_sbl)
print(nmse_amp_sbl)

plt.figure()
plt.plot(np.abs(prediction_X[:,0]))
plt.plot(np.abs(prediction_X[:,-1]))


y_list = np.zeros(Y_list.shape,dtype=np.complex64)
for i in range(num_sc):
    U,Sigma,V = np.linalg.svd(Phi_list_polar[i])
    y_list[:,i] = np.transpose(np.transpose(np.conjugate(U)).dot(np.transpose(Y_list[:,i])))
    Phi_list_polar[i] = np.transpose(np.conjugate(U)).dot(Phi_list_polar[i])
y_list = np.expand_dims(y_list,axis=-1)

error_list_polar = []
error_nmse_list_polar = []
for i in range(test_num): 
    if i%10==0:
        print('Sample %d/%d'%(i,test_num))
    true_H = H_list[i]
    prediction_H = np.zeros(true_H.shape,dtype=np.complex64)
    prediction_X = np.zeros((G_polar,num_sc),dtype=np.complex64)
    for j in range(num_sc):
        r = y_list[i,j]
        x_hat = AMP_SBL(r,Phi_list_polar[j],G_polar,Mr,sigma_2,num_iter)
        prediction_X[:,j] = np.squeeze(x_hat)
        prediction_h = A_list_polar[j].dot(x_hat)
        prediction_H[:,j] = np.squeeze(prediction_h)
    error_list_polar.append(np.linalg.norm(prediction_H - true_H) ** 2)
    error_nmse_list_polar.append((np.linalg.norm(prediction_H-true_H)/np.linalg.norm(true_H)) ** 2)

mse_amp_sbl = np.mean(error_list_polar)/(Nr * num_sc)
nmse_amp_sbl = np.mean(error_nmse_list_polar)
print('Polar Dicts')
print(mse_amp_sbl)
print(nmse_amp_sbl)

plt.figure()
plt.plot(np.abs(prediction_X[:,0]))
plt.plot(np.abs(prediction_X[:,-1]))


print('Performance difference:')
difference = np.array(error_nmse_list) - np.array(error_nmse_list_polar)
difference = np.maximum(difference,-1)
difference = np.minimum(difference,1)
print(difference)
plt.figure()
plt.plot(difference)
plt.plot(np.zeros(test_num))
