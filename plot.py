from matplotlib import pyplot as plt
import numpy as np
from scipy import io


#%% impact of Mr, SNR = 10 dB
#M_list = [32,40,48,56,64]
#
#SOMP_list = 10*np.log10([0.5825,0.5476,0.5234,0.5051,0.5007])
#SOMP_Angular_list = 10*np.log10([0.6599,0.6162,0.5911,0.5772,0.5673])
#MSBL_list = 10*np.log10([0.8574,0.7609,0.6626,0.5517,0.4275])
#MSBL_Angular_list = 10*np.log10([0.3492,0.2610,0.2075,0.1647,0.1361])
#AMP_SBL_unfolding_list = 10*np.log10([0.1558,0.1155,0.0998,0.0731,0.0633])
#AMP_SBL_unfolding_Angular_list = 10*np.log10([0.2377,0.1836,0.1462,0.1205,0.1016])
#
#plt.figure()
#
#plt.plot(M_list,SOMP_list,'go-',label='SOMP-PD')
#plt.plot(M_list,SOMP_Angular_list,'go--',label='SOMP-AD')
#plt.plot(M_list,MSBL_list,'rd-',label='MSBL-PD')
#plt.plot(M_list,MSBL_Angular_list,'rd--',label='MSBL-AD')
#plt.plot(M_list,AMP_SBL_unfolding_list,'b*-',label='AMP-SBL unfolding-PD')
#plt.plot(M_list,AMP_SBL_unfolding_Angular_list,'b*--',label='AMP-SBL unfolding-AD')
#
#plt.xlabel('M')
#plt.ylabel('NMSE (dB)')
#plt.legend(loc='lower left')
#
#plt.xlim(32,64)
#plt.xticks(M_list)
#
#plt.ylim(-15,0)
#plt.yticks([-15,-12,-9,-6,-3,0])
#
#plt.grid()
#plt.savefig('./figures/impact_of_M.pdf')


#%%
M_list = [32,40,48,56,64]

SOMP_list = 10*np.log10([0.5825,0.5476,0.5234,0.5051,0.5007])
SOMP_Angular_list = 10*np.log10([0.6599,0.6162,0.5911,0.5772,0.5673])
MSBL_list = 10*np.log10([0.8574,0.7609,0.6626,0.5517,0.4275])
MSBL_Angular_list = 10*np.log10([0.3492,0.2610,0.2075,0.1647,0.1361])
AMP_SBL_unfolding_list = 10*np.log10([0.1558,0.1155,0.0998,0.0731,0.0633])
AMP_SBL_unfolding_Angular_list = 10*np.log10([0.2377,0.1836,0.1462,0.1205,0.1016])

plt.figure()

plt.plot(M_list,SOMP_list,'go',label='SOMP')
plt.plot(M_list,MSBL_list,'rd',label='MSBL')
plt.plot(M_list,AMP_SBL_unfolding_list,'b*',label='Proposed')

plt.plot(M_list,np.ones(5),'k-',label='With PD')
plt.plot(M_list,np.ones(5),'k:',label='With AD')

plt.plot(M_list,SOMP_list,'go-')
plt.plot(M_list,SOMP_Angular_list,'go:')
plt.plot(M_list,MSBL_list,'rd-')
plt.plot(M_list,MSBL_Angular_list,'rd:')
plt.plot(M_list,AMP_SBL_unfolding_list,'b*-')
plt.plot(M_list,AMP_SBL_unfolding_Angular_list,'b*:')

plt.xlabel('M')
plt.ylabel('NMSE (dB)')
plt.legend(loc='lower left')

plt.xlim(32,64)
plt.xticks(M_list)

plt.ylim(-15,0)
plt.yticks([-15,-12,-9,-6,-3,0])

plt.grid()
plt.savefig('./figures/impact_of_M.pdf')


#%% impact of SNR
SNR_list = [0,5,10,15,20]

# M = 32
MSBL_Angular_list1 = 10*np.log10([0.6405,0.4644,0.3492,0.2885,0.2629])
AMP_SBL_unfolding_list_ST1 = 10*np.log10([0.4652,0.2728,0.1558,0.1011,0.0725])
AMP_SBL_unfolding_list_MT1 = 10*np.log10([0.4821,0.2793,0.1632,0.1057,0.0825])

# M = 48
MSBL_Angular_list2 = 10*np.log10([0.5055,0.3227,0.2075,0.1434,0.1117])
AMP_SBL_unfolding_list_ST2 = 10*np.log10([0.3449,0.1800,0.0998,0.0493,0.0307])
AMP_SBL_unfolding_list_MT2 = 10*np.log10([0.3561,0.1862,0.0957,0.0534,0.0346])

# M = 64
MSBL_Angular_list3 = 10*np.log10([0.4175,0.2438,0.1361,0.0765,0.0458])
AMP_SBL_unfolding_list_ST3 = 10*np.log10([0.2775,0.1356,0.0633,0.0300,0.0163])
AMP_SBL_unfolding_list_MT3 = 10*np.log10([0.2844,0.1403,0.0682,0.0342,0.0193])

plt.figure()

plt.plot(SNR_list,MSBL_Angular_list1,'go',label='MSBL-AD')
plt.plot(SNR_list,AMP_SBL_unfolding_list_ST1,'rd',label='Proposed-PD-ST')
plt.plot(SNR_list,AMP_SBL_unfolding_list_MT1,'b*',label='Proposed-PD-MT')

plt.plot(SNR_list,np.ones(5),'k-',label='$M=32$')
plt.plot(SNR_list,np.ones(5),'k-.',label='$M=48$')
plt.plot(SNR_list,np.ones(5),'k:',label='$M=64$')

plt.plot(SNR_list,MSBL_Angular_list1,'go-')
plt.plot(SNR_list,AMP_SBL_unfolding_list_ST1,'rd-')
plt.plot(SNR_list,AMP_SBL_unfolding_list_MT1,'b*-')

plt.plot(SNR_list,MSBL_Angular_list2,'go-.')
plt.plot(SNR_list,AMP_SBL_unfolding_list_ST2,'rd-.')
plt.plot(SNR_list,AMP_SBL_unfolding_list_MT2,'b*-.')

plt.plot(SNR_list,MSBL_Angular_list3,'go:') # -.
plt.plot(SNR_list,AMP_SBL_unfolding_list_ST3,'rd:')
plt.plot(SNR_list,AMP_SBL_unfolding_list_MT3,'b*:')


plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.legend(loc='lower left')

plt.xlim(0,20)
plt.xticks([0,5,10,15,20])
plt.ylim(-20,0)
plt.yticks([-20,-15,-10,-5,0])
plt.grid()

plt.savefig('./figures/impact_of_SNR.eps')


#%% performance-complexity tradeoff

#M = 48 # beams 
#N = 256 # antennas 
#K = 32 # subcarriers
#G_A = 512 # grids 
#
### per iteration real FLOPs 
#AMP_SBL_unfolding = (20*K*M+432)*G
#M_SBL_AF = 16*K*M**2*G_A
#LISTA = 4*K*((4*M+256)*N+32768)
#
## with a common frequency-independent angular dictionary
#M_SBL_AF_FID = 16*M**2*G_A+8*K*M*G_A
#
#
### recovery real FLOPs
#AF_recovery =  8*K*G_A*N
#AD_recovery = 8*K*G_A*N+8*K*G
#
### number of iterations 
#PC_SBL_iter = 100
#SBL_unfolding_iter = 3
#AMP_SBL_unfolding_iter = 10
#M_SBL_AF_iter = 100
#LISTA_iter = 20
#
#SBL_iter = 100
#SBL_AF_iter = 100
#
### different algorithms' flops 
#PC_SBL_flops = np.log10(PC_SBL*PC_SBL_iter + AD_recovery)
#SBL_unfolding_flops = np.log10(SBL_unfolding*SBL_unfolding_iter + AD_recovery)
#AMP_SBL_unfolding_flops = np.log10(AMP_SBL_unfolding*AMP_SBL_unfolding_iter + AD_recovery)
#M_SBL_AF_flops = np.log10(M_SBL_AF*M_SBL_AF_iter + AF_recovery)
#LISTA_flops = np.log10(LISTA*LISTA_iter)
#
#M_SBL_AF_FID_flops = np.log10(M_SBL_AF_FID*M_SBL_AF_iter + AF_recovery)
#PC_SBL_FID_flops = PC_SBL_flops
#
#
#### nmse performance
#PC_SBL_nmse = 10*np.log10([0.0226])
#M_SBL_AF_nmse = 10*np.log10([0.1029])
#SBL_unfolding_nmse = 10*np.log10([0.0148])
#AMP_SBL_unfolding_nmse = 10*np.log10([0.0225])
#LISTA_nmse = 10*np.log10([0.0806])
#
#SBL_FID_nmse = 10*np.log10([0.0885])
#SBL_nmse = 10*np.log10([0.0561])
#SBL_AF_nmse = 10*np.log10([0.3105])
#M_SBL_AF_FID_nmse = 10*np.log10([0.1373])
#PC_SBL_FID_nmse = 10*np.log10([0.0350])
#
#
#plt.figure()
#plt.plot(SBL_unfolding_flops,SBL_unfolding_nmse,'yd',markerfacecolor='white',label='SBL unfolding, %d iters'%SBL_unfolding_iter)
#plt.plot(AMP_SBL_unfolding_flops, AMP_SBL_unfolding_nmse,'b*',markerfacecolor='white',label='AMP-SBL unfolding, %d iters'%AMP_SBL_unfolding_iter)
#plt.plot(PC_SBL_FID_flops,SBL_FID_nmse,'r>',markerfacecolor='white',label='SBL FID, %d iters'%SBL_iter)
#plt.plot(PC_SBL_flops,SBL_nmse,'^',markerfacecolor='white',label='SBL, %d iters'%SBL_iter)
#plt.plot(PC_SBL_flops,PC_SBL_nmse,'bo',markerfacecolor='white',label='PC-SBL, %d iters'%PC_SBL_iter)
#plt.plot(M_SBL_AF_flops,SBL_AF_nmse,'x',markerfacecolor='white',label='SBL AF, %d iters'%SBL_AF_iter)
#plt.plot(M_SBL_AF_FID_flops,M_SBL_AF_FID_nmse,'c<',markerfacecolor='white',label='M-SBL AF FID, %d iters'%M_SBL_AF_iter)
#plt.plot(M_SBL_AF_flops,M_SBL_AF_nmse,'m+',markerfacecolor='white',label='M-SBL AF, %d iters'%M_SBL_AF_iter)
#plt.plot(LISTA_flops,LISTA_nmse,'kv',markerfacecolor='white',label='LISTA, %d iters'%LISTA_iter)
#
#
#plt.xlim(7,13)
#plt.xticks([7,8,9,10,11,12,13])
#plt.yticks([-20,-15,-10,-5,0])
#
#plt.grid()
#
#plt.xlabel('FLOPs (10^)')
#plt.ylabel('NMSE (dB)')
#
#plt.legend(loc='upper right',facecolor='none')
#
#plt.savefig('./figures/tradeoff.eps')