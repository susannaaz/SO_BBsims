import pysm
from pysm.nominal import models
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# Map parameters
#---------------------------------------------------
nside = 512 #int(len(ells)/3)
lmax = 3*nside-1
ells = np.arange(lmax+1)

# Power law model
#---------------------------------------------------
cl_betaSync= ((ells+0.001) / 80.)**gamma_beta #  # A_beta *                                 

# Map from given power spectrum as mean + gaussian random field
#---------------------------------------------------
delta_map = hp.synfast(cl_betaSync, nside, new=True, verbose=False)

sigma_map = np.sqrt(np.mean(delta_map**2))
sigma_des = 0.1 #desired typical map variation                                                                             
delta_beta = delta_map * sigma_des / sigma_map

map_beta = delta_beta + np.mean(delta_beta)

hp.mollview(map_beta, unit='$\\beta$',  title='Scaling Index beta_sync') #unit='$\\beta$'                                  
delta_map -= delta_beta + np.mean(delta_beta)
map_beta1 = delta_map

hp.mollview(map_beta1, unit='$\\beta$',  title='Scaling Index beta_sync') #unit='$\\beta$'                                 
plt.show()

# Power spectrum from map (output)
#---------------------------------------------------
cl_betaSync_out = hp.anafast(map_beta, lmax=lmax)
cl_betaSync_out1 = hp.anafast(map_beta1, lmax=lmax)

# Compare input and output power spectra 
#---------------------------------------------------
plt.figure()
plt.semilogy(ells, cl_betaSync, 'b',label="input power spectrum")
#plt.semilogy(ells, cl_betaSync_out, 'r',label="output power spectrum")
plt.semilogy(ells, cl_betaSync_out1, 'r',label="output power spectrum 1")
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell$')
plt.legend()
plt.grid()
plt.show()


