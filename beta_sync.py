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

# Gamma must be less than -2 for convergence in 0x2 term
gamma_beta = -2.5 

# Critical value (for convergence) 
# SO freqs: 27., 39., 93., 145., 225., 280. GHz
# ~10 is the ratio between our highest and lowest frequency 280/27 GHz
sigma_crit = 2/np.log(10) 

# Model the standard deviation map as a function of gamma
# std = a * (-gamma)^b * exp(c * gamma)
# best fit parameters a b c
a = 4.16190627
b = -3.28619789
c = -2.56282892
sigma_emp = a * (-gamma_beta)**b * np.exp(c*gamma_beta)                       

# Expected: A_beta=1e-6
A_beta = (sigma_crit / sigma_emp)**2

nu0_sync= 23. 

def fcmb(nu):
    x=0.017608676067552197*nu
    ex=np.exp(x)
    return ex*(x/(ex-1))**2

A_beta = A_beta * fcmb(nu0_sync)**2

# Calculate power spectrum (input)
#---------------------------------------------------

#(Gaussian, power law: C_{\ell} = A \cdot (\frac{\ell+0.001}{80.})^\alpha $) 
cl_betaSync= A_beta * ((ells+0.001) / 80.)**gamma_beta #


# Map from given power spectrum as mean + gaussian random field
#---------------------------------------------------

map_beta = hp.synfast(cl_betaSync, nside, new=True, verbose=False)
delta_beta = map_beta *  sigma_crit / sigma_emp  

print(np.mean(map_beta))

print(delta_beta)

map_beta -= np.mean(map_beta) -  delta_beta

hp.mollview(map_beta, unit='$\\beta$',  title='Scaling Index beta_sync') 

plt.show() 


# Power spectrum from map (output)
#---------------------------------------------------

cl_betaSync_out = hp.anafast(map_beta, lmax=lmax)  
dl_betaSync_out = cl_betaSync_out / dlfac


# Compare input and output power spectra 
#---------------------------------------------------

plt.figure()
plt.semilogy(ells, dl_betaSync, 'b',label="input power spectrum")
plt.semilogy(ells, dl_betaSync_out, 'r',label="output power spectrum")
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell$')
plt.legend()
plt.grid()
plt.show()


