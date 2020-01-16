import pysm
from pysm.nominal import models
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from noise_calc import Simons_Observatory_V3_SA_noise,Simons_Observatory_V3_SA_beams # /Users/susannaazzoni/Desktop/Software/BBPipe/examples/noise_calc.py

import warnings

warnings.simplefilter("ignore")

'''
--------------------------------------------------------------------------------
Generate components with different templates for spectral indices and amplitudes
--------------------------------------------------------------------------------
'''

#nside= 64
nside = 512

'''
# Select models for each component
'''
#d2 = models("d2", nside) # dust: modified black body model, spatially varying emissivity
d2 = models("d1", nside) # dust: single component mbb
s1 = models("s1", nside) # synchrotron: simple power law with no curved index (curvature = 0)
c1 = models("c1", nside) # cmb: lensed CMB realisation computed using Taylens

'''
# Modify models
'''
# Dust
A_dust_BB=5.0
EB_dust=2.  # ratio between B and E modes from Planck IX 2018, B_to_E = 0.5, i.e. E_to_B=2
alpha_dust_EE=-0.42 # spectral tilt from Planck IX 2018, alpha = -0.42
alpha_dust_BB=-0.42
nu0_dust=353. #corresponds to nu_0_P' : 353. # Set as default for d2
beta_dust = 1.59 # spectral index and temperature from Planck IX 2018, beta = 1.53, T=19.6 K
temp_dust = 19.6

# Sync
A_sync_BB=2.0
EB_sync=2.
alpha_sync_EE=-0.6
alpha_sync_BB=-0.4
nu0_sync=23. #nu_0_P # Set as default
beta_sync=-3. # spectral index 

def fcmb(nu):
    x=0.017608676067552197*nu
    ex=np.exp(x)
    return ex*(x/(ex-1))**2

A_sync_BB = A_sync_BB * fcmb(nu0_sync)**2
A_dust_BB = A_dust_BB * fcmb(nu0_dust)**2

'''
# Calculate power spectrum
'''
print("Calculating power spectra")

# Gaussian power spectrum : A*((ls+0.001)/80.)**alpha

lmax = 3*nside-1
ells = np.arange(lmax+1)

dlfac=2*np.pi/(ells*(ells+1.)); dlfac[0]=1


# Dust
dl_dust_bb = A_dust_BB * ((ells+0.001) / 80.)**alpha_dust_BB # + 0.001 added to avoid divide by zero 
dl_dust_ee = EB_dust * A_dust_BB * ((ells+0.001) / 80.)**alpha_dust_EE
cl_dust_bb = dl_dust_bb * dlfac
cl_dust_ee = dl_dust_ee * dlfac
cl_dust_tt = 0 * cl_dust_bb
cl_dust_tb = 0 * cl_dust_bb
cl_dust_eb = 0 * cl_dust_bb
cl_dust_te = 0 * cl_dust_bb

# Sync
dl_sync_bb = A_sync_BB * ((ells+0.001) / 80.)**alpha_sync_BB 
dl_sync_ee = EB_sync * A_sync_BB * ((ells+0.001) / 80.)**alpha_sync_EE
cl_sync_bb = dl_sync_bb * dlfac
cl_sync_ee = dl_sync_ee * dlfac
cl_sync_tt = 0 * cl_sync_bb
cl_sync_tb = 0 * cl_sync_bb
cl_sync_eb = 0 * cl_sync_bb
cl_sync_te = 0 * cl_sync_bb

# CMB
l,dtt,dee,dbb,dte=np.loadtxt("/mnt/zfsusers/susanna/camb_lens_nobb.dat",unpack=True)

#select first ell from 0 not 1
l=l.astype(int)
msk=l<=lmax
l=l[msk]

dltt=np.zeros(len(ells)); dltt[ells]=dtt[ells]
dlee=np.zeros(len(ells)); dlee[ells]=dee[ells]
dlbb=np.zeros(len(ells)); dlbb[ells]=dbb[ells]
dlte=np.zeros(len(ells)); dlte[ells]=dte[ells]  
cl_cmb_bb=dlbb*dlfac
cl_cmb_ee=dlee*dlfac
cl_cmb_tt = 0 * cl_cmb_bb
cl_cmb_tb = 0 * cl_cmb_bb
cl_cmb_eb = 0 * cl_cmb_bb
cl_cmb_te = 0 * cl_cmb_bb

'''
Write cls outputs to file 
'''

prefix_out="/mnt/extraspace/susanna/SO/PySM-test-outputs/sim3_d1s1_outp"

np.savetxt(prefix_out + "/cls_cmb.txt",np.transpose([ells, cl_cmb_ee, cl_cmb_bb, cl_cmb_tt]))
np.savetxt(prefix_out + "/cls_sync.txt",np.transpose([ells, cl_sync_ee, cl_sync_bb, cl_sync_tt]))
np.savetxt(prefix_out + "/cls_dust.txt",np.transpose([ells, cl_dust_ee, cl_dust_bb, cl_dust_tt]))


'''
# Generate amplitude maps with hp.synfast
'''
print("Generating amplitude maps")

# Dust
A_I_dust,A_Q_dust,A_U_dust = hp.synfast([cl_dust_tt, cl_dust_ee, cl_dust_bb, cl_dust_te],
                                        nside=nside, new=True) #since cl_tt = 0, A_I will be zeros, new=True parameter used to read right order of Cls
# Sync
A_I_sync,A_Q_sync,A_U_sync = hp.synfast([cl_sync_tt, cl_sync_ee, cl_sync_bb, cl_sync_te],
                                        nside=nside, new=True)
# cmb
A_I_cmb,A_Q_cmb,A_U_cmb = hp.synfast([cl_cmb_tt, cl_cmb_ee, cl_cmb_bb, cl_cmb_te],
                                     nside=nside, new=True)


'''
# Set the newly defined attributes in models
'''
# Dust
d2[0]['A_I'] = A_I_dust
d2[0]['A_Q'] = A_Q_dust
d2[0]['A_U'] = A_U_dust
d2[0]['spectral_index'] = beta_dust
d2[0]['temp'] = temp_dust * np.ones(d2[0]['temp'].size) #need array, no const value for temp with PySM

# Sync
s1[0]['A_I'] = A_I_sync
s1[0]['A_Q'] = A_Q_sync
s1[0]['A_U'] = A_U_sync
s1[0]['spectral_index'] = beta_sync

# cmb
c1[0]['A_I'] = A_I_cmb
c1[0]['A_Q'] = A_Q_cmb
c1[0]['A_U'] = A_U_cmb


'''
# Plot the new templates: (maps of Q U amplitudes) # RJ unit
'''



'''
# Plot I Q U maps of each components over total sky with PySM
'''

# Define configuration dictionaries for each component
# use previously defined models
sky_config = {
#'dust' : models("d2", nside),    
'dust' : d2,
'synchrotron' : s1,
'cmb' : c1}

# initialise Sky 
sky = pysm.Sky(sky_config)

# components for array of frequencies
nu = np.array([27., 39., 93., 145., 225., 280.]) # elements [0,1,2,3,4,5]
dust = sky.dust(nu)
sync = sky.synchrotron(nu)
cmb = sky.cmb(nu)


#-----------------------
# Plot

#-----------------------


'''
# Add instrumental effects
'''

print("Adding instrumental effects")

# import bandpasses
# bpass centred at frequency freqs
freqs_LF1, bpass_LF1 = np.loadtxt("/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/data/LF/LF1.txt",unpack=True) 
N_freqs_LF1 = len(freqs_LF1)
freqs_LF2, bpass_LF2 = np.loadtxt("/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/data/LF/LF2.txt",unpack=True)
freqs_MF1, bpass_MF1 = np.loadtxt("/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/data/MF/MF1.txt",unpack=True)
freqs_MF2, bpass_MF2 = np.loadtxt("/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/data/MF/MF2.txt",unpack=True)
freqs_UHF1, bpass_UHF1 = np.loadtxt("/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/data/UHF/UHF1.txt",unpack=True)
freqs_UHF2, bpass_UHF2 = np.loadtxt("/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/data/UHF/UHF2.txt",unpack=True)


# PySM currenlty only passes gaussian beams
# so we input the sigma and it will calculate bl automatically
# sigmaFWHM = lambda*1.22/D [rad], sigmaG = sigmaFWHM/2.355 [rad], 
# beam in arcmin = (...)Rad × (60 × 180)/π
beams=Simons_Observatory_V3_SA_beams() #already given in arcmin

# use only for beam and bpass, add noise and mask later
# Pysm works on total sky maps --> no mask at this stage
# we want the noise not convolved with beam, Pysm computes noise convolved --> no noise at this stage
instrument_config = {
    'nside' : nside,
    # 'frequencies' : #Expected in GHz # not needed if use_bpass true
    'use_smoothing' : True,
    'beams' : beams, #Expected beam fwhm in arcmin #Only used if use_smoothing is True
    'add_noise' : False,
    'sens_I' : None, #Expected in units uK_RJ #Only used if add_noise is True
    'sens_P' : None,  #channel sensitivities in uK_CMB amin #Only used if add_noise is True
    'noise_seed' : 1234,
    'use_bandpass' : True,
    'channels' : [(freqs_LF1, bpass_LF1), (freqs_LF2, bpass_LF2), #frequencies and weights of channels to be calculated as a list of tuples
                  (freqs_MF1, bpass_MF1), (freqs_MF2, bpass_MF2), # [(frequencies_1, weights_1), (frequencies_2, weights_2) ...]
                  (freqs_UHF1, bpass_UHF1), (freqs_UHF2, bpass_UHF2)], 
    'channel_names' : ['LF1', 'LF2', 'MF1', 'MF2', 'UHF1', 'UHF2'],
    'output_units' : 'uK_CMB',
    'output_directory' :"/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples",
    'output_prefix' : 'test',
    'pixel_indices' : None, # added to dictionary for partial sky
}


sky = pysm.Sky(sky_config)

# Integration of the signal over an arbitrary bandpass, smoothing with a Gaussian beam, and the addition of
# Gaussian white noise. These are all done using the pysm.pysm.Instrument object
# Here used for beam and bpass
instrument = pysm.Instrument(instrument_config)

print("Writing T,Q,U maps with instrumental effects")
# This will write maps of (T, Q, U) as observed at the given frequencies with the given instrumental effect, at each band
instrument.observe(sky)

# Read map of sum of components including instrum effects at different bpass
# Create map of full sky
# Corresponds to maps_freq in generate_SO_maps
sky_maps = [] #empty array
band_nm = ["L", "M", "UH"] # bandpasses names 
for n in range(len(band_nm)): 
    #"%dF" % (band_nm[n]) NO, used for numbers
    for i in range(2): # 2 frequencies per bpass name (LF, MF, UHF)
        print ("test_bandpass_{}F%d_total_nside%04d.fits" 
               .format(band_nm[n]) # "{}".format(..[n]..) used for strings, select the bpass name
               % (i+1 , nside)) # "%d" %(..i) used for numbers, start i from 1 not zero
        mp = hp.read_map("test_bandpass_{}F%d_total_nside%04d.fits".format(band_nm[n]) % (i+1 , nside), 
                         field=[1,2]) #field = [T,Q,U] = [0,1,2], only want Q,U
#        mp = hp.read_map("test_bandpass_{}F%d_total_nside0064.fits".format(band_nm[n]) % (i+1))
        sky_maps.append(mp)
sky_maps = np.array(sky_maps)

print(sky_maps.shape)

nfreqs  = len(nu)
npix= hp.nside2npix(nside) # number of pixels for the given nside.


hp.write_map(prefix_out+"/sky_sign_inst.fits", sky_maps.reshape([nfreqs*2,npix]) , overwrite=True) 
#hp.write_map(prefix_out+"/sky_prova_inst.fits", sky_maps.reshape([nfreqs**2,npix]) , overwrite=True)
#hp.write_map(prefix_out+"/sky_sign_inst.fits", sky_maps, overwrite=True) 


# Display maps with instrumental effects                                                                                                  

'''
for q,u in sky_maps:
    hp.mollview(q, title="Q bpass")
    hp.mollview(u, title="U bpass")
plt.show()
''' 

'''
# Calculate power spectrum ClBB with anafast at lowest, mid, and high frequency including instrumental effects
'''
print("Calculating power spectrum")

mpU_LF1 = hp.read_map("test_bandpass_LF1_total_nside0512.fits", field=[2])
#mpQ_LF1 = hp.read_map("test_bandpass_LF1_total_nside0512.fits", field=[1])                                       
cl_LF1 = hp.anafast(mpU_LF1)

ells_sim = np.arange(len(cl_LF1))

mpU_MF1 = hp.read_map("test_bandpass_MF1_total_nside0512.fits", field=[2])
#mpQ_MF1 = hp.read_map("test_bandpass_MF1_total_nside0512.fits", field=[1])                                       
cl_MF1 = hp.anafast(mpU_MF1)

mpU_UHF2 = hp.read_map("test_bandpass_UHF2_total_nside0512.fits", field=[2])
cl_UHF2 = hp.anafast(mpU_UHF2)

# Write Cls^BB maps
hp.write_cl("clBB_LF1_inst.fits", cl_LF1, overwrite=True)
hp.write_cl("clBB_MF1_inst.fits", cl_MF1, overwrite=True)
hp.write_cl("clBB_HF2_inst.fits", cl_UHF2, overwrite=True)

# Compare ClBB at LF1, MF1, UHF1 for simulation vs given data                                                     
plt.figure()                                                                                                     
plt.loglog(ells_sim, cl_LF1, "r--", label = "LF - simulation")
plt.loglog(ells_sim, cl_MF1, "g--", label = "MF - simulation")
plt.loglog(ells_sim, cl_UHF2, "b--", label = "UHF - simulation")
#plt.legend(fontsize = 8, ncol = 2)
#plt.xlabel("$\ell$", fontsize = 15)
#plt.ylabel("$C_{\ell}^{BB}$", fontsize = 8)
#plt.show()                  


# Plot total sky power spectrum (which contains synchrotron, dust and cmb) for the given data
# nu = np.array([27., 39., 93., 145., 225., 280.])                                
data=np.load("c_ells_sky.npz")
ls_da = data['ls']
cls_ee_da = data['cls_ee']
cls_bb_da = data['cls_bb']
# cls_bb_da.size : array with size [number_of_frequencies, number_of_frequencies, number_of_ells]

#plt.figure()
# Autocorrelated plots
# 27 x 27                                                                                                           
plt.plot(ls_da, cls_bb_da[0, 0, :],"r-", label = "LF-da")
#plt.plot(ells, Cl_inst[0,0,:], "r--", label = "LF-simulation")
# 93 x 93                                                                                                                                             
plt.plot(ls_da, cls_bb_da[2, 2, :],"g-", label = "MF-da")
#plt.plot(ells, Cl_inst[2,2,:], "g--", label = "MF-simulation")
# 280 x 280                                                                                                                                           
plt.plot(ls_da, cls_bb_da[-1, -1, :],"b-", label = "UHF-da") #-1 gives you the last one
#plt.plot(ells, Cl_inst[-1,-1,:], "b--", label = "UHF-simulation")
plt.legend(fontsize = 8, ncol = 2)
plt.xlabel("$\ell$", fontsize = 15)
plt.ylabel("$C_{\ell}^{BB}$", fontsize = 15)

plt.show()


#exit()



'''
--------------------------------------------------------------------------------
# Generate mask
--------------------------------------------------------------------------------
'''

npix= hp.nside2npix(nside) # number of pixels for the given nside.

nhits=hp.ud_grade(hp.read_map("norm_nHits_SA_35FOV.fits",  verbose=False),nside_out=nside)

'''
                         field=[1,2]) #field = [T,Q,U] = [0,1,2], only want Q,U
'''
nhits/=np.amax(nhits) # equiv to nhits = nhits / np.amax(nhits) #np.amax --> max of array nhits
fsky=np.mean(nhits) # mean value of array nhits


'''
--------------------------------------------------------------------------------
Generate noise maps and splits
--------------------------------------------------------------------------------
'''
# set to zero everything outside what observed in mask 
nhits_binary=np.zeros_like(nhits) # array of zeros w/ same length as nhits
inv_sqrtnhits=np.zeros_like(nhits) # //
inv_sqrtnhits[nhits>1E-3]=1./np.sqrt(nhits[nhits>1E-3]) #inv_sqrtnhits = 1/(Nhits)^0.5 for nhits > 10^-3
nhits_binary[nhits>1E-3]=1 

# set parameters Simons_Observatory_V3_SA_noise()
ylf=1
nside=nside
sens=1
knee=1
nsplits=4 #TODs divided in 4 splits

#Add noise
nell=np.zeros([nfreqs,lmax+1])
_,nell[:,2:],_=Simons_Observatory_V3_SA_noise(sens,knee,ylf,fsky,lmax+1,1,include_kludge=False)

# Noise is convolved with beam in PySM
# beams=Simons_Observatory_V3_SA_beams() were given in arcmin
# We want noise deconvolved from beam:
for i,(n,b) in enumerate(zip(nell,beams)):
    sig = b * np.pi/180./60/2.355 # turns beam b into sigma, b*pi/180/60 transform beam into rad, 
    bl = np.exp(-sig**2*ells*(ells+1)) # beam in Fourier space
    n *= bl # Remove beam
    n[:2]=n[2] # Fill to ell=0 


# Generating noise maps
noimaps = np.zeros([nsplits,nfreqs,2,npix]) # only noise component changes

for s in range(nsplits):
    for f in range(nfreqs):
        noimaps[s,f,0,:]=hp.synfast(nell[f] * nsplits, nside, pol=False, verbose=False, new=True) * inv_sqrtnhits
        noimaps[s,f,1,:]=hp.synfast(nell[f] * nsplits, nside, pol=False, verbose=False, new=True) * inv_sqrtnhits
#noi_coadd = np.mean(noimaps, axis=0)


print(sky_maps.shape,noimaps.shape)
print(len(sky_maps[0]), len(noimaps[0]), len(nhits))


'''
print((sky_maps[:,:,:]).shape, (noimaps[s,:,:,:]).shape)
print((sky_maps[f,:,:]).shape, (noimaps[s,f,:,:]).shape)
print((sky_maps[f,1,:]).shape, (noimaps[s,f,1,:]).shape)

sky_maps = sky_maps[f,1,:]
noimaps = noimaps[s,f,1,:]

print(len(sky_maps[0]), len(noimaps[0]), len(nhits))

exit()

#print((np.transpose(sky_maps)).shape)

sky_maps = np.transpose(sky_maps)
noimaps = np.transpose(noimaps)

exit()
'''
# Added

'''
# Beam-convolution                                                                                             
for f,b in enumerate(beams):
    fwhm = b * np.pi/180./60.
    for i in [0,1]:
        sky_maps[f,i,:] = hp.smoothing(sky_maps[f,i,:], fwhm=fwhm, verbose=False)
'''

for s in range(nsplits):
    hp.write_map(prefix_out+"/obs_split%dof%d.fits.gz" % (s+1, nsplits),
                 ((sky_maps[:,:,:]+noimaps[s,:,:,:])*nhits_binary).reshape([nfreqs*2,npix]),
                 overwrite=True)

for f in range(nfreqs):
    np.savetxt(prefix_out + "/cls_noise_b%d.txt" % (f+1),np.transpose([ells,nell[f]]))


# Write splits list
f=open(prefix_out+"/splits_list.txt","w")
stout=""
for i in range(nsplits):
    stout += prefix_out+'/obs_split%dof%d.fits.gz\n' % (i+1, nsplits)
f.write(stout)
f.close()
              

#plt.show()





