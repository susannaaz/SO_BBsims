import pysm
from pysm.nominal import models
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from noise_calc import Simons_Observatory_V3_SA_noise,Simons_Observatory_V3_SA_beams
import warnings
warnings.simplefilter("ignore")

#--------------------------------------------------------
## Generate components with different templates for spectral indices and amplitudes
print("Generate maps of components")

nside = 512

## Select models for each component
d2 = models("d2", nside) # dust: modified black body model
s1 = models("s1", nside) # synchrotron: simple power law with no curved index
c1 = models("c1", nside) # cmb: lensed CMB realisation computed using Taylens

## Modify models

# Dust
A_dust_BB=5.0
EB_dust=2.  # ratio between B and E modes from Planck IX 2018, B_to_E = 0.5
alpha_dust_EE=-0.42 # spectral tilt from Planck IX 2018, alpha = -0.42
alpha_dust_BB=-0.42
nu0_dust=353. #corresponds to nu_0_P' : 353. # Set as default for d2
beta_dust = 1.59 # spectral index and temperature from Planck IX 2018, beta = 1.53, T=19.6 K
#beta_dust = read_map(template('beta_mean1p59_std0p2.fits'), nside, field=0, pixel_indices=pixel_indices, mpi_comm=mpi_comm) #Varying w/ PySM model
temp_dust = 19.6

prefix_in='/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/template_PySM/'
# Sync
A_sync_BB=2.0
EB_sync=2.
alpha_sync_EE=-0.6
alpha_sync_BB=-0.4
nu0_sync=23. #nu_0_P # Set as default
beta_sync=-3. # spectral index 
#beta_sync=-3. # spectral index #Const                                                                                   
#beta_sync = hp.ud_grade(hp.read_map('synch_beta.fits', verbose=False,  field=[0]), nside_out=nside)  #Varying with PySM map           #beta_sync = hp.ud_grade(hp.read_map(prefix_in+'map_beta_sync.fits', verbose=False), nside_out=nside)  #Varying with new map_beta_sync
beta_sync = hp.ud_grade(hp.read_map(prefix_in+'map_beta_sync_fin.fits', verbose=False), nside_out=nside)  #Varying with new map_beta_sync_fin

def fcmb(nu):
    x=0.017608676067552197*nu
    ex=np.exp(x)
    return ex*(x/(ex-1))**2

A_sync_BB = A_sync_BB * fcmb(nu0_sync)**2
A_dust_BB = A_dust_BB * fcmb(nu0_dust)**2

## Calculate power spectrum
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
# select first ell from 0 not 1
l=l.astype(int)
msk=l<=lmax
l=l[msk]
#
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

## Write cls outputs to file 
prefix_out="/mnt/extraspace/susanna/SO/PySM-test-outputs/sim3_d1s1_outp"
np.savetxt(prefix_out + "/cls_cmb.txt",np.transpose([ells, cl_cmb_ee, cl_cmb_bb, cl_cmb_tt]))
np.savetxt(prefix_out + "/cls_sync.txt",np.transpose([ells, cl_sync_ee, cl_sync_bb, cl_sync_tt]))
np.savetxt(prefix_out + "/cls_dust.txt",np.transpose([ells, cl_dust_ee, cl_dust_bb, cl_dust_tt]))

## Generate amplitude maps with hp.synfast
# Dust
A_I_dust,A_Q_dust,A_U_dust = hp.synfast([cl_dust_tt, cl_dust_ee, cl_dust_bb, cl_dust_te],
                                        nside=nside, new=True)
# Sync
A_I_sync,A_Q_sync,A_U_sync = hp.synfast([cl_sync_tt, cl_sync_ee, cl_sync_bb, cl_sync_te],
                                        nside=nside, new=True)
# cmb
A_I_cmb,A_Q_cmb,A_U_cmb = hp.synfast([cl_cmb_tt, cl_cmb_ee, cl_cmb_bb, cl_cmb_te],
                                     nside=nside, new=True)

## Set the newly defined attributes in models
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

## Define configuration dictionaries for each component
sky_config = {
'dust' : d2,
'synchrotron' : s1,
'cmb' : c1}

## Initialise Sky 
sky = pysm.Sky(sky_config)

## Components for array of frequencies
nu = np.array([27., 39., 93., 145., 225., 280.]) 
dust = sky.dust(nu)
sync = sky.synchrotron(nu)
cmb = sky.cmb(nu)

#--------------------------------------------------------
print("Adding instrumental effects")

freqs_LF1, bpass_LF1 = np.loadtxt("/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/data/LF/LF1.txt",unpack=True) 
N_freqs_LF1 = len(freqs_LF1)
freqs_LF2, bpass_LF2 = np.loadtxt("/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/data/LF/LF2.txt",unpack=True)
freqs_MF1, bpass_MF1 = np.loadtxt("/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/data/MF/MF1.txt",unpack=True)
freqs_MF2, bpass_MF2 = np.loadtxt("/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/data/MF/MF2.txt",unpack=True)
freqs_UHF1, bpass_UHF1 = np.loadtxt("/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/data/UHF/UHF1.txt",unpack=True)
freqs_UHF2, bpass_UHF2 = np.loadtxt("/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples/data/UHF/UHF2.txt",unpack=True)

## PySM currenlty only passes gaussian beams so we input the sigma and it will calculate bl automatically
beams=Simons_Observatory_V3_SA_beams() #already given in arcmin

## Use only for beam and bpass, add noise and mask later (PySM works with total sky maps, noise convolved with beams) 
instrument_config = {
    'nside' : nside,
    'use_smoothing' : True,
    'beams' : beams, #Expected beam fwhm in arcmin 
    'add_noise' : False,
    'sens_I' : None, #Expected in units uK_RJ #Only used if add_noise is True
    'sens_P' : None,  #channel sensitivities in uK_CMB amin #Only used if add_noise is True
    'noise_seed' : 1234,
    'use_bandpass' : True,
    'channels' : [(freqs_LF1, bpass_LF1), (freqs_LF2, bpass_LF2), #frequencies and weights of channels to be calculated as a list of tuples
                  (freqs_MF1, bpass_MF1), (freqs_MF2, bpass_MF2), 
                  (freqs_UHF1, bpass_UHF1), (freqs_UHF2, bpass_UHF2)], 
    'channel_names' : ['LF1', 'LF2', 'MF1', 'MF2', 'UHF1', 'UHF2'],
    'output_units' : 'uK_CMB',
    'output_directory' :"/mnt/zfsusers/susanna/PySM-tests2/BBPipe/examples",
    'output_prefix' : 'test',
    'pixel_indices' : None, # added to dictionary for partial sky
}

sky = pysm.Sky(sky_config)

## Integrate the signal over bandpass and smooth with a Gaussian beam
instrument = pysm.Instrument(instrument_config)

## Write maps of (T, Q, U)
instrument.observe(sky)

## Read map of sum of components including instrum effects at different bpass
sky_maps = [] 
band_nm = ["L", "M", "UH"] # bandpasses names 
for n in range(len(band_nm)): 
    for i in range(2):
        print ("test_bandpass_{}F%d_total_nside%04d.fits" 
               .format(band_nm[n]) 
               % (i+1 , nside)) 
        mp = hp.read_map("test_bandpass_{}F%d_total_nside%04d.fits".format(band_nm[n]) % (i+1 , nside), 
                         field=[1,2]) #only want Q,U
        sky_maps.append(mp)
sky_maps = np.array(sky_maps)

nfreqs  = len(nu)
npix= hp.nside2npix(nside)

## Write full sky  map
hp.write_map(prefix_out+"/sky_sign_inst.fits", sky_maps.reshape([nfreqs*2,npix]) , overwrite=True) 

#--------------------------------------------------------
print("Generating mask")

npix= hp.nside2npix(nside) 
nhits=hp.ud_grade(hp.read_map("norm_nHits_SA_35FOV.fits",  verbose=False),nside_out=nside)
nhits/=np.amax(nhits) 
fsky=np.mean(nhits) 

#--------------------------------------------------------
print("Generating noise maps and splits")

## Set to zero everything outside what observed in mask 
nhits_binary=np.zeros_like(nhits) 
inv_sqrtnhits=np.zeros_like(nhits)
inv_sqrtnhits[nhits>1E-3]=1./np.sqrt(nhits[nhits>1E-3]) 
nhits_binary[nhits>1E-3]=1 

## Set parameters Simons_Observatory_V3_SA_noise()
ylf=1
nside=nside
sens=1
knee=1
nsplits=4 

## Add noise
nell=np.zeros([nfreqs,lmax+1])
_,nell[:,2:],_=Simons_Observatory_V3_SA_noise(sens,knee,ylf,fsky,lmax+1,1,include_kludge=False)

## Noise is convolved with beam in PySM, we now deconvolve it
## beams=Simons_Observatory_V3_SA_beams() were given in arcmin
for i,(n,b) in enumerate(zip(nell,beams)):
    sig = b * np.pi/180./60/2.355 # b*pi/180/60 transform beam into rad
    bl = np.exp(-sig**2*ells*(ells+1)) 
    n *= bl # Remove beam
    n[:2]=n[2] # Fill to ell=0 

## Generating noise maps
noimaps = np.zeros([nsplits,nfreqs,2,npix]) # only noise component changes

for s in range(nsplits):
    for f in range(nfreqs):
        noimaps[s,f,0,:]=hp.synfast(nell[f] * nsplits, nside, pol=False, verbose=False, new=True) * inv_sqrtnhits
        noimaps[s,f,1,:]=hp.synfast(nell[f] * nsplits, nside, pol=False, verbose=False, new=True) * inv_sqrtnhits

for s in range(nsplits):
    hp.write_map(prefix_out+"/obs_split%dof%d.fits.gz" % (s+1, nsplits),
                 ((sky_maps[:,:,:]+noimaps[s,:,:,:])*nhits_binary).reshape([nfreqs*2,npix]),
                 overwrite=True)

for f in range(nfreqs):
    np.savetxt(prefix_out + "/cls_noise_b%d.txt" % (f+1),np.transpose([ells,nell[f]]))

## Write splits list
f=open(prefix_out+"/splits_list.txt","w")
stout=""
for i in range(nsplits):
    stout += prefix_out+'/obs_split%dof%d.fits.gz\n' % (i+1, nsplits)
f.write(stout)
f.close()
              






