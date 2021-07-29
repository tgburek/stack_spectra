import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import join,isfile
import numpy as np
from astropy.io import ascii,fits
import spectres as res
from tkinter import Tk
from tkinter.filedialog import askdirectory
from IPython import embed
from astropy import cosmology
Tk().withdraw()
path = askdirectory(title="Select Folder with files to stack")
os.chdir(path)

files = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(),f))]
#embed()
table = ascii.read('Full_table.txt')
spec = []
H0 = 70.0
Om0 = 0.3
Ob0 = 0.05
Tcmb0 = 2.725
cosmo = cosmology.FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, Tcmb0=Tcmb0)
flux_to_lum = lambda flux, lum_dist: np.multiply(np.multiply(flux, np.square(lum_dist)), 4.*np.pi)
cm_in_Mpc = 3.086e24
for f in files:
     for e,name in enumerate(table['col1']):
         st = f.split('_')[0]
         if ((st == name) or ((name == '873_382') and (st.find('873') != -1))) and (float(table['col52'][e]) >= 0):
             print(f)
             print(name)
             data = fits.open(f)
             data = data[1].data
             try: err = data['sig']
             except: err = 1/np.sqrt(data['ivar'])
             z = float(table['col12'][e])
             obj_lum_dist_Mpc = cosmo.luminosity_distance(z).value
             obj_lum_dist_cm  = obj_lum_dist_Mpc * cm_in_Mpc
             flux = data['flux']
             wl = data['wave']
             mask = (wl > 3100) & (wl < 5550)
             err = 1e-17*err[mask]/float(table['col52'][e])
             flux = 1e-17*flux[mask]/float(table['col52'][e])#replace x with column containing value for normalization
             converted_meas = flux_to_lum(flux, obj_lum_dist_cm)
             conv_meas_errs = flux_to_lum(err, obj_lum_dist_cm)
             converted_meas = np.multiply(converted_meas, 1.+ z)
             conv_meas_errs = np.multiply(conv_meas_errs, 1.+ z)
             wl = wl[mask]
             rwl = wl/(z+1)
             spec.append((converted_meas,rwl,conv_meas_errs))
             #TODO: add in normalization. Likely grab from photometry for ~1500AA flux
resdata = []
new_grid = np.arange(720,2135,2.18)#range is z~1.6-3.3 using 3100-5500
for s in spec:
      resdata.append(res.spectres(new_grid,s[1],s[0],fill=0,spec_errs=s[2]))
#embed()   
for r in resdata:
    r[1][r[1]==0] = np.max(r[1])*1e1
weights = 1/(np.transpose(resdata,(1,0,2))[1])**2
weights[weights == np.inf] = 3
embed()
Flux = np.average(np.transpose(resdata,(1,0,2))[0],axis=0,weights=weights)
print(np.shape(resdata))
plt.plot(new_grid,Flux)
plt.show()
