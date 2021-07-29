#! /usr/bin/env python

from astropy.io import fits

def rc(fname, ext=1):
    print ('Reading in extension '+str(ext)+' of FITS file "'+fname+'" and copying table\n')
    f=fits.open(fname)
    fd=f[ext].data
    fdc=fd.copy()
    return fdc

