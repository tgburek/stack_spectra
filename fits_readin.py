#! /usr/bin/env python

import os
from astropy.io import fits
from termcolor import colored

def rc(fname, ext=1):
    eop = fname.rfind('/') #eop = End Of Path; "rfind" finds last occurence of designated string

    if eop != -1:
        path = fname[:eop+1]
        file_only = fname[eop+1:]
    else:
        path = os.getcwd() + '/'
        file_only = fname

    print ('')
    print ('Reading in extension '+colored(ext, 'white')+' of FITS file '+colored(file_only, 'white')+' and copying table')
    print ('File located at: '+colored(path, 'white'))
    print ('')

    f   = fits.open(fname)
    fd  = f[ext].data
    fdc = fd.copy()

    return fdc

