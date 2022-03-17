#!/usr/bin/env python

import os
import re
import sys
import time
import numpy as np
import pandas as pd
import fits_readin as fr
from astropy.io import fits
from termcolor import colored
from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentDefaultsHelpFormatter

print

class HelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    pass

parser = ArgumentParser(formatter_class=HelpFormatter, description=(

"""This script applies stellar absorption corrections to already-fit Balmer emission-line luminosities.
The stellar absorption corrections should be given as EW_abs / EW_obs."""

))

parser.add_argument('Eline_Table', \
                    help='The filename of the FITS table containing the already-fit Balmer emission-line luminosities')

parser.add_argument('Abs_Corr_Table', \
                    help='The filename of the text file with the Balmer stellar absorption corrections\n'
                         '(should be located in current working directory)')

parser.add_argument('Normalizing_Eline', choices=['OIII5007', 'H-alpha'], \
                    help='The emission-line name of the line used to normalize the spectra during stacking')

parser.add_argument('Stacking_Method', choices=['median', 'average'], \
                    help='The method with which the spectra in the sample were stacked')

parser.add_argument('Uncertainty', choices=['bootstrap', 'statistical'], \
                    help='How the uncertainty spectrum was calculated\n'
                         '(including cosmic variance or just statistically)')

parser.add_argument('Balmer_Lines', nargs='+', \
                    help='The Balmer emission lines to correct\n'
                         '(ex. H-beta H-alpha)')


args = parser.parse_args()

eline_table = args.Eline_Table
corr_table  = args.Abs_Corr_Table
norm_eline  = args.Normalizing_Eline   
stack_meth  = args.Stacking_Method
uncert      = args.Uncertainty
blines      = args.Balmer_Lines


class Logger(object):
    def __init__(self, logname='log', mode='a'):            
        self.terminal = sys.stdout
        self.logname  = logname+'_'+time.strftime('%m-%d-%Y')+'.log'
        self.mode = mode
        self.log = open(self.logname, self.mode)

    def write(self, message):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        self.terminal.write(message)
        self.log.write(ansi_escape.sub('', message))

    def flush(self):
        pass


percent_change = lambda old_val, new_val: ((new_val - old_val) / old_val) * 100.


cwd = os.getcwd()

logname_base = cwd + '/logfiles/correcting_balmer_elines_for_stellar_abs_'+norm_eline+'_'+stack_meth+'_'+uncert
sys.stdout   = Logger(logname=logname_base, mode='w')

print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print '- '+colored(('This script will correct the Balmer emission-line luminosities for stellar absorption.\n'
                    'This will be done by multiplying the observed luminosity by (1 + EW_abs/EW_obs)'
                   ), 'cyan', attrs=['bold']),
print ' -'
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print

output_fname = eline_table[:-5] + '_sa_corr.fits'

eline_fpath = cwd + '/uncertainty_'+uncert+'_fitting_analysis/'+norm_eline+'_norm/fw_full_spectrum/'
output_path = cwd + '/uncertainty_'+uncert+'_fitting_analysis/'+norm_eline+'_norm/'

etable = fr.rc(eline_fpath + eline_table)

sa_corrs = pd.read_csv(corr_table, comment='#', delim_whitespace=True, names=['Balmer_Lines', 'SA_Corrections'], \
                       index_col='Balmer_Lines', dtype={'Balmer_Lines': np.string_, 'SA_Corrections': np.float64})

print
print 'Table of Balmer stellar absorption correction factors:'
print
print sa_corrs
print

print 'The Balmer emission lines to have their luminosities corrected:'
print colored(blines, 'green')
print

for bline in blines:

    etable_idx = int(np.where(np.isnan(etable[bline+'_Total_Lum']) == False)[0])

    obs_emiss  = etable[bline+'_Total_Lum'][etable_idx]
    obs_esig   = etable[bline+'_Total_Lum_sig'][etable_idx]

    SAcorr = sa_corrs.loc[bline, 'SA_Corrections']

    print 'The luminosity of ' + colored(bline, 'magenta') + ' before SA correction is: ',
    print colored('%.4e' % obs_emiss, 'green') + ' +/- ' + colored('%.4e' % obs_esig, 'green')
    print 'The correction (EW_abs/EW_obs) to be applied is: ' + colored('%.4f' % SAcorr, 'green')
    print

    corr_emiss = obs_emiss * (1.+SAcorr)
    corr_esig  = obs_esig  * (1.+SAcorr)

    print colored('The luminosity of ' + colored(bline, 'magenta') + ' after SA correction is: ', attrs=['bold']),
    print colored('%.4e' % corr_emiss, 'green') + ' +/- ' + colored('%.4e' % corr_esig, 'green')
    print 'This corresponds to a percentage change of ' + colored('%.2f' % percent_change(obs_emiss, corr_emiss), 'green'),
    print colored(' %', 'green')
    print

    etable[bline+'_Total_Lum'][etable_idx] = corr_emiss
    etable[bline+'_Total_Lum_sig'][etable_idx] = corr_esig

hdu = fits.BinTableHDU(data=etable)
hdu.writeto(output_path + output_fname, overwrite=True)

print '--> '+colored(output_fname, 'green')+' written at location: '
print colored(output_path, 'white')
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print 