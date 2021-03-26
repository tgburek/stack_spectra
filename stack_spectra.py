#! /usr/bin/env python

import os
import re
import sys
import time
import numpy as np
import pandas as pd
import fits_readin as fr
import stacking_functions as sf
import matplotlib.pyplot as plt
import seaborn as sns
import sns_setstyle
from glob import glob
from astropy.io import fits
from collections import OrderedDict
from termcolor import colored
from matplotlib.backends.backend_pdf import PdfPages
from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentDefaultsHelpFormatter

print

class HelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    pass


parser = ArgumentParser(formatter_class=HelpFormatter, description=(
    
"""STACK INDIVIDUAL ALREADY SLIT-LOSS-CORRECTED SPECTRA.

- Spectra will be shifted to the rest frame.
- Spectra will have their flux densities converted to luminosity densities.
- Spectra MAY be dust-corrected. This option is specified in the call to this script (Option present but not currently usable).
- Spectra will be normalized by an emission line given in the call to this script (Currently supported: OIII5007 and H-alpha).
- Spectra will be resampled onto a wavelength grid with spacing equal to that at the {median, average, weighted-average} sample redshift (by filter).
- Spectra will be combined via the method (median, average, weighted-average) given in the call to this script.
- Spectra will be multiplied by the {median, average, weighted-average} line luminosity corresponding to the emission line used for normalization.

FOR MORE INFO ON THE PROCEDURE IN THIS SCRIPT, SEE THE README (NOT YET CREATED)."""  ## Have not made the README yet

))


parser.add_argument('-t', '--SLC_Table', metavar='str', \
                    help='The FITS filename of the slit-loss-correction table for emission lines')

parser.add_argument('-d', '--Dust_Correct', action='store_true', \
                    help='If called, each individual spectrum will be dust-corrected (not currently supported)')

parser.add_argument('-s', '--Include_Stacks', action='store_true', \
                    help='If called, stacking sample will include previously made stacks\n(such as from multiply-imaged galaxies)')

parser.add_argument('Flux_Table', \
                    help='The FITS filename of the emission-line flux table')

parser.add_argument('Norm_ELine', choices=['OIII5007','H-alpha'], \
                    help='The emission line name of the line used to normalize each spectrum')

parser.add_argument('Stacking_Method', choices=['median','average','weighted-average'], \
                    help='The method with which the spectra will be stacked')

parser.add_argument('Stacking_Sample', \
                    help='The FITS file with the spectrum IDs for stacking')


args = parser.parse_args()

slc_cat    = args.SLC_Table
dust_corr  = args.Dust_Correct
inc_stacks = args.Include_Stacks
flux_cat   = args.Flux_Table
norm_eline = args.Norm_ELine
stack_meth = args.Stacking_Method
stack_samp = args.Stacking_Sample


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


def inner_perc_max(yaxis_vals, percentage=90.):
    excluded_percentage = 1. - (percentage/100.)
    end_exc_percentages = excluded_percentage / 2.

    max_idx_of_array = len(yaxis_vals) - 1

    exc_ind = range(int(round(max_idx_of_array * end_exc_percentages))) + \
              range(int(round(max_idx_of_array * (1. - end_exc_percentages))), len(yaxis_vals))

    inner_vals = np.delete(yaxis_vals, exc_ind)

    return max(inner_vals)



def band_stack_wave_range(min_wave_arr,max_wave_arr):
    max_of_min_arr = max(min_wave_arr)
    min_of_max_arr = min(max_wave_arr)

    min_band_stack_range = round(max_of_min_arr / 5.) * 5.

    if min_band_stack_range < max_of_min_arr:
        min_band_stack_range += 5.

    max_band_stack_range = round(min_of_max_arr / 5.) * 5.

    if max_band_stack_range > min_of_max_arr:
        max_band_stack_range -= 5.

    return min_band_stack_range, max_band_stack_range

  

def plot_resampled_spectra(axes_obj, x, y, y_err, eline_waves, color, linestyle, linewidth, alpha, label):
    axes_obj.step(x, y, where='mid', color=color[0], linestyle=linestyle[0], linewidth=linewidth[0], alpha=alpha[0], label=label[0])
    axes_obj.step(x, y_err, where='mid', color=color[1], linestyle=linestyle[1], linewidth=linewidth[1], alpha=alpha[1], label=label[1])

    within_plotted_wrange = np.where((eline_waves >= min(x)) & (eline_waves <= max(x)))[0]
    eline_waves_to_plot   = eline_waves[within_plotted_wrange]

    for wavelength in eline_waves_to_plot:
        axes_obj.axvline(x = wavelength, color=color[2], linestyle=linestyle[2], linewidth=linewidth[2], alpha=alpha[2])

    #ax.set_ylim([-0.07, 1.1*inner_perc_max(resampled_spectra['YJ']['New_Luminosities'][yj_idx], percentage=90.)])

    return axes_obj


cwd = os.getcwd()
    
sys.stdout = Logger(logname=cwd+'/logfiles/stacking_spectra_'+norm_eline+'_'+stack_meth, mode='w')


print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print colored(('This program will stack individual spectra that meet certain criteria\n'
               'and have entries in both the flux catalog and photometric catalog.\n'
               'THIS CODE IS IN DEVELOPMENT.'
              ), 'cyan',attrs=['bold'])
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print

print 'The path and current working directory are: ',colored(cwd,'green')

mask_path = cwd+'/fc_1d_spectra/'

print 'The path with the mask sub-directories is: ',colored(mask_path,'green')
print

mosfire_masks = sorted([x[len(mask_path):].rstrip('/') for x in glob(mask_path+'*/')])

print "The following masks will have any appropriate associated spectra stacked:"
print

for mask in mosfire_masks:
    print colored(mask,'green')

print

flux_table = fr.rc(flux_cat)
samp_table = fr.rc(stack_samp)

if slc_cat is not None:
    slc_table = fr.rc(slc_cat)

eline_list, eline_rwave = np.loadtxt('loi.txt', comments='#', usecols=(0,2), dtype='str', unpack=True)
eline_rwave = eline_rwave.astype(float)

##############################################################


stacking_sample = OrderedDict.fromkeys(['fpath','mask','id','filt'])

for key in stacking_sample.keys():
    stacking_sample[key] = np.array([])

for mask in mosfire_masks:

    slc_path  = mask_path + mask + '/error_spectra_corrected/slit_loss_corrected/'

    slc_files = sorted([x for x in os.listdir(slc_path) if 'fc.1d.esc.slc.txt' in x])

    for fname in slc_files:
        print 'Checking to see if '+colored(fname,'white')+' corresponds to a galaxy to be stacked: ',

        filt   = fname[len(mask)+1]
        id_num = fname[len(mask)+3:-18]

        if id_num in samp_table['ID'] and (id_num != '370' or (id_num == '370' and filt == 'H')):
        #if id_num in samp_table['ID'] and filt != 'H':
            stacking_sample['fpath'] = np.append(stacking_sample['fpath'], slc_path + fname)
            stacking_sample['mask']  = np.append(stacking_sample['mask'], mask)
            stacking_sample['id']    = np.append(stacking_sample['id'], id_num)
            stacking_sample['filt']  = np.append(stacking_sample['filt'], filt)
            print colored('True', 'green', attrs=['bold'])
            print

        else:
            print colored('False', 'red', attrs=['bold'])
            print
            

if inc_stacks == True:
    
    prior_stacks = sorted(glob(cwd + '/stacks/stacked_spectrum_*.txt'))

    for fname in prior_stacks:
        print 'Adding the previously made stack '+colored(fname,'white')+' to the stacking sample'
        print

        id_start = re.search(r'\d', fname).start()
        id_num   = fname[id_start : -4]

        if id_num == '1197_370':
            filt = fname[17]
        else:
            filt = fname[18]

        mask = samp_table['Mask'][samp_table['ID'] == id_num].rstrip()

        stacking_sample['fpath'] = np.append(stacking_sample['fpath'], fname)
        stacking_sample['mask']  = np.append(stacking_sample['mask'], mask)
        stacking_sample['id']    = np.append(stacking_sample['id'], id_num)
        stacking_sample['filt']  = np.append(stacking_sample['filt'], filt)
        

stacking_sample_DF = pd.DataFrame.from_dict(stacking_sample, orient='columns')

print stacking_sample_DF
    

exp_stack_sample_size = len(samp_table) - 1
gals_with_data_found  = len(stacking_sample['fpath']) / 3


print
print
print
print 'Number of galaxies that should be stacked: ', exp_stack_sample_size
print 'Number of galaxies with spectral data found: ', gals_with_data_found
print
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print

if exp_stack_sample_size != gals_with_data_found:
    raise ValueError(('The number of galaxies that should have their spectra stacked does not match the number of spectra found\n'
                      '(number of spectra found divided by 3 to account for the three filters considered)'))
#sys.exit()
##############################################################

if norm_eline == 'no-norm':
    line_of_interest = raw_input('Enter the emission line you are interested in eventually fitting (OIII5007 or H-alpha): ')
    mult_imgs = '_'.join(sorted(list(set(stacking_sample['id']))))
    pp_name   = 'restframe_lum_no-norm_spectra_'+mult_imgs+'.pdf'
else:
    line_of_interest = norm_eline
    pp_name = 'restframe_lum_normlum_spectra.pdf'


pp = PdfPages(cwd + '/intermed_spectrum_plots_'+norm_eline+'_'+stack_meth+'/' + pp_name)

sample_params = pd.DataFrame(index=range(len(stacking_sample['fpath'])), \
                             columns=['ID', 'Mask', 'Filter', 'Min_Wave', 'Max_Wave', 'Redshift', 'Redshift_Error', line_of_interest+'_Flux', line_of_interest+'_Lum', line_of_interest+'_Lum_Err']
                            )

seen_idmask = []


for i, file_path in enumerate(stacking_sample['fpath']):

    mask   = stacking_sample['mask'][i]
    id_num = stacking_sample['id'][i]
    filt   = stacking_sample['filt'][i]

    idx_in_samp_table = int(np.where(samp_table['ID'] == id_num)[0])
    
    slc_path = mask_path + mask + '/error_spectra_corrected/slit_loss_corrected/'

    if samp_table['Multiple_Images'][idx_in_samp_table] == False:

        fname = file_path[len(slc_path):]

    else:
        fname = file_path


    print colored('--> ','cyan',attrs=['bold'])+'Preparing spectrum in file '+colored(fname,'white')+' for resampling...'
    print
    print 'Mask: ', colored(mask,'green')
    print 'ID: ', colored(id_num,'green')
    print 'Filter: ', colored(filt,'green')
    print
    print 'Emission line with which the spectrum will be normalized: ', colored(norm_eline,'green')
    

    if samp_table['Multiple_Images'][idx_in_samp_table] == False:

        if stack_meth != 'weighted-average' and id_num == '370':

            z = samp_table['Weighted_z'][idx_in_samp_table]
            z_err = samp_table['Weighted_z_Sig'][idx_in_samp_table]
            eline_lum = samp_table[norm_eline+'_Lum'][idx_in_samp_table]
            eline_lum_error = samp_table[norm_eline+'_Lum_Sig'][idx_in_samp_table]

            print 'Measured emission-line luminosity (NOT dust-corrected): ',colored('%.5e' % eline_lum,'green')
            print
            
        else:

            idx_in_FT = int(np.where((flux_table['Mask'] == mask) & (flux_table['ID'] == id_num))[0])
        
            z = flux_table['Weighted_z'][idx_in_FT]
            z_err = flux_table['Weighted_z_Sig'][idx_in_FT]
            eline_flux = flux_table[line_of_interest+'_Flux'][idx_in_FT]
            eline_sig  = flux_table[line_of_interest+'_Sig'][idx_in_FT]

            print 'Measured emission-line flux (NOT dust-corrected): ',colored('%.5e' % eline_flux,'green')

            if slc_cat != None:
                idx_in_SLC = int(np.where((slc_table['Mask'] == mask) & (slc_table['ID_spec'] == id_num))[0])
                star_corr, obj_corr = slc_table[line_of_interest+'_Star_Slit'][idx_in_SLC], slc_table[line_of_interest+'_Obj_Slit'][idx_in_SLC]

                print colored('-> ','magenta')+'Slit-loss-correcting the flux of '+colored(line_of_interest,'green')
                print 'The star-based slit-loss-correction factor to be undone is: '+colored(star_corr,'green')
                print 'The object-based slit-loss-correction factor to be applied is: '+colored('%.5f' % obj_corr,'green')
                print 'The total correction factor will be: '+colored('%.5f' % (star_corr/obj_corr),'green')
                print

                eline_flux *= (star_corr / obj_corr)
                eline_sig  *= (star_corr / obj_corr)
    
            eline_lum, eline_lum_error = sf.Flux_to_Lum(eline_flux, eline_sig, redshift = z, densities=False, verbose=True)


        obs_waves, fluxes, flux_errs = np.loadtxt(file_path, comments='#', usecols=(0,1,2), dtype='float', unpack=True)
        rest_waves                   = sf.shift_to_rest_frame(obs_waves, redshift = z)
        luminosities, lum_errs       = sf.Flux_to_Lum(fluxes, flux_errs, redshift = z,  densities=True, verbose=True)


    else:
        rest_waves, luminosities, lum_errs = np.loadtxt(file_path, comments='#', usecols=(0,1,2), dtype='float', unpack=True)

        nans = np.where(np.isnan(luminosities) == True)[0]

        rest_waves   = np.delete(rest_waves, nans)
        luminosities = np.delete(luminosities, nans)
        lum_errs     = np.delete(lum_errs, nans)

        z = samp_table['Weighted_z'][idx_in_samp_table]
        z_err = samp_table['Weighted_z_Sig'][idx_in_samp_table]
        eline_lum = samp_table[norm_eline+'_Lum'][idx_in_samp_table]
        eline_lum_error = samp_table[norm_eline+'_Lum_Sig'][idx_in_samp_table]

        print 'Measured emission-line luminosity (NOT dust-corrected): ',colored('%.5e' % eline_lum,'green')
        print

        ## To make plotting and file writing less convoluted

        obs_waves = sf.shift_to_obs_frame(rest_waves, redshift=z)
        fluxes, flux_errs = sf.Flux_to_Lum(luminosities, lum_errs, redshift = z, Lum_to_Flux=True, densities=True)
        eline_flux, eline_sig = sf.Flux_to_Lum(eline_lum, eline_lum_error, redshift = z, Lum_to_Flux=True, densities=False)

    if norm_eline != 'no-norm':
        lum_norm, lum_norm_errs = sf.normalize_spectra(luminosities, norm_eline, eline_lum, int_lum_errs=lum_errs, int_eline_lum_err=eline_lum_error) 

    print colored('-> ','magenta')+'Writing spectrum parameters to PANDAS DataFrame of sample parameters to be considered later...'

    
    if (id_num, mask) not in seen_idmask and ((stack_meth != 'weighted-average' and (id_num, mask) != ('370', 'a1689_z1_1')) or stack_meth == 'weighted-average'):
        sample_params.iloc[i] = pd.Series([id_num, mask, filt, min(rest_waves), max(rest_waves), z, z_err, eline_flux, eline_lum, eline_lum_error], index=sample_params.columns)
        seen_idmask = seen_idmask + [(id_num,mask)]
        
    elif (id_num, mask) in seen_idmask or (stack_meth != 'weighted-average' and (id_num, mask) == ('370', 'a1689_z1_1')):
        sample_params.iloc[i] = pd.Series([id_num, mask, filt, min(rest_waves), max(rest_waves), np.nan, np.nan, np.nan, np.nan, np.nan], index=sample_params.columns)

    
    print colored('-> ','magenta')+'Plotting original spectrum shifted to rest-frame, rest-frame luminosity spectrum, and rest-frame normalized luminosity spectrum (if made)...'

    fig = plt.figure(figsize=(7,9))

    if norm_eline == 'no-norm':
        nrows = 2
    else:
        nrows = 3

    ax1 = fig.add_subplot(nrows, 1, 1)
    ax1.step(rest_waves, fluxes, where='mid', color='black', linewidth=0.5, label='Spectrum - Fluxes')
    ax1.step(rest_waves, flux_errs, where='mid', color='red', linewidth=0.5, alpha=0.5, label='Error Spectrum')
    ax1.minorticks_on()
    ax1.tick_params(axis='both',which='both',left=True,right=True,bottom=True,top=True,labelbottom=False)
    ax1.set_ylim([-0.5e-17, 1.1*inner_perc_max(fluxes, percentage=85.)])
    ax1.legend(loc='best',frameon=True,fancybox=True,framealpha=0.8,edgecolor='black',fontsize='x-small')
    ax1.set_ylabel(r'$F_\lambda$ ($erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$)')
    ax1.set_title(mask+'.'+filt+'.'+id_num)
             
    ax2 = fig.add_subplot(nrows, 1, 2, sharex = ax1)
    ax2.step(rest_waves, luminosities, where='mid', color='black', linewidth=0.5, label='Spectrum - Luminosities')
    ax2.step(rest_waves, lum_errs, where='mid', color='red', linewidth=0.5, alpha=0.5, label='Error Spectrum')
    ax2.minorticks_on()
    ax2.tick_params(axis='both',which='both',left=True,right=True,bottom=True,top=True,labelbottom=False)
    ax2.set_ylim([-0.2e42, 1.1*inner_perc_max(luminosities, percentage=85.)])
    ax2.legend(loc='best',frameon=True,fancybox=True,framealpha=0.8,edgecolor='black',fontsize='x-small')
    ax2.set_ylabel(r'$L_\lambda$ ($erg\ s^{-1}\ \AA^{-1}$)')

    if nrows == 3:
    
        ax3 = fig.add_subplot(nrows, 1, 3, sharex = ax1)
        ax3.step(rest_waves, lum_norm, where='mid', color='black', linewidth=0.5, label='Spectrum - Normalized Lum.')
        ax3.step(rest_waves, lum_norm_errs, where='mid', color='red', linewidth=0.5, alpha=0.5, label='Error Spectrum')
        ax3.minorticks_on()
        ax3.tick_params(axis='both',which='both',left=True,right=True,bottom=True,top=True)
        ax3.set_ylim([-0.07, 1.1*inner_perc_max(lum_norm, percentage=85.)])
        ax3.legend(loc='best',frameon=True,fancybox=True,framealpha=0.8,edgecolor='black',fontsize='x-small')
        ax3.set_xlabel(r'Rest-Frame Wavelength ($\AA$)')
        ax3.set_ylabel(r'Normalized $L_\lambda$ ($\AA^{-1}$)')

    plt.tight_layout()
    pp.savefig()
    plt.close(fig)
    
    print colored('-> ','magenta')+'Saving spectrum to a file to be accessed later...'


    if norm_eline == 'no-norm':
        spectral_table = np.array([obs_waves, rest_waves, fluxes, flux_errs, luminosities, lum_errs]).T

        fname_out = mask+'.'+filt+'.'+id_num+'.rest-frame.lum.no-norm.not-resampled.txt'
        format_   = ['%10.5f','%10.5f','%6.5e','%6.5e','%6.5e','%6.5e']
        file_cols = 'Obs. Wave. (A) | Rest Wave. (A) | Flux (erg/s/cm2/A) | Flux Error | Luminosity (erg/s/A) | Luminosity Error'

    else:
        spectral_table = np.array([obs_waves, rest_waves, fluxes, flux_errs, luminosities, lum_errs, lum_norm, lum_norm_errs]).T
               
        fname_out = mask+'.'+filt+'.'+id_num+'.rest-frame.lum.norm-lum.not-resampled.txt'
        format_   = ['%10.5f','%10.5f','%6.5e','%6.5e','%6.5e','%6.5e','%10.5f','%10.5f']
        file_cols = 'Obs. Wave. (A) | Rest Wave. (A) | Flux (erg/s/cm2/A) | Flux Error | Luminosity (erg/s/A) | Luminosity Error | Normalized Lum. (A^-1) | Normalized Lum. Error'

    
    np.savetxt(cwd + '/intermed_spectrum_tables_' + norm_eline + '_' + stack_meth + '/' + fname_out,
               spectral_table, fmt=format_, delimiter='\t', newline='\n', comments='#', \
               header=fname_out+'\n'+'Normalized by emission line: '+norm_eline+'\n'+file_cols+'\n'
              )
    
    print
    print colored(fname_out,'green')+' written!'
    print
    print
    print
    
pp.close()

print colored(pp_name,'green')+' written!'
print
print
print



print colored('Sample parameters of interest:', 'magenta', attrs=['bold'])
print sample_params
print
print
    
sample_params_z11   = sample_params[sample_params['Mask'] == 'a1689_z1_1']
sample_params_other = sample_params[sample_params['Mask'] != 'a1689_z1_1']

yjband_stack_min, yjband_stack_max = band_stack_wave_range( np.append(sample_params_z11[sample_params_z11['Filter'] == 'Y']['Min_Wave'], \
                                                                      sample_params_other[sample_params_other['Filter'] == 'J']['Min_Wave']), \
                                                            np.append(sample_params_z11[sample_params_z11['Filter'] == 'Y']['Max_Wave'], \
                                                                      sample_params_other[sample_params_other['Filter'] == 'J']['Max_Wave']) \
                                                          )

jhband_stack_min, jhband_stack_max = band_stack_wave_range( np.append(sample_params_z11[sample_params_z11['Filter'] == 'J']['Min_Wave'], \
                                                                      sample_params_other[sample_params_other['Filter'] == 'H']['Min_Wave']), \
                                                            np.append(sample_params_z11[sample_params_z11['Filter'] == 'J']['Max_Wave'], \
                                                                      sample_params_other[sample_params_other['Filter'] == 'H']['Max_Wave']) \
                                                          )
###################
hkband_stack_min, hkband_stack_max = band_stack_wave_range( np.append(sample_params_z11[sample_params_z11['Filter'] == 'H']['Min_Wave'], \
                                                                      sample_params_other[sample_params_other['Filter'] == 'K']['Min_Wave']), \
                                                            np.append(sample_params_z11[sample_params_z11['Filter'] == 'H']['Max_Wave'], \
                                                                      sample_params_other[sample_params_other['Filter'] == 'K']['Max_Wave']) \
                                                          )
##################


print 'The stack of '+colored('Y','magenta')+' and '+colored('J','magenta')+'-band spectra will cover the wavelength range (A): '+colored(str(yjband_stack_min)+' - '+str(yjband_stack_max),'green')
print 'The stack of '+colored('J','magenta')+' and '+colored('H','magenta')+'-band spectra will cover the wavelength range (A): '+colored(str(jhband_stack_min)+' - '+str(jhband_stack_max),'green')
print 'The stack of '+colored('H','magenta')+' and '+colored('K','magenta')+'-band spectra will cover the wavelength range (A): '+colored(str(hkband_stack_min)+' - '+str(hkband_stack_max),'green') #################
print

if stack_meth == 'average':
    sample_z = sample_params['Redshift'].mean()
    sample_eline_lum = sample_params[line_of_interest+'_Lum'].mean()
    
elif stack_meth == 'median':
    sample_z = sample_params['Redshift'].median()
    sample_eline_lum = sample_params[line_of_interest+'_Lum'].median()

elif stack_meth == 'weighted-average':
    sample_params['Redshift_Weights'] = sample_params['Redshift_Error'].apply(lambda x: 1./(x**2))
    sample_params[line_of_interest+'_Lum_Weights'] = sample_params[line_of_interest+'_Lum_Err'].apply(lambda x: 1./(x**2))

    sample_z = (sample_params['Redshift'] * sample_params['Redshift_Weights']).sum() / sample_params['Redshift_Weights'].sum()
    sample_z_err = np.sqrt(np.divide(1., sample_params['Redshift_Weights'].sum()))
    sample_eline_lum = (sample_params[line_of_interest+'_Lum'] * sample_params[line_of_interest+'_Lum_Weights']).sum() / sample_params[line_of_interest+'_Lum_Weights'].sum()
    sample_eline_lum_err = np.sqrt(np.divide(1., sample_params[line_of_interest+'_Lum_Weights'].sum()))
    
if norm_eline == 'no-norm':
    tname_out = 'sample_parameters_' + norm_eline + '_' + stack_meth + '_' + line_of_interest + '_' + mult_imgs + '.txt'
else:
    tname_out = 'sample_parameters_' + norm_eline + '_' + stack_meth + '.txt'

sample_params.to_csv(cwd + '/intermed_spectrum_tables_'+norm_eline+'_'+stack_meth+'/'+tname_out, \
                     sep='\t', header=True, index=True, index_label='#', line_terminator='\n', na_rep = np.nan \
                    )

print tname_out+' written.'
print

    
yj_disp = round(1.3028 / (sample_z + 1.), 4)  ## Filter dispersions from https://www2.keck.hawaii.edu/inst/mosfire/grating.html
jh_disp = round(1.6269 / (sample_z + 1.), 4)
hk_disp = round(2.1691 / (sample_z + 1.), 4)

stack_min_arr  = np.array([yjband_stack_min, jhband_stack_min, hkband_stack_min])   ##Add hkband_stack_min back in
stack_max_arr  = np.array([yjband_stack_max, jhband_stack_max, hkband_stack_max])   ##Add hkband_stack_max back in
dispersion_arr = np.array([yj_disp, jh_disp, hk_disp])                              ##Add hk_disp back in

resamp_params  = np.array([stack_min_arr, stack_max_arr, dispersion_arr]).T

np.savetxt(cwd + '/intermed_spectrum_tables_' + norm_eline + '_' + stack_meth + '/' + 'resampled_wavelength_parameters.txt', \
           resamp_params, fmt=['%6.5f', '%6.5f', '%2.5f'], delimiter='\t', newline='\n', comments='#', header='resampled_wavelength_parameters.txt'+'\n'+stack_meth+' redshift: '+str(sample_z)+ \
           '\n'+'In YJ, JH, HK descending order'+'\n'+'Min Wavelength (A) | Max Wavelength (A) | Dispersion (A/pix)'+'\n'
          )

print 'The '+colored(stack_meth,'green')+' redshift of the sample is: '+colored(sample_z,'green')
print
print 'From the '+colored('J','magenta')+'-band observed-frame dispersion of '+colored('1.3028','magenta')+' A/pixel that will be used for the '+colored('Y/J','magenta')+' stack, ',
print 'the rest-frame dispersion will be: '+colored(yj_disp,'green')

print 'From the '+colored('H','magenta')+'-band observed-frame dispersion of '+colored('1.6269','magenta')+' A/pixel that will be used for the '+colored('J/H','magenta')+' stack, ',
print 'the rest-frame dispersion will be: '+colored(jh_disp,'green')

print 'From the '+colored('K','magenta')+'-band observed-frame dispersion of '+colored('2.1691','magenta')+' A/pixel that will be used for the '+colored('H/K','magenta')+' stack, ',
print 'the rest-frame dispersion will be: '+colored(hk_disp,'green')
print
print colored('resampled_wavelength_parameters.txt','green')+' - which includes the '+colored('new minimum','magenta')+' and '+colored('new maximum wavelengths','magenta'),
print ' and the '+colored('rest-frame dispersions','magenta')+' - written'
print
print


resampled_spectra = OrderedDict.fromkeys(['YJ', 'JH', 'HK'])  ## Add 'HK' key back in

for key in resampled_spectra.keys():
    resampled_spectra[key] = OrderedDict.fromkeys(['New_Wavelengths', 'New_Luminosities','New_Lum_Errors'])

resampled_spectra['YJ']['New_Wavelengths'] = np.arange(yjband_stack_min, yjband_stack_max + yj_disp, yj_disp)
resampled_spectra['JH']['New_Wavelengths'] = np.arange(jhband_stack_min, jhband_stack_max + jh_disp, jh_disp)
resampled_spectra['HK']['New_Wavelengths'] = np.arange(hkband_stack_min, hkband_stack_max + hk_disp, hk_disp)  ###############################

for key in resampled_spectra.keys():
    resampled_spectra[key]['New_Luminosities'] = np.zeros((len(sample_params)/3, len(resampled_spectra[key]['New_Wavelengths'])))  ## Change 2 -> 3
    resampled_spectra[key]['New_Lum_Errors']   = np.zeros((len(sample_params)/3, len(resampled_spectra[key]['New_Wavelengths'])))  ## Change 2 -> 3
    

path_for_resampling  = cwd + '/intermed_spectrum_tables_' + norm_eline + '_' + stack_meth + '/'

if norm_eline == 'no-norm':
    files_for_resampling = sorted([x for x in os.listdir(path_for_resampling) if any(id_num in x for id_num in list(set(stacking_sample['id']))) and 'not-resampled.txt' in x])
    pp1_name = 'resampled_spectra_'+mult_imgs+'.pdf'
else:
    files_for_resampling = sorted([x for x in os.listdir(path_for_resampling) if 'not-resampled.txt' in x])
    pp1_name = 'resampled_normalized_spectra.pdf'


pp1 = PdfPages(cwd + '/intermed_spectrum_plots_'+norm_eline+'_'+stack_meth+'/' + pp1_name)


yj_idx, jh_idx, hk_idx = 0, 0, 0

for fname in files_for_resampling:

    fig, ax = plt.subplots()

    print colored('--> ','cyan',attrs=['bold'])+'Considering '+colored(fname,'white')+' for resampling:'
    print

    mask   = fname[:fname.index('.')]
    filt   = fname[len(mask)+1]
    id_num = fname[len(mask)+3: fname.index('.rest')]

    print 'Mask: '+colored(mask,'green')
    print 'Filter: '+colored(filt,'green')
    print 'ID: '+colored(id_num,'green')
    print

    if norm_eline == 'no-norm':
        columns = (1,4,5)

    else:
        columns = (1,6,7)
        
    rest_waves, lums_for_resamp, lum_errs_for_resamp = np.loadtxt(path_for_resampling + fname, comments='#', usecols=columns, dtype='float', unpack=True)

    
    if (mask == 'a1689_z1_1' and filt == 'Y') or (mask != 'a1689_z1_1' and filt == 'J'):

        resampled = sf.resample_spectra(resampled_spectra['YJ']['New_Wavelengths'], rest_waves, lums_for_resamp, lum_errors=lum_errs_for_resamp, fill=0., verbose=True)
        resampled_spectra['YJ']['New_Luminosities'][yj_idx] = resampled[:,1]
        resampled_spectra['YJ']['New_Lum_Errors'][yj_idx]   = resampled[:,2]

        
        ax = plot_resampled_spectra(ax, resampled_spectra['YJ']['New_Wavelengths'], resampled_spectra['YJ']['New_Luminosities'][yj_idx], resampled_spectra['YJ']['New_Lum_Errors'][yj_idx], \
                                    eline_rwave, color=['xkcd:sea blue','red','black'], linestyle=['-','-','--'], linewidth=[0.7,0.7,0.5], alpha=[1.,0.5,0.6], \
                                    label=['Dispersion = '+'%.6s' % str(yj_disp)+' A/pix', 'Error_Spectrum']
                                   )

        yj_idx += 1

    elif (mask == 'a1689_z1_1' and filt == 'J') or (mask != 'a1689_z1_1' and filt == 'H'):

        resampled = sf.resample_spectra(resampled_spectra['JH']['New_Wavelengths'], rest_waves, lums_for_resamp, lum_errors=lum_errs_for_resamp, fill=0., verbose=True)
        resampled_spectra['JH']['New_Luminosities'][jh_idx] = resampled[:,1]
        resampled_spectra['JH']['New_Lum_Errors'][jh_idx]   = resampled[:,2]

        
        ax = plot_resampled_spectra(ax, resampled_spectra['JH']['New_Wavelengths'], resampled_spectra['JH']['New_Luminosities'][jh_idx], resampled_spectra['JH']['New_Lum_Errors'][jh_idx], \
                                    eline_rwave, color=['xkcd:sea blue','red','black'], linestyle=['-','-','--'], linewidth=[0.7,0.7,0.5], alpha=[1.,0.5,0.6], \
                                    label=['Dispersion = '+'%.6s' % str(jh_disp)+' A/pix', 'Error_Spectrum']
                                   )
        
        jh_idx += 1

    elif (mask == 'a1689_z1_1' and filt == 'H') or (mask != 'a1689_z1_1' and filt == 'K'):

        resampled = sf.resample_spectra(resampled_spectra['HK']['New_Wavelengths'], rest_waves, lums_for_resamp, lum_errors=lum_errs_for_resamp, fill=0., verbose=True)
        resampled_spectra['HK']['New_Luminosities'][hk_idx] = resampled[:,1]
        resampled_spectra['HK']['New_Lum_Errors'][hk_idx]   = resampled[:,2]

        
        ax = plot_resampled_spectra(ax, resampled_spectra['HK']['New_Wavelengths'], resampled_spectra['HK']['New_Luminosities'][hk_idx], resampled_spectra['HK']['New_Lum_Errors'][hk_idx], \
                                    eline_rwave, color=['xkcd:sea blue','red','black'], linestyle=['-','-','--'], linewidth=[0.7,0.7,0.5], alpha=[1.,0.5,0.6], \
                                    label=['Dispersion = '+'%.6s' % str(hk_disp)+' A/pix', 'Error_Spectrum']
                                   )

        hk_idx += 1

        
    ax.minorticks_on()
    ax.tick_params(axis='both',which='both',left=True,right=True,bottom=True,top=True)
    ax.set_xlabel(r'Rest-Frame Wavelength ($\AA$)')
    if norm_eline == 'no-norm':
        ax.set_ylabel(r'$L_\lambda$ ($erg\ s^{-1}\ \AA^{-1}$)')
    else:
        ax.set_ylabel(r'Normalized $L_\lambda$ ($\AA^{-1}$)')
    ax.legend(loc='best',frameon=True,fancybox=True,framealpha=0.8,edgecolor='black',fontsize='x-small')
    ax.set_title(mask+'.'+filt+'.'+id_num+' Resampled')
    
    plt.tight_layout()
    pp1.savefig()
    plt.close(fig)
    
    fname_out = mask+'.'+filt+'.'+id_num+'.resampled_noDC.txt'

    np.savetxt(path_for_resampling + 'resampled_spectra/' + fname_out, resampled, fmt=['%10.5f','%10.5f','%10.5f'], delimiter='\t', newline='\n', comments='#', \
               header=fname_out+'\n'+'Rest Wavelengths (A) | Resampled Luminosities (A^-1) | Resampled Lum. Errors'+ '\n'
              )

    print
    print colored(fname_out,'green')+' written!'
    print
    print
    print
    
    
pp1.close()

print colored(pp1_name,'green')+' written!'
print
print
print


for bands in resampled_spectra.keys():

    print colored('--> ','cyan',attrs=['bold'])+'Stacking the spectra and finalizing the stacks...'
    print

    if norm_eline == 'no-norm':
        fname_out = 'stacked_spectrum_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_noDC_'+mult_imgs+'.txt'
    else:
        fname_out = 'stacked_spectrum_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_noDC.txt'
        
    final_wavelengths  = resampled_spectra[bands]['New_Wavelengths']

    if stack_meth == 'weighted-average':

        stacked_luminosities, stacked_lum_errs = sf.combine_spectra(resampled_spectra[bands]['New_Luminosities'], stack_meth, resampled_lum_errs=resampled_spectra[bands]['New_Lum_Errors'], axis=0)

        if norm_eline == 'no-norm':
            final_luminosities = stacked_luminosities
            final_lum_errors   = stacked_lum_errs
        else:
            final_luminosities, final_lum_errors = sf.multiply_stack_by_eline(stacked_luminosities, stack_meth, norm_eline, sample_eline_lum, comp_errs=stacked_lum_errs, eline_lum_error=sample_eline_lum_err)


        stacked_spectrum_vals = np.array([final_wavelengths, final_luminosities, final_lum_errors]).T

        np.savetxt(fname_out, stacked_spectrum_vals, fmt=['%10.5f','%6.5e','%6.5e'], delimiter='\t', newline='\n', comments='#', \
                   header=fname_out+'\n'+'Weighted Redshift: '+str('%.5f' % sample_z)+' +/- '+str('%.5e' % sample_z_err)+'\n'+ \
                   'Weighted Lum: '+str('%.5e' % sample_eline_lum)+' +/- '+str('%.5e' % sample_eline_lum_err)+'\n'+ \
                   'Rest-frame wavelength (A) | Luminosity (erg/s/A) | Luminosity Errors'+'\n'
                  )

    else:

        stacked_luminosities  = sf.combine_spectra(resampled_spectra[bands]['New_Luminosities'], stack_meth, axis=0)
        final_luminosities    = sf.multiply_stack_by_eline(stacked_luminosities, stack_meth, norm_eline, sample_eline_lum)

        stacked_spectrum_vals = np.array([final_wavelengths, final_luminosities]).T

        np.savetxt(fname_out, stacked_spectrum_vals, fmt=['%10.5f','%6.5e'], delimiter='\t', newline='\n', comments='#', \
                   header=fname_out+'\n'+stack_meth+' Redshift: '+str('%.5f' % sample_z)+'\n'+ \
                   stack_meth+' Lum: '+str('%.5e' % sample_eline_lum)+'\n'+ \
                   'Rest-frame wavelength (A) | Luminosity (erg/s/A)'+'\n'
                  )

    print
    print colored(fname_out,'green')+' written!'
    print
    print
    print

    
print colored('stacking_spectra_'+norm_eline+'_'+stack_meth+'.log','green')+' - which logs the terminal output - has been written!'
print
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print
print
