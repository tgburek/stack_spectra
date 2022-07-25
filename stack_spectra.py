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
from IPython import embed

print()

class HelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    pass


parser = ArgumentParser(formatter_class=HelpFormatter, description=(
    
"""STACK INDIVIDUAL ALREADY SLIT-LOSS-CORRECTED SPECTRA.

- Spectra will be shifted to the rest frame.
- Spectra will have their flux densities converted to luminosity densities.
- Spectra MAY be dust-corrected. This option is specified in the call to this script (Option present but not currently usable).
- Spectra will be normalized by either continuum luminosity density or the integrated luminosity of an emission line given in the call to this script (Currently supported: OIII5007 and H-alpha).
- Spectra will be resampled onto a wavelength grid with a dispersion (A/pix) equal to the MOSFIRE dispersion de-redshifted by the {median, average, weighted-average} sample redshift (by filter).
- Spectra will be combined via the method (median, average, weighted-average) given in the call to this script.
- Stacked spectrum will be multiplied by the {median, average, weighted-average} sample continuum luminosity density or integrated line luminosity (depending on normalization choice in call to script).
--- The tabulated stacked spectrum without this final step applied will also be written to a file ending in "final_step-stacked.txt"

FOR MORE INFO ON THE PROCEDURE IN, AND OUTPUT OF, THIS SCRIPT, SEE THE README (NOT YET CREATED)."""  ## Have not made the README yet

))


parser.add_argument('-s', '--SLC_Table', metavar='str', \
                    help='The FITS filename of the slit-loss-correction table for emission lines')

parser.add_argument('-d', '--Dust_Correct', action='store_true', \
                    help='If called, each individual spectrum will be dust-corrected (not currently supported)')

parser.add_argument('-m', '--Mult_Images', action='store_true', \
                    help='If called, multiple images (spectra) of the same galaxy are being stacked')

parser.add_argument('-i', '--Include_Stacks', action='store_true', \
                    help='If called, stacking sample will include previously made stacks from "./mult_img_stacks/"')

parser.add_argument('Flux_Table', \
                    help='The FITS filename of the table containing the values that will be used to normalize the spectra')

parser.add_argument('Norm_Feature', choices=['OIII5007', 'H-alpha', 'Lum_Density'], \
                    help='The feature used to normalize each spectrum\n'
                         "Current options include an emission-line's integrated flux (will be converted in-script to luminosity)\n"
                         "  or a continuum's luminosity density at a given wavelength\n"
                         '("Lum_Density" option not currently available when "Mult_Images" or "Include_Stacks" is called)')

parser.add_argument('Stacking_Method', choices=['median', 'average', 'weighted-average'], \
                    help='The method with which the spectra will be stacked at each wavelength element')

parser.add_argument('Stacking_Sample', \
                    help='The FITS file with the spectroscopic IDs to be stacked')

parser.add_argument('--Path',metavar='file',type=str,default='/fc_1d_spectra/',\
                    help='Path to directory with subdirectories of masks with spectra to stack')


args = parser.parse_args()

slc_cat    = args.SLC_Table
dust_corr  = args.Dust_Correct
mult_imgs  = args.Mult_Images
inc_stacks = args.Include_Stacks
flux_cat   = args.Flux_Table
norm_feat  = args.Norm_Feature
stack_meth = args.Stacking_Method
stack_samp = args.Stacking_Sample
path       = args.Path

emiss_lines_to_norm_by = ['OIII5007', 'H-alpha'] ##All current emission-line options to use for normalization

if norm_feat == 'Lum_Density':
    lum_density_wave = str(input('Enter the wavelength, in Angstroms, at which the luminosity density is being derived: '))
    print()

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

    exc_ind = list(range(int(round(max_idx_of_array * end_exc_percentages)))) + \
              list(range(int(round(max_idx_of_array * (1. - end_exc_percentages))), len(yaxis_vals)))

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

intermed_dir = 'intermed_stacking_output_'+norm_feat+'_'+stack_meth

if os.path.isdir('logfiles') == False:
    os.mkdir(cwd + '/logfiles')
    print( 'Created directory: '+colored(cwd+'/logfiles', 'white'))
    print()
    
if os.path.isdir(intermed_dir) == False:
    os.mkdir(cwd + '/' + intermed_dir)
    os.mkdir(cwd + '/' + intermed_dir + '/plots')
    os.mkdir(cwd + '/' + intermed_dir + '/tables')
    print( 'Created directory: '+colored(cwd+'/'+intermed_dir,'white')+' as well as subdirectories '+colored('/plots', 'white')+' and ',)
    print( colored('/tables', 'white')+' therein')
    print()

    
sys.stdout = Logger(logname=cwd+'/logfiles/stacking_spectra_'+norm_feat+'_'+stack_meth, mode='w')


print( '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print( colored(('This program will stack individual spectra that meet certain criteria\n'
               'and have entries in both the flux catalog and photometric catalog.\n'
               'THIS CODE IS IN DEVELOPMENT.'
              ), 'cyan',attrs=['bold']))
print( '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print()
print()

print( 'Review of options called and arguments given to this script:')
print( '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print()
print( 'Options:')
print( '-> SLC Table: ', colored(slc_cat, 'green'))
print( '-> Dust Correct: ', colored(dust_corr, 'green'))
print( '-> Multiple Images: ', colored(mult_imgs, 'green'))
print( '-> Include Stacks: ', colored(inc_stacks, 'green'))
print( '-> Path to parent directory of spectra to stack: ', colored(path, 'green'))
print()
print( 'Arguments:')
print( '-> Flux Table: ', colored(flux_cat, 'green'))
print( '-> Normalize By: ', colored(norm_feat, 'green'))
if norm_feat == 'Lum_Density':
    print( '---> Luminosity density taken at wavelength '+colored(lum_density_wave, 'green')+' Angstroms')
print( '-> Stacking Method: ', colored(stack_meth, 'green'))
print( '-> Stacking Sample Table: ', colored(stack_samp, 'green'))
print()
print()

print( 'The path and current working directory are: ',colored(cwd,'green'))

mask_path = path + '/'

print( 'The path with the mask sub-directories is: ',colored(mask_path,'green'))
print()

mosfire_masks = sorted([x[len(mask_path):].rstrip('/') for x in glob(mask_path+'*/')])

print( "The following masks will have any appropriate associated spectra stacked:")
print()

for mask in mosfire_masks:
    print( colored(mask,'green'))

print()

mult_img_stack_path = cwd+'/mult_img_stacks/'

print( ('If you are including in this stacking procedure previously made stacks for multiply-imaged galaxies,\n'
       'they should be in, and will be pulled from, the directory: '+colored(mult_img_stack_path, 'green')))
print()

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
    
    slc_path  = mask_path + mask + '/error_spectra_corrected/slit_loss_corrected/'#Grabbing all files in directory
    print(slc_path)
    try: 
        slc_files = sorted([x for x in os.listdir(slc_path) if 'fc.1d.esc.slc.txt' in x])
        if len(slc_files) == 0:
            raise RuntimeError
    except RuntimeError:
        slc_files = sorted([x for x in os.listdir(slc_path) if x.find('.fits') != -1])


    for fname in slc_files:
        print( 'Checking to see if '+colored(fname,'white')+' corresponds to a galaxy to be stacked: ',)

        filt   = fname[len(mask)+1]
        id_num = fname[len(mask)+3:-18]
        if path.find('LRIS') != -1:
            filt = 'rest_UV'
            id_num = int(fname.split('_')[0])
            print(type(id_num))
            print(id_num)
            print(type(samp_table['ID'][0]))
        if id_num in samp_table['ID']:
        #if id_num in samp_table['ID'] and (id_num == '370' or (id_num == '1197' and filt != 'H')):
            stacking_sample['fpath'] = np.append(stacking_sample['fpath'], slc_path + fname)
            stacking_sample['mask']  = np.append(stacking_sample['mask'], mask)
            stacking_sample['id']    = np.append(stacking_sample['id'], id_num)
            stacking_sample['filt']  = np.append(stacking_sample['filt'], filt)
            print( colored('True', 'green', attrs=['bold']))
            print()

        else:
            print( colored('False', 'red', attrs=['bold']))
            print()
            

if inc_stacks == True:
    
    mult_image_stacks = sorted([x for x in glob(mult_img_stack_path + 'stacked_spectrum_*.txt') if 'final_step-stacked' not in x])

    for file_path in mult_image_stacks:
        fname = file_path[len(mult_img_stack_path):]

        #id_start = re.search(r'\d', fname).start()
        id_start = fname.find('_noDC_') + len('_noDC_')
        id_num   = fname[id_start : -4]

        if id_num == '1197_370':
            filt = fname[17]
        else:
            filt = fname[18]

        mask = samp_table['Mask'][samp_table['ID'] == id_num].rstrip()

        print( 'Adding the multiple-image stack '+colored(fname,'white')+' to the stacking sample')
        print()

        stacking_sample['fpath'] = np.append(stacking_sample['fpath'], file_path)
        stacking_sample['mask']  = np.append(stacking_sample['mask'], mask)
        stacking_sample['id']    = np.append(stacking_sample['id'], id_num)
        stacking_sample['filt']  = np.append(stacking_sample['filt'], filt)
        
stacking_sample_DF = pd.DataFrame.from_dict(stacking_sample, orient='columns')

print( stacking_sample_DF)
    

exp_stack_sample_size = len(samp_table)                     ###########################
gals_with_data_found  = len(stacking_sample['fpath']) / 3
if path.find('LRIS') != -1:
    gals_with_data_found = len(stacking_sample['fpath'])


print()
print()
print()
print( 'Number of galaxies that should be stacked: ', exp_stack_sample_size)
print( 'Number of galaxies with spectral data found: ', gals_with_data_found)
print()
print( '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print()

if exp_stack_sample_size != gals_with_data_found:
    raise ValueError(('The number of galaxies that should have their spectra stacked does not match the number of spectra found\n'
                      '(number of spectra found divided by 3 to account for the three filters considered)'))
#sys.exit()
##############################################################


if mult_imgs == True:
    mult_img_ids = '_'.join(sorted(list(set(stacking_sample['id']))))
    
    intermed_plot_dir  = cwd + '/' + intermed_dir + '/plots/' + mult_img_ids
    intermed_table_dir = cwd + '/' + intermed_dir + '/tables/' + mult_img_ids

    if os.path.isdir('mult_img_stacks') == False:
        os.mkdir(cwd + '/mult_img_stacks')
        print( 'Created directory: '+colored(cwd+'/mult_img_stacks', 'white'))
        print()

    if os.path.isdir(intermed_plot_dir) == False:
        os.mkdir(intermed_plot_dir)
        print( 'Created directory: '+colored(intermed_plot_dir, 'white'))
        print()

    if os.path.isdir(intermed_table_dir) == False:
        os.mkdir(intermed_table_dir)
        os.mkdir(intermed_table_dir + '/resampled_spectra')
        print( 'Created directory: '+colored(intermed_table_dir, 'white')+' and subdirectory '+colored('/resampled_spectra', 'white')+' therein')
        print()


else:
    intermed_plot_dir  = cwd + '/' + intermed_dir + '/plots'
    intermed_table_dir = cwd + '/' + intermed_dir + '/tables'

    if os.path.isdir(intermed_table_dir + '/resampled_spectra') == False:
        os.mkdir(intermed_table_dir + '/resampled_spectra')
        print( 'Created directory: '+colored(intermed_table_dir, 'white')+' and subdirectory '+colored('/resampled_spectra', 'white')+' therein')
        print()
        


pp_name = 'restframe_lum_normlum_spectra_'+norm_feat+'.pdf'
pp      = PdfPages(intermed_plot_dir + '/' + pp_name)

sample_params_base_dfcols = ['ID', 'Mask', 'Filter', 'Min_Wave', 'Max_Wave', 'Redshift', 'Redshift_Error']

if norm_feat in emiss_lines_to_norm_by:
    norm_specific_colnames = [norm_feat+'_Flux', norm_feat+'_Lum', norm_feat+'_Lum_Err']

elif norm_feat == 'Lum_Density':
    norm_specific_colnames = ['Lum_Density_'+lum_density_wave, 'Lum_Density_'+lum_density_wave+'_Err']

sample_params = pd.DataFrame(index=range(len(stacking_sample['fpath'])), columns=sample_params_base_dfcols + norm_specific_colnames)

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
        fname = file_path[len(mult_img_stack_path):]


    print( colored('--> ','cyan',attrs=['bold'])+'Preparing spectrum in file '+colored(fname,'white')+' for resampling...')
    print()
    print( 'Mask: ', colored(mask,'green'))
    print( 'ID: ', colored(id_num,'green'))
    print( 'Filter: ', colored(filt,'green'))
    print()
    print( 'Spectral feature by which the spectrum will be normalized: ', colored(norm_feat,'green'))
    
    print(type(samp_table['Multiple_Images'][idx_in_samp_table]))
    if samp_table['Multiple_Images'][idx_in_samp_table] == False:
        mask = np.char.replace(mask,'a','A')
        id_num = int(id_num)
        print(id_num)
        print(flux_table['ID'])
        print(flux_table['Mask'])
        try: idx_in_FT = int(np.where((flux_table['Mask'] == mask) & (np.array(flux_table['ID'],dtype=str) == id_num))[0])
        except: 
            print('happened')
            for e,I in enumerate(flux_table['ID']):
                m = flux_table['Mask'][e]
                if I == id_num:
                    print(e)
                    print(m,I)
                    idx_in_FT = e
                    break
        z = flux_table['Weighted_z'][idx_in_FT]
        z_err = flux_table['Weighted_z_Sig'][idx_in_FT]

        if norm_feat in emiss_lines_to_norm_by:
            eline_flux = flux_table[norm_feat+'_Flux'][idx_in_FT]
            eline_sig  = flux_table[norm_feat+'_Sig'][idx_in_FT]

            print( 'Measured emission-line flux (NOT dust-corrected or de-magnified): ',colored('%.5e' % eline_flux,'green'))

            if slc_cat is not None:
                idx_in_SLC = int(np.where((slc_table['Mask'] == mask) & (slc_table['ID_spec'] == id_num))[0])
                star_corr, obj_corr = slc_table[norm_feat+'_Star_Slit'][idx_in_SLC], slc_table[norm_feat+'_Obj_Slit'][idx_in_SLC]

                print( colored('-> ','magenta')+'Slit-loss-correcting the flux of '+colored(norm_feat,'green'))
                print( 'The star-based slit-loss-correction factor to be undone is: '+colored(star_corr,'green'))
                print( 'The object-based slit-loss-correction factor to be applied is: '+colored('%.5f' % obj_corr,'green'))
                print( 'The total correction factor will be: '+colored('%.5f' % (star_corr/obj_corr),'green'))
                print()

                eline_flux = eline_flux * (star_corr / obj_corr)
                eline_sig  = eline_sig  * (star_corr / obj_corr)
                

            eline_lum, eline_lum_error = sf.Flux_to_Lum(eline_flux, eline_sig, redshift = z, densities=False, verbose=True)

            norm_factor = eline_lum
            norm_factor_err = eline_lum_error
            norm_feature_descr = norm_feat + ' integrated luminosity'

            print( 'Calculated, slit-loss-corrected, emission-line luminosity (NOT dust-corrected or de-magnified):',colored('%.5e' % eline_lum,'green'))

        elif norm_feat == 'Lum_Density':
            lum_density     = flux_table['L_UV'][idx_in_FT]
            lum_density_err = flux_table['L_UV_sig'][idx_in_FT]

            norm_factor = lum_density
            norm_factor_err = lum_density_err
            norm_feature_descr = 'continuum luminosity density at '+lum_density_wave+' Angstroms'

            print ( 'Measured continuum luminosity density at '+colored(lum_density_wave, 'magenta')+' Angstroms: ',colored('%.5e' % lum_density,'green'))
            
        print()

        try: obs_waves, fluxes, flux_errs = np.loadtxt(file_path, comments='#', usecols=(0,1,2), dtype='float', unpack=True)
        except: 
            with fits.open(file_path) as hdul:
                data = hdul[1].data
                fluxes = data['flux']*1e-17
                obs_waves = data['wave']
                try: flux_errs = data['sig']*1e-17
                except: flux_errs = (1/np.sqrt(data['ivar']))*1e-17

        rest_waves             = sf.shift_to_rest_frame(obs_waves, redshift = z)
        luminosities, lum_errs = sf.Flux_to_Lum(fluxes, flux_errs, redshift = z,  densities=True, verbose=True)

    else:
        rest_waves, luminosities, lum_errs = np.loadtxt(file_path, comments='#', usecols=(0,1,2), dtype='float', unpack=True)

        nans = np.where(np.isnan(luminosities) == True)[0]

        rest_waves   = np.delete(rest_waves, nans)
        luminosities = np.delete(luminosities, nans)
        lum_errs     = np.delete(lum_errs, nans)

        z = samp_table['Weighted_z'][idx_in_samp_table]
        z_err = samp_table['Weighted_z_Sig'][idx_in_samp_table]
        eline_lum = samp_table[norm_feat+'_Lum'][idx_in_samp_table]
        eline_lum_error = samp_table[norm_feat+'_Lum_Sig'][idx_in_samp_table]

        norm_factor = eline_lum
        norm_factor_err = eline_lum_error
        norm_feature_descr = norm_feat + ' integrated luminosity'

        print( 'Measured emission-line luminosity from multiple-image stack (NOT dust-corrected or de-magnified): ',colored('%.5e' % eline_lum,'green'))
        print()

        ## To make plotting and file writing less convoluted

        obs_waves = sf.shift_to_obs_frame(rest_waves, redshift=z)
        fluxes, flux_errs = sf.Flux_to_Lum(luminosities, lum_errs, redshift = z, Lum_to_Flux=True, densities=True)
        eline_flux, eline_sig = sf.Flux_to_Lum(eline_lum, eline_lum_error, redshift = z, Lum_to_Flux=True, densities=False)


    lum_norm, lum_norm_errs = sf.normalize_spectra(luminosities, norm_feature_descr, norm_factor, error_spectrum=lum_errs, norm_factor_err=norm_factor_err) 

    
    print( colored('-> ','magenta')+'Writing spectrum parameters to PANDAS DataFrame of sample parameters to be considered later...')

    if (id_num, mask) not in seen_idmask:

        if norm_feat in emiss_lines_to_norm_by:
            norm_specific_cols = [eline_flux, eline_lum, eline_lum_error]
        elif norm_feat == 'Lum_Density':
            norm_specific_cols = [lum_density, lum_density_err]

        sample_params.iloc[i] = pd.Series([id_num, mask, filt, min(rest_waves), max(rest_waves), z, z_err]+norm_specific_cols, index=sample_params.columns)
        
        seen_idmask.append((id_num, mask))
        
    else:
        norm_specific_cols = np.full(len(norm_specific_colnames), np.nan)

        sample_params.iloc[i] = pd.Series([id_num, mask, filt, min(rest_waves), max(rest_waves), np.nan, np.nan]+list(norm_specific_cols), index=sample_params.columns)

    
    print( colored('-> ','magenta')+'Plotting original spectrum shifted to rest-frame, rest-frame luminosity spectrum, and rest-frame normalized luminosity spectrum...')

    fig = plt.figure(figsize=(7,9))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.step(rest_waves, fluxes, where='mid', color='black', linewidth=0.5, label='Spectrum - Fluxes')
    ax1.step(rest_waves, flux_errs, where='mid', color='red', linewidth=0.5, alpha=0.5, label='Error Spectrum')
    ax1.minorticks_on()
    ax1.tick_params(axis='both',which='both',left=True,right=True,bottom=True,top=True,labelbottom=False)
    ax1.set_ylim([-0.5e-17, 1.1*inner_perc_max(fluxes, percentage=85.)])
    ax1.legend(loc='best',frameon=True,fancybox=True,framealpha=0.8,edgecolor='black',fontsize='x-small')
    ax1.set_ylabel(r'$F_\lambda$ ($erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$)')
    ax1.set_title(str(mask)+'.'+str(filt)+'.'+str(id_num))
             
    ax2 = fig.add_subplot(3, 1, 2, sharex = ax1)
    ax2.step(rest_waves, luminosities, where='mid', color='black', linewidth=0.5, label='Spectrum - Luminosities')
    ax2.step(rest_waves, lum_errs, where='mid', color='red', linewidth=0.5, alpha=0.5, label='Error Spectrum')
    ax2.minorticks_on()
    ax2.tick_params(axis='both',which='both',left=True,right=True,bottom=True,top=True,labelbottom=False)
    ax2.set_ylim([-0.2e42, 1.1*inner_perc_max(luminosities, percentage=85.)])
    ax2.legend(loc='best',frameon=True,fancybox=True,framealpha=0.8,edgecolor='black',fontsize='x-small')
    ax2.set_ylabel(r'$L_\lambda$ ($erg\ s^{-1}\ \AA^{-1}$)')
    
    ax3 = fig.add_subplot(3, 1, 3, sharex = ax1)
    ax3.step(rest_waves, lum_norm, where='mid', color='black', linewidth=0.5, label='Spectrum - Normalized Lum.')
    ax3.step(rest_waves, lum_norm_errs, where='mid', color='red', linewidth=0.5, alpha=0.5, label='Error Spectrum')
    ax3.minorticks_on()
    ax3.tick_params(axis='both',which='both',left=True,right=True,bottom=True,top=True)
    ax3.set_ylim([-0.07, 1.1*inner_perc_max(lum_norm, percentage=85.)])
    ax3.legend(loc='best',frameon=True,fancybox=True,framealpha=0.8,edgecolor='black',fontsize='x-small')
    ax3.set_xlabel(r'Rest-Frame Wavelength ($\AA$)')
    if norm_feat in emiss_lines_to_norm_by:
        ax3.set_ylabel(r'Normalized $L_\lambda$ ($\AA^{-1}$)')
        norm_lum_colname = 'Normalized Lum. (A^-1)'
    elif norm_feat == 'Lum_Density':
        ax3.set_ylabel(r'Normalized $L_\lambda$ (Unitless)')
        norm_lum_colname = 'Normalized Lum. (Unitless)'

    plt.tight_layout()
    pp.savefig()
    plt.close(fig)
    
    print( colored('-> ','magenta')+'Saving spectrum to a file to be accessed later...')


    spectral_table = np.array([obs_waves, rest_waves, fluxes, flux_errs, luminosities, lum_errs, lum_norm, lum_norm_errs]).T
               
    fname_out = str(mask)+'.'+str(filt)+'.'+str(id_num)+'.rest-frame.lum.norm-lum.not-resampled.txt'
    format_   = ['%10.5f','%10.5f','%6.5e','%6.5e','%6.5e','%6.5e','%10.5f','%10.5f']
    file_cols = 'Obs. Wave. (A) | Rest Wave. (A) | Flux (erg/s/cm2/A) | Flux Error | Luminosity (erg/s/A) | Luminosity Error | ' + norm_lum_colname + ' | Normalized Lum. Error'

    
    np.savetxt(intermed_table_dir + '/' + fname_out, spectral_table, fmt=format_, delimiter='\t', newline='\n', comments='#', \
               header=fname_out+'\n'+'Normalized by: '+norm_feature_descr+'\n'+file_cols+'\n')
    
    print()
    print( colored(fname_out,'green')+' written!')
    print()
    print()
    print()
    
pp.close()

print( colored(pp_name,'green')+' written!')
print()
print()
print()



print( colored('Sample parameters of interest:', 'magenta', attrs=['bold']))
print( sample_params)
print()
print()
    
sample_params_z11   = sample_params[sample_params['Mask'] == 'a1689_z1_1']
sample_params_other = sample_params[sample_params['Mask'] != 'a1689_z1_1']
if path.find('LRIS') != -1:
    print(filt)
    LRISband_stack_min, LRISband_stack_max = band_stack_wave_range( np.append(sample_params_z11[sample_params_z11['Filter'] == 'rest_UV']['Min_Wave'], \
                                                                      sample_params_other[sample_params_other['Filter'] == 'rest_UV']['Min_Wave']), \
                                                            np.append(sample_params_z11[sample_params_z11['Filter'] == 'rest_UV']['Max_Wave'], \
                                                                      sample_params_other[sample_params_other['Filter'] == 'rest_UV']['Max_Wave']) \
                                                          )
else:                                                          
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


    print( 'The stack of '+colored('Y','magenta')+' and '+colored('J','magenta')+'-band spectra will cover the wavelength range (A): '+colored(str(yjband_stack_min)+' - '+str(yjband_stack_max),'green'))
    print( 'The stack of '+colored('J','magenta')+' and '+colored('H','magenta')+'-band spectra will cover the wavelength range (A): '+colored(str(jhband_stack_min)+' - '+str(jhband_stack_max),'green'))
    print( 'The stack of '+colored('H','magenta')+' and '+colored('K','magenta')+'-band spectra will cover the wavelength range (A): '+colored(str(hkband_stack_min)+' - '+str(hkband_stack_max),'green')) #################
    print()

if stack_meth == 'average':
    sample_z = sample_params['Redshift'].mean()

elif stack_meth == 'median':
    sample_z = sample_params['Redshift'].median()

elif stack_meth == 'weighted-average':
    sample_params['Redshift_Weights'] = sample_params['Redshift_Error'].apply(lambda x: 1./(x**2))
    sample_z = (sample_params['Redshift'] * sample_params['Redshift_Weights']).sum() / sample_params['Redshift_Weights'].sum()
    sample_z_err = np.sqrt(np.divide(1., sample_params['Redshift_Weights'].sum()))


if mult_imgs == False:

    print( 'At the end of the stacking process, the sample '+colored(stack_meth, 'magenta')+' of the normalization feature ('+colored(norm_feat, 'magenta')+') will be multiplied into the stack')
    print( 'This value will be written to the header of the tabulated stacked spectrum multiplied by this value')
    print( '--- A version of the stack without this final multiplication will also be created ---')
    print()

    if norm_feat in emiss_lines_to_norm_by:
        norm_feat_colname = norm_feat+'_Lum'
    elif norm_feat == 'Lum_Density':
        norm_feat_colname = 'Lum_Density_'+lum_density_wave
    
    if stack_meth == 'average':
        sample_norm_fact = sample_params[norm_feat_colname].mean()

    elif stack_meth == 'median':
        sample_norm_fact = sample_params[norm_feat_colname].median()

    elif stack_meth == 'weighted-average':
        sample_params[norm_feat_colname+'_Weights'] = sample_params[norm_feat_colname+'_Err'].apply(lambda x: 1./(x**2))
        sample_norm_fact = (sample_params[norm_feat_colname] * sample_params[norm_feat_colname+'_Weights']).sum() / sample_params[norm_feat_colname+'_Weights'].sum()
        sample_norm_fact_err = np.sqrt(np.divide(1., sample_params[norm_feat_colname+'_Weights'].sum()))

else:

    print( 'At the end of the stacking process, the lowest luminosity (proxy for least magnified) normalization feature ('+colored(norm_feat, 'magenta')+') will be multiplied into the stack')
    print( 'This value will be written to the header of the tabulated stacked spectrum multiplied by this value')
    print( '--- A version of the stack without this final multiplication will also be created ---')
    print()

    norm_feat_colname = norm_feat+'_Lum'

    min_lum_row = sample_params[sample_params[norm_feat_colname] == sample_params[norm_feat_colname].min()]
    sample_norm_fact = sample_params.loc[min_lum_row.index.values[0], norm_feat_colname]
    sample_norm_fact_err = sample_params.loc[min_lum_row.index.values[0], norm_feat_colname+'_Err']
    

tname_out = 'sample_parameters_' + norm_feat + '_' + stack_meth + '.txt'

sample_params.to_csv(intermed_table_dir + '/' + tname_out, sep='\t', header=True, index=True, index_label='#', line_terminator='\n', na_rep = np.nan)

print( tname_out+' written.')
print()

if path.find('LRIS') != -1:
    LRIS_disp = round(2.18/(sample_z+1.),4)
    stack_min_arr = np.array([LRISband_stack_min])
    stack_max_arr = np.array([LRISband_stack_max])
    dispersion_arr = np.array([LRIS_disp])
else:
    yj_disp = round(1.3028 / (sample_z + 1.), 4)  ## Filter dispersions from https://www2.keck.hawaii.edu/inst/mosfire/grating.html
    jh_disp = round(1.6269 / (sample_z + 1.), 4)
    hk_disp = round(2.1691 / (sample_z + 1.), 4)

    stack_min_arr  = np.array([yjband_stack_min, jhband_stack_min, hkband_stack_min])   ##Add hkband_stack_min back in
    stack_max_arr  = np.array([yjband_stack_max, jhband_stack_max, hkband_stack_max])   ##Add hkband_stack_max back in
    dispersion_arr = np.array([yj_disp, jh_disp, hk_disp])                              ##Add hk_disp back in

resamp_params  = np.array([stack_min_arr, stack_max_arr, dispersion_arr]).T

np.savetxt(intermed_table_dir + '/' + 'resampled_wavelength_parameters.txt', resamp_params, fmt=['%6.5f', '%6.5f', '%2.5f'], delimiter='\t', newline='\n', comments='#', \
           header='resampled_wavelength_parameters.txt'+'\n'+stack_meth+' redshift: '+str(sample_z)+ \
           '\n'+'In YJ, JH, HK descending order'+'\n'+'Min Wavelength (A) | Max Wavelength (A) | Rest-Frame Dispersion (A/pix)'+'\n')


print( 'The '+colored(stack_meth,'green')+' redshift of the sample is: '+colored(sample_z,'green'))
print()
if path.find('LRIS') != -1:
    print( 'From the '+colored('LRIS','magenta')+'-band observed-frame dispersion of '+colored('2.18','magenta')+' A/pixel that will be used for the '+colored('LRIS','magenta')+' stack, ',)
    print( 'the rest-frame dispersion will be: '+colored(LRIS_disp,'green'))
else:
    print( 'From the '+colored('J','magenta')+'-band observed-frame dispersion of '+colored('1.3028','magenta')+' A/pixel that will be used for the '+colored('Y/J','magenta')+' stack, ',)
    print( 'the rest-frame dispersion will be: '+colored(yj_disp,'green'))

    print( 'From the '+colored('H','magenta')+'-band observed-frame dispersion of '+colored('1.6269','magenta')+' A/pixel that will be used for the '+colored('J/H','magenta')+' stack, ',)
    print( 'the rest-frame dispersion will be: '+colored(jh_disp,'green'))

    print( 'From the '+colored('K','magenta')+'-band observed-frame dispersion of '+colored('2.1691','magenta')+' A/pixel that will be used for the '+colored('H/K','magenta')+' stack, ',)
    print( 'the rest-frame dispersion will be: '+colored(hk_disp,'green'))
print()
print( colored('resampled_wavelength_parameters.txt','green')+' - which includes the '+colored('new minimum','magenta')+' and '+colored('new maximum wavelengths','magenta'),)
print( ' and the '+colored('rest-frame dispersions','magenta')+' - written')
print()
print()

if path.find('LRIS') != -1:
    resampled_spectra = OrderedDict.fromkeys(['LRIS'])
else:
    resampled_spectra = OrderedDict.fromkeys(['YJ', 'JH', 'HK'])  ## Add 'HK' key back in

for key in resampled_spectra.keys():
    resampled_spectra[key] = OrderedDict.fromkeys(['New_Wavelengths', 'New_Luminosities','New_Lum_Errors'])
if path.find('LRIS') != -1:
    resampled_spectra['LRIS']['New_Wavelengths'] = np.arange(LRISband_stack_min,LRISband_stack_max+LRIS_disp,LRIS_disp)
else:
    resampled_spectra['YJ']['New_Wavelengths'] = np.arange(yjband_stack_min, yjband_stack_max + yj_disp, yj_disp)
    resampled_spectra['JH']['New_Wavelengths'] = np.arange(jhband_stack_min, jhband_stack_max + jh_disp, jh_disp)
    resampled_spectra['HK']['New_Wavelengths'] = np.arange(hkband_stack_min, hkband_stack_max + hk_disp, hk_disp)  ###############################

for key in resampled_spectra.keys():
    # if key == 'YJ' or key == 'JH':
    #     resampled_spectra[key]['New_Luminosities'] = np.zeros((2, len(resampled_spectra[key]['New_Wavelengths'])))
    #     resampled_spectra[key]['New_Lum_Errors']   = np.zeros((2, len(resampled_spectra[key]['New_Wavelengths'])))
    # else:
    #     resampled_spectra[key]['New_Luminosities'] = np.zeros((1, len(resampled_spectra[key]['New_Wavelengths'])))
    #     resampled_spectra[key]['New_Lum_Errors']   = np.zeros((1, len(resampled_spectra[key]['New_Wavelengths'])))
    b = int(len(sample_params)/3)
    if path.find('LRIS') != -1: b = int(len(sample_params))
    a = len(resampled_spectra[key]['New_Wavelengths'])
    dim = (b,a)
    resampled_spectra[key]['New_Luminosities'] = np.zeros(dim)  ## Change 2 -> 3
    resampled_spectra[key]['New_Lum_Errors']   = np.zeros(dim) ## Change 2 -> 3
    

path_for_resampling  = intermed_table_dir + '/'

files_for_resampling = sorted([x for x in os.listdir(path_for_resampling) if 'not-resampled.txt' in x])

pp1_name = 'resampled_normalized_spectra.pdf'
pp1 = PdfPages(intermed_plot_dir + '/' + pp1_name)


yj_idx, jh_idx, hk_idx, lris_idx = 0, 0, 0, 0
print(colored('problem','red'))
print(files_for_resampling)
for fname in files_for_resampling:
    print("The length is {}".format(len(files_for_resampling)))

    fig, ax = plt.subplots()

    print( colored('--> ','cyan',attrs=['bold'])+'Considering '+colored(fname,'white')+' for resampling:')
    print()

    mask   = fname[:fname.index('.')]
    filt   = fname[len(mask)+1]
    id_num = fname[len(mask)+3: fname.index('.rest')]
    if path.find('LRIS') != -1:
        print(fname)
        filt = 'rest_UV'
        id_num = fname.split('.')[2]
        print(type(id_num))

    print( 'Mask: '+colored(mask,'green'))
    print( 'Filter: '+colored(filt,'green'))
    print( 'ID: '+colored(id_num,'green'))
    print()
        
    rest_waves, lums_for_resamp, lum_errs_for_resamp = np.loadtxt(path_for_resampling + fname, comments='#', usecols=(1,6,7), dtype='float', unpack=True)

    
    if (mask == 'a1689_z1_1' and filt == 'Y') or (mask != 'a1689_z1_1' and filt == 'J'):

        resampled = sf.resample_spectra(resampled_spectra['YJ']['New_Wavelengths'], rest_waves, lums_for_resamp, error_spectrum=lum_errs_for_resamp, fill=0., verbose=True)
        resampled_spectra['YJ']['New_Luminosities'][yj_idx] = resampled[:,1]
        resampled_spectra['YJ']['New_Lum_Errors'][yj_idx]   = resampled[:,2]

        
        ax = plot_resampled_spectra(ax, resampled_spectra['YJ']['New_Wavelengths'], resampled_spectra['YJ']['New_Luminosities'][yj_idx], resampled_spectra['YJ']['New_Lum_Errors'][yj_idx], \
                                    eline_rwave, color=['xkcd:sea blue','red','black'], linestyle=['-','-','--'], linewidth=[0.7,0.7,0.5], alpha=[1.,0.5,0.6], \
                                    label=['Dispersion = '+'%.6s' % str(yj_disp)+' A/pix', 'Error_Spectrum']
                                   )

        yj_idx += 1

    elif (mask == 'a1689_z1_1' and filt == 'J') or (mask != 'a1689_z1_1' and filt == 'H'):

        resampled = sf.resample_spectra(resampled_spectra['JH']['New_Wavelengths'], rest_waves, lums_for_resamp, error_spectrum=lum_errs_for_resamp, fill=0., verbose=True)
        resampled_spectra['JH']['New_Luminosities'][jh_idx] = resampled[:,1]
        resampled_spectra['JH']['New_Lum_Errors'][jh_idx]   = resampled[:,2]

        
        ax = plot_resampled_spectra(ax, resampled_spectra['JH']['New_Wavelengths'], resampled_spectra['JH']['New_Luminosities'][jh_idx], resampled_spectra['JH']['New_Lum_Errors'][jh_idx], \
                                    eline_rwave, color=['xkcd:sea blue','red','black'], linestyle=['-','-','--'], linewidth=[0.7,0.7,0.5], alpha=[1.,0.5,0.6], \
                                    label=['Dispersion = '+'%.6s' % str(jh_disp)+' A/pix', 'Error_Spectrum']
                                   )
        
        jh_idx += 1

    elif (mask == 'a1689_z1_1' and filt == 'H') or (mask != 'a1689_z1_1' and filt == 'K'):

        resampled = sf.resample_spectra(resampled_spectra['HK']['New_Wavelengths'], rest_waves, lums_for_resamp, error_spectrum=lum_errs_for_resamp, fill=0., verbose=True)
        resampled_spectra['HK']['New_Luminosities'][hk_idx] = resampled[:,1]
        resampled_spectra['HK']['New_Lum_Errors'][hk_idx]   = resampled[:,2]

        
        ax = plot_resampled_spectra(ax, resampled_spectra['HK']['New_Wavelengths'], resampled_spectra['HK']['New_Luminosities'][hk_idx], resampled_spectra['HK']['New_Lum_Errors'][hk_idx], \
                                    eline_rwave, color=['xkcd:sea blue','red','black'], linestyle=['-','-','--'], linewidth=[0.7,0.7,0.5], alpha=[1.,0.5,0.6], \
                                    label=['Dispersion = '+'%.6s' % str(hk_disp)+' A/pix', 'Error_Spectrum']
                                   )

        hk_idx += 1
    elif (mask == 'a1689_z1_1' and filt == 'rest_UV') or (mask != 'a1689_z1_1' and filt == 'rest_UV'):
        resampled = sf.resample_spectra(resampled_spectra['LRIS']['New_Wavelengths'], rest_waves, lums_for_resamp, error_spectrum=lum_errs_for_resamp, fill=0., verbose=True)
        print(resampled)
        resampled_spectra['LRIS']['New_Luminosities'][lris_idx] = resampled[:,1]
        resampled_spectra['LRIS']['New_Lum_Errors'][lris_idx]   = resampled[:,2]


        
        ax = plot_resampled_spectra(ax, resampled_spectra['LRIS']['New_Wavelengths'], resampled_spectra['LRIS']['New_Luminosities'][lris_idx], resampled_spectra['LRIS']['New_Lum_Errors'][lris_idx], \
                                    eline_rwave, color=['xkcd:sea blue','red','black'], linestyle=['-','-','--'], linewidth=[0.7,0.7,0.5], alpha=[1.,0.5,0.6], \
                                    label=['Dispersion = '+'%.6s' % str(LRIS_disp)+' A/pix', 'Error_Spectrum']
                                   )
        lris_idx += 1

        
    ax.minorticks_on()
    ax.tick_params(axis='both',which='both',left=True,right=True,bottom=True,top=True)
    ax.set_xlabel(r'Rest-Frame Wavelength ($\AA$)')
    if norm_feat in emiss_lines_to_norm_by:
        ax.set_ylabel(r'Normalized $L_\lambda$ ($\AA^{-1}$)')
        norm_lum_colname = 'Resampled Luminosities (A^-1)'
    elif norm_feat == 'Lum_Density':
        ax.set_ylabel(r'Normalized $L_\lambda$ (Unitless)')
        norm_lum_colname = 'Resampled Luminosities (Unitless)'
    ax.legend(loc='best',frameon=True,fancybox=True,framealpha=0.8,edgecolor='black',fontsize='x-small')
    ax.set_title(str(mask)+'.'+str(filt)+'.'+str(id_num)+' Resampled')
    
    plt.tight_layout()
    pp1.savefig()
    plt.close(fig)
    
    fname_out = mask+'.'+filt+'.'+id_num+'.resampled_noDC.txt'

    np.savetxt(path_for_resampling + 'resampled_spectra/' + fname_out, resampled, fmt=['%10.5f','%10.5f','%10.5f'], delimiter='\t', newline='\n', comments='#', \
               header=fname_out+'\n'+'Rest Wavelengths (A) | ' + norm_lum_colname + ' | Resampled Lum. Errors'+ '\n')

    print()
    print( colored(fname_out,'green')+' written!')
    print()
    print()
    print()
    
    
pp1.close()

print( colored(pp1_name,'green')+' written!')
print()
print()
print()


for bands in resampled_spectra.keys():

    print( colored('--> ','cyan',attrs=['bold'])+'Stacking the spectra and finalizing the stacks...')
    print()

    if mult_imgs == True:
        file_path = mult_img_stack_path
        fname_out_mult_sfeat = 'stacked_spectrum_'+bands+'-bands_'+stack_meth+'_'+norm_feat+'_noDC_'+mult_img_ids+'.txt'
        fname_out_stacked    = 'stacked_spectrum_'+bands+'-bands_'+stack_meth+'_'+norm_feat+'_noDC_'+mult_img_ids+'_final_step-stacked.txt'
        lum_designation      = 'Minimum '+norm_feature_descr+' of sample: '
        # if bands == 'HK':
        #     header_add = 'THIS IS NOT A STACK OF IDs 1197 AND 370. THIS IS JUST THE H-BAND SPECTRUM OF 370 DUE TO THE ABSENCE OF [NII]6583 IN THE H-BAND 1197 SPECTRUM.'
        # else:
        #     header_add = ''
    else:
        file_path = cwd + '/'
        fname_out_mult_sfeat = 'stacked_spectrum_'+bands+'-bands_'+stack_meth+'_'+norm_feat+'_noDC.txt'
        fname_out_stacked    = 'stacked_spectrum_'+bands+'-bands_'+stack_meth+'_'+norm_feat+'_noDC_final_step-stacked.txt'
        lum_designation      = stack_meth.capitalize()+' '+norm_feature_descr+' of sample: '
        
    final_wavelengths = resampled_spectra[bands]['New_Wavelengths']

    if stack_meth == 'weighted-average':
        stacked_luminosities, stacked_lum_errs = sf.combine_spectra(resampled_spectra[bands]['New_Luminosities'], stack_meth, resampled_error_spectra=resampled_spectra[bands]['New_Lum_Errors'], axis=0)
        final_luminosities, final_lum_errors   = sf.multiply_stack_by_sfeat(stacked_luminosities, stack_meth, norm_feature_descr, sample_norm_fact, comp_error_spect=stacked_lum_errs, sample_val_error=sample_norm_fact_err)

        print(stacked_luminosities)
        stacked_spectrum_vals = np.array([final_wavelengths, final_luminosities, final_lum_errors]).T

        np.savetxt(file_path + fname_out_mult_sfeat, stacked_spectrum_vals, fmt=['%10.5f','%6.5e','%6.5e'], delimiter='\t', newline='\n', comments='#', \
                   header=fname_out_mult_sfeat+'\n'+'Weighted Redshift: '+str('%.5f' % sample_z)+' +/- '+str('%.5e' % sample_z_err)+'\n'+ \
                   lum_designation+str('%.5e' % sample_norm_fact)+' +/- '+str('%.5e' % sample_norm_fact_err)+'\n'+ \
                   'Rest-frame wavelength (A) | Luminosity (erg/s/A) | Luminosity Errors'+'\n') #######################At end of header, remove "+header_add+'\n'"

        stacked_spectrum_vals = np.array([final_wavelengths, stacked_luminosities, stacked_lum_errs]).T

        np.savetxt(file_path + fname_out_stacked, stacked_spectrum_vals, fmt=['%10.5f','%6.5e','%6.5e'], delimiter='\t', newline='\n', comments='#', \
                   header=fname_out_stacked+'\n'+'Weighted Redshift: '+str('%.5f' % sample_z)+' +/- '+str('%.5e' % sample_z_err)+'\n'+ \
                   'Rest-frame wavelength (A) | Luminosity (erg/s/A) | Luminosity Errors'+'\n') ########################At end of header, remove "+header_add+'\n'"

    else:

        stacked_luminosities  = sf.combine_spectra(resampled_spectra[bands]['New_Luminosities'], stack_meth, axis=0)
        final_luminosities    = sf.multiply_stack_by_sfeat(stacked_luminosities, stack_meth, norm_feature_descr, sample_norm_fact)

        stacked_spectrum_vals = np.array([final_wavelengths, final_luminosities]).T

        np.savetxt(file_path + fname_out_mult_sfeat, stacked_spectrum_vals, fmt=['%10.5f','%6.5e'], delimiter='\t', newline='\n', comments='#', \
                   header=fname_out_mult_sfeat+'\n'+stack_meth+' Redshift: '+str('%.5f' % sample_z)+'\n'+ \
                   lum_designation+str('%.5e' % sample_norm_fact)+'\n'+ \
                   'Rest-frame wavelength (A) | Luminosity (erg/s/A)'+'\n')

        stacked_spectrum_vals = np.array([final_wavelengths, stacked_luminosities]).T

        np.savetxt(file_path + fname_out_stacked, stacked_spectrum_vals, fmt=['%10.5f','%6.5e'], delimiter='\t', newline='\n', comments='#', \
                   header=fname_out_stacked+'\n'+stack_meth+' Redshift: '+str('%.5f' % sample_z)+'\n'+ \
                   'Rest-frame wavelength (A) | Luminosity (erg/s/A)'+'\n')

    print()
    print( colored(fname_out_mult_sfeat, 'green')+' written!')
    print( colored(fname_out_stacked, 'green')+' written!')
    print()
    print()
    print()

    
print( colored('stacking_spectra_'+norm_feat+'_'+stack_meth+'_'+time.strftime('%m-%d-%Y')+'.log','green')+' - which logs the terminal output - has been written')
print( 'and stored in: '+cwd+'/logfiles/')
print()
print( '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print()
print()
print()
