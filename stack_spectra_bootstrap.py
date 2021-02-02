#! /usr/bin/env python

import os
import re
import sys
import time
import numpy as np
import pandas as pd
import fits_readin as fr
import stacking_functions as sf
from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
from glob import glob
from collections import OrderedDict
from termcolor import colored


print

class HelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    pass


parser = ArgumentParser(formatter_class=HelpFormatter, description=(
    
"""ESTIMATE UNCERTAINTY SPECTRUM OF STACK.
UNCERTAINTY CAN BE DESIGNATED TO BE STATISTICAL ONLY OR INCLUDE COSMIC VARIANCE THROUGH BOOTSTRAP RESAMPLING.

THIS INVOLVES STACKING INDIVIDUAL, PERTURBED, SLIT-LOSS-CORRECTED SPECTRA.
- Spectra have already been shifted to the rest frame.
- Spectra have had their flux densities converted to luminosity densities.
- Spectra MAY be dust-corrected. This option is specified in the call to this script (Code to dust-correct not currently implemented).
- Spectra MAY be normalized by an emission line given in the call to this script (Currently they will be normalized).
- Spectra will be resampled onto a wavelength grid with spacing equal to that at the {median, average} sample redshift (by filter).
- Spectra will be combined via the method given in the call to this script.
- Spectra MAY be multiplied by the {median, average} line luminosity corresponding to the emission line used for normalization, if done.

FOR MORE INFO ON THE PROCEDURE IN THIS SCRIPT, SEE THE README (NOT YET CREATED)."""  ## Have not made the README yet

))


parser.add_argument('-n', '--N_Comp', metavar='int', type=int, default=500, \
                    help='The number of composite spectra to generate')  ##Typically 500

parser.add_argument('-c', '--Cosmic_Var', action='store_true', \
                    help='If called, cosmic variance will be included through bootstrap resampling\n'
                         'Otherwise, the uncertainty spectrum will be statistical only')  ##If this option is called, this argument will be True.  Otherwise it's False

parser.add_argument('-d', '--Dust_Correct', action='store_true', \
                    help='If called, each individual spectrum will be dust-corrected.')

parser.add_argument('Norm_Eline', choices=['OIII5007', 'H-alpha'], \
                    help='The emission line name of the line used to normalize each spectrum')

parser.add_argument('Stacking_Method', choices=['median', 'average'], \
                    help='The method with which the spectra will be stacked')

parser.add_argument('Stacking_Sample', \
                    help='The FITS file with the spectrum IDs for stacking')



args = parser.parse_args()

ncomp       = args.N_Comp
inc_cos_var = args.Cosmic_Var
dust_corr   = args.Dust_Correct
norm_eline  = args.Norm_Eline
stack_meth  = args.Stacking_Method
stack_samp  = args.Stacking_Sample



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


def write_term_file(output, filename = 'stack_uncertainty_est_'+norm_eline+'_'+stack_meth):
    term_only  = sys.stdout
    sys.stdout = Logger(logname=cwd+'/logfiles/'+filename, mode='a')
    print output
    sys.stdout = term_only

    return


def create_samp_cat(ids, masks, dirpath, return_DF=False):
    samp_dict = OrderedDict.fromkeys(['fpath', 'mask', 'id', 'filt'])

    for key in samp_dict.keys():
        samp_dict[key] = np.array([])

    for id_num, mask in zip(ids, masks):

        file_names = sorted(glob(dirpath + mask + '.?.' + id_num + '.rest-frame.lum.norm-lum.not-resampled.txt'))

        if id_num == '1197_370':
            file_names.append(dirpath + 'a1689_z1_1.H.370.rest-frame.lum.norm-lum.not-resampled.txt')

        for file_ in file_names:

            fname = file_[len(dirpath):]
            filt  = fname[len(mask)+1]
            ID    = fname[len(mask)+3 : -42]
    
            samp_dict['fpath'] = np.append(samp_dict['fpath'], file_)
            samp_dict['mask']  = np.append(samp_dict['mask'], mask)
            samp_dict['id']    = np.append(samp_dict['id'], ID)
            samp_dict['filt']  = np.append(samp_dict['filt'], filt)

    stacking_sample_DF = pd.DataFrame.from_dict(samp_dict, orient='columns')

    write_term_file(stacking_sample_DF)
    write_term_file('\n\n\n')
            
    print
    print
    print
    print 'Number of galaxies that should be stacked: ', colored(len(ids), 'green')
    print 'Number of galaxies with spectral data found: ', colored(len(samp_dict['fpath']) / 3, 'green')
    print
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print

    if return_DF == True:
        return stacking_sample_DF

    else:
        del stacking_sample_DF
        return samp_dict



start_time = time.time()

cwd = os.getcwd()


write_term_file('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
write_term_file(("This program will estimate the stack's uncertainty spectrum\n"
                 "(either purely statistically or with cosmic variance included\n"
                 "through bootstrap resampling)\n"
                 "THIS CODE IS IN DEVELOPMENT."))
write_term_file('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'+'\n\n')

print 'The path and current working directory are: ', colored(cwd, 'green')
print

logfile = cwd + '/logfiles/stack_uncertainty_est_'+norm_eline+'_'+stack_meth+'_'+time.strftime('%m-%d-%Y')+'.log'
f = open(logfile, 'w')

filepath = cwd + '/intermed_spectrum_tables_' + norm_eline + '_' + stack_meth + '/'

if inc_cos_var == True:
    samples_type = 'bootstrap'
else:
    samples_type = 'statistical'
    
tab_stacks_opath = cwd + '/' + samples_type + '_samples_' + norm_eline + '_' + stack_meth + '/'


samp_table = fr.rc(stack_samp)

write_term_file("THE STACKED SAMPLE'S IDs ARE GIVEN IN THE FITS FILE: "+stack_samp)
write_term_file('THE NUMBER OF COMPOSITE SPECTRA TO BE GENERATED: '+str(ncomp))
write_term_file('STACKING METHOD: '+stack_meth)
write_term_file('EMISSION LINE USED TO NORMALIZE SPECTRA: '+norm_eline)
write_term_file('CORRECTING FOR DUST EXTINCTION: '+str(dust_corr))
write_term_file('COSMIC VARIANCE INCLUDED IN THE UNCERTAINTY ESTIMATE: '+str(inc_cos_var)+'\n\n\n')



eline_lum_table = pd.read_csv(filepath + 'sample_parameters_' + norm_eline + '_' + stack_meth + '.txt', delim_whitespace=True, header=0, index_col=0, \
                              usecols=['ID', 'Mask', norm_eline+'_Lum', norm_eline+'_Lum_Err'], \
                              dtype={'ID': np.string_, 'Mask': np.string_, norm_eline+'_Lum': np.float64, norm_eline+'_Lum_Err': np.float64} \
                             )[['Mask', norm_eline+'_Lum', norm_eline+'_Lum_Err']]

eline_lum_table = eline_lum_table[eline_lum_table[norm_eline+'_Lum'].notna()]

write_term_file('EMISSION-LINE LUMINOSITY TABLE:\n')
write_term_file(eline_lum_table)
write_term_file('\n\n\n')

resamp_wave_params = pd.read_csv(filepath + 'resampled_wavelength_parameters.txt', delim_whitespace=True, header=None, comment='#', \
                                 names=['Min Wavelength','Max Wavelength','RF Dispersion'], index_col=False, \
                                 dtype={'Min Wavelength': np.float64, 'Max Wavelength': np.float64, 'RF Dispersion': np.float64} \
                                )[['Min Wavelength', 'Max Wavelength', 'RF Dispersion']]

resamp_wave_params.set_index(pd.Index(['YJ', 'JH', 'HK'], name='Filters'), inplace=True)

write_term_file('RESAMPLED WAVELENGTH PARAMETERS TO BE USED FOR ALL COMPOSITE STACKS:\n')
write_term_file(resamp_wave_params)
write_term_file('\n\n\n')


resampled_spectra = OrderedDict.fromkeys(['YJ', 'JH', 'HK'])

for filts in resampled_spectra.keys():
    resampled_spectra[filts] = OrderedDict.fromkeys(['New_Wavelengths', 'New_Luminosities', 'CS_Luminosities'])

    new_wavelengths = np.arange(resamp_wave_params.loc[filts,'Min Wavelength'], \
                                resamp_wave_params.loc[filts,'Max Wavelength'] + resamp_wave_params.loc[filts,'RF Dispersion'], \
                                resamp_wave_params.loc[filts,'RF Dispersion'] \
                               )

    resampled_spectra[filts]['New_Wavelengths']  = new_wavelengths
    resampled_spectra[filts]['New_Luminosities'] = np.zeros((len(samp_table), len(new_wavelengths)))
    resampled_spectra[filts]['CS_Luminosities']  = np.zeros((ncomp, len(new_wavelengths)))

##############################################################

for iter_ in range(ncomp):

    if inc_cos_var == True:
        
        write_term_file('SAMPLE: '+str(iter_ + 1)+'\n')

        boot_samp_ind = np.random.randint(0, len(samp_table), size=len(samp_table))

        samp_ids   = samp_table['ID'][boot_samp_ind]
        samp_masks = samp_table['Mask'][boot_samp_ind]

        if len(samp_ids) != len(samp_table):
            raise ValueError('A bootstrap sample should have the same number of galaxies in it as the original sample')

        stacking_sample = create_samp_cat(samp_ids, samp_masks, filepath)

        write_term_file('GALAXIES NOT INCLUDED IN THIS BOOTSTRAP SAMPLE:\n')
        
        gals_not_in_samp = [gal_id for gal_id in samp_table['ID'] if gal_id not in samp_ids]

        for gal in gals_not_in_samp:
            write_term_file(gal)

        write_term_file('\n\n\n')

    else:
        if iter_ == 0:
            samp_ids   = samp_table['ID']
            samp_masks = samp_table['Mask']
            stacking_sample = create_samp_cat(samp_ids, samp_masks, filepath)
        else:
            pass
    
    #sys.exit()
##############################################################


    sample_params = pd.DataFrame(index=range(1, len(samp_ids)+1), \
                                 columns=['ID', 'Mask', norm_eline+'_Lum', norm_eline+'_Lum_Err', norm_eline+'_Lum_Pert'])
    
    seen_galaxy = []

    prev_id = ''
    gal_num, filt_cons = 1, 1
    yj_idx, jh_idx, hk_idx = 0, 0, 0


    for i, file_path in enumerate(stacking_sample['fpath']):

        mask   = stacking_sample['mask'][i]
        id_num = stacking_sample['id'][i]
        filt   = stacking_sample['filt'][i]

        fname  = file_path[len(filepath):]

        if i == 0:
            pass
        
        else:
            if id_num == prev_id:
                if filt_cons < 3:
                    filt_cons += 1
                elif filt_cons == 3:
                    gal_num  += 1
                    filt_cons = 1
            else:
                if id_num != '370':
                    gal_num  += 1
                    filt_cons = 1
                else:
                    filt_cons += 1

        print colored('--> ','cyan',attrs=['bold'])+'Preparing spectrum in file '+colored(fname,'white')+' for resampling...'
        print
        print 'ID: ', colored(id_num, 'green')
        print 'Mask: ', colored(mask, 'green')
        print 'Filter: ', colored(filt, 'green')
        print 'Galaxy in stack: ', colored(str(gal_num)+'/'+str(len(samp_ids)), 'green')
        print

   
        rest_waves, luminosities, lum_errs = np.loadtxt(file_path, comments='#', usecols=(1,4,5), dtype='float', unpack=True)

        if len(luminosities) != len(lum_errs) or len(luminosities) != len(rest_waves):
            raise Exception('Rest-frame wavelength array, luminosity array, and luminosity error array must all be the same length')
        

        pert_lums = np.add(luminosities, np.multiply(lum_errs, np.random.randn(len(lum_errs))))  ##I have perturbed the spectra

        if gal_num not in seen_galaxy:
            eline_lum = eline_lum_table.loc[id_num, norm_eline+'_Lum']
            eline_lum_error = eline_lum_table.loc[id_num, norm_eline+'_Lum_Err']

            pert_eline_lum  = eline_lum_error * np.random.randn() + eline_lum ##I have perturbed the normalizing emission line luminosity

            print colored('-> ','magenta')+'Writing perturbed emission-line luminosity to PANDAS DataFrame to be considered later...'
            print

            sample_params.loc[gal_num] = pd.Series([id_num, mask, eline_lum, eline_lum_error, pert_eline_lum], index=sample_params.columns)

            seen_galaxy.append(gal_num)



        print 'Emission line with which the spectrum will be normalized: ', colored(norm_eline,'green')
        print 'Measured emission-line luminosity (NOT dust-corrected): ', colored('%.5e' % eline_lum,'green'), '+/-', colored('%.5e' % eline_lum_error,'green')
        print 'Perturbed emission-line luminosity: ', colored('%.5e' % pert_eline_lum,'green')
        print


        pert_lum_norm = sf.normalize_spectra(pert_lums, norm_eline, pert_eline_lum)  ## I have normalized the perturbed spectrum with the perturbed emission-line luminosity
        

        if (mask == 'a1689_z1_1' and filt == 'Y') or (mask != 'a1689_z1_1' and filt == 'J'):

            resampled = sf.resample_spectra(resampled_spectra['YJ']['New_Wavelengths'], rest_waves, pert_lum_norm, fill=0., verbose=True)
            resampled_spectra['YJ']['New_Luminosities'][yj_idx] = resampled[:,1]
            
            yj_idx += 1

        elif (mask == 'a1689_z1_1' and filt == 'J') or (mask != 'a1689_z1_1' and filt == 'H'):

            resampled = sf.resample_spectra(resampled_spectra['JH']['New_Wavelengths'], rest_waves, pert_lum_norm, fill=0., verbose=True)
            resampled_spectra['JH']['New_Luminosities'][jh_idx] = resampled[:,1]

            jh_idx += 1

        elif (mask == 'a1689_z1_1' and filt == 'H') or (mask != 'a1689_z1_1' and filt == 'K'):

            resampled = sf.resample_spectra(resampled_spectra['HK']['New_Wavelengths'], rest_waves, pert_lum_norm, fill=0., verbose=True)
            resampled_spectra['HK']['New_Luminosities'][hk_idx] = resampled[:,1]

            hk_idx += 1
            

        prev_id = id_num

        

    if stack_meth == 'average':
        sample_eline_lum = sample_params[norm_eline+'_Lum_Pert'].mean()

    elif stack_meth == 'median':
        sample_eline_lum = sample_params[norm_eline+'_Lum_Pert'].median()
        
    print
    print
    
    write_term_file('PERTURBED LUMINOSITIES OF THE EMISSION LINE USED TO NORMALIZE THE SPECTRA:\n')
    write_term_file(sample_params)
    write_term_file('\n\nThe '+stack_meth+' perturbed '+norm_eline+' luminosity of the sample is: '+str(sample_eline_lum)+'\n\n')
    

    for bands in resampled_spectra.keys():

        print colored('--> ','cyan',attrs=['bold'])+'Stacking the spectra of this sample and finalizing the stack...'
        print

        fname_out = 'sample_'+str(iter_+1)+'_stacked_spectrum_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_noDC.txt'

        stacked_luminosities = sf.combine_spectra(resampled_spectra[bands]['New_Luminosities'], stack_meth, axis=0)

        if len(stacked_luminosities) != len(resampled_spectra[bands]['New_Wavelengths']):
            raise Exception(('Array of stacked luminosity values is not the same length as the array of wavelengths.\n'
                             '"Axis" keyword in "sf.combine_spectra" call is likely wrong'))

        final_luminosities = sf.multiply_stack_by_eline(stacked_luminosities, stack_meth, norm_eline, sample_eline_lum)
        final_wavelengths  = resampled_spectra[bands]['New_Wavelengths']

        resampled_spectra[bands]['CS_Luminosities'][iter_] = final_luminosities

        stacked_spectrum_vals = np.array([final_wavelengths, final_luminosities]).T

        np.savetxt(tab_stacks_opath + fname_out, stacked_spectrum_vals, fmt=['%10.5f','%6.5e'], delimiter='\t', newline='\n', comments='#', \
                   header=fname_out+'\n'+stack_meth+' Lum: '+str('%.5e' % sample_eline_lum)+'\n'+ \
                   'Rest-frame wavelength (A) | Luminosity (erg/s/A)'+'\n' \
                  )

        print
        print colored(fname_out,'green')+' written!'
        print
        
print
print 

        
for bands in resampled_spectra.keys():

    print colored('--> ','cyan', attrs=['bold'])+'Calculating the standard deviation of luminosities in each pixel for the '+colored(bands,'magenta')+'-band composites'
    print

    fname_out = samples_type+'_std_by_pixel_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_noDC.txt'

    std_arr = np.std(resampled_spectra[bands]['CS_Luminosities'], axis=0, dtype=np.float64)

    wavelengths = resampled_spectra[bands]['New_Wavelengths']

    if len(wavelengths) != len(std_arr):
        raise Exception('STD array is not the same length as the array of wavelengths. "Axis" keyword in "np.std" call is likely wrong')
    
    comp_unc_spectra = np.array([wavelengths, std_arr]).T

    np.savetxt(fname_out, comp_unc_spectra, fmt=['%10.5f', '%6.5e'], delimiter='\t', newline='\n', comments='#', \
               header=fname_out+'\n\n'+'Rest-frame Wavelength (A) | Luminosity 1-Sigma Uncertainty (erg/s/A)'+'\n' \
              )

    print
    print colored(fname_out,'green')+' '+colored('written!','red',attrs=['bold'])
    print
    print
    print

end_time = time.time()
tot_time = end_time - start_time

print 'Total run-time for '+colored(ncomp,'cyan')+' samples:  ',colored('--- %.1f seconds ---' % (tot_time),'cyan'),'===>',colored('--- %.1f minutes ---' % (tot_time / 60.),'cyan')
print
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print
print
