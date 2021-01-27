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

THIS INVOLVES STACKING INDIVIDUAL, SAMPLED, SLIT-LOSS-CORRECTED SPECTRA.
- Spectra will be shifted to the rest frame.
- Spectra will have their flux densities converted to luminosity densities.
- Spectra MAY be dust-corrected. This option is specified in the call to this script.
- Spectra MAY be normalized by an emission line given in the call to this script.
- Spectra will be resampled onto a wavelength grid with spacing equal to that at the {median, average} sample redshift (by filter).
- Spectra will be combined via the method given in the call to this script.
- Spectra MAY be multiplied by the {median, average} line luminosity corresponding to the emission line used for normalization, if done.

FOR MORE INFO ON THE PROCEDURE IN THIS SCRIPT, SEE THE README."""  ## Have not made the README yet

))


parser.add_argument('-n', '--N_Comp', metavar='int', type=int, default=500, \
                    help='The number of composite spectra to generate')  ##Typically 500

parser.add_argument('-c', '--Cosmic_Var', action='store_true', \
                    help='If called, cosmic variance will be included through bootstrap resampling\n'
                         'Otherwise, the uncertainty spectrum will be statistical only')  ##If this option is called, this argument will be True.  Otherwise it's False

parser.add_argument('-d', '--Dust_Correct', action='store_true', \
                    help='If called, each individual spectrum will be dust-corrected.')

parser.add_argument('Norm_ELine', choices=['OIII5007', 'H-alpha'], \
                    help='The emission line name of the line used to normalize each spectrum')

parser.add_argument('Stacking_Method', choices=['median', 'average'], \
                    help='The method with which the spectra will be stacked')

parser.add_argument('Stacking_Sample', \
                    help='The FITS file with the spectrum IDs for stacking')



args = parser.parse_args()

ncomp       = args.N_Comp
uncert_comp = args.Cosmic_Var
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


def write_term_file(output, filename = 'bootstrap_samples_'+norm_eline+'_'+stack_meth):
    term_only  = sys.stdout
    sys.stdout = Logger(logname=cwd+'/logfiles/'+filename, mode='a')
    print output
    sys.stdout = term_only

    return



start_time = time.time()


print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print colored(("This program will estimate the stack's uncertainty spectrum\n"
               "through bootstrap resampling (uncertainty either purely\n"
               "statistical or with cosmic variance included)"
               "THIS CODE IS IN DEVELOPMENT."
              ), 'cyan',attrs=['bold'])
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print

cwd = os.getcwd()

filename = cwd + '/logfiles/bootstrap_samples_'+norm_eline+'_'+stack_meth+'_'+time.strftime('%m-%d-%Y')+'.log'
f = open(filename, 'w')

fpath   = cwd + '/intermed_spectrum_tables_' + norm_eline + '_' + stack_meth + '/'
bs_path = cwd + '/bootstrap_samples_' + norm_eline + '_' + stack_meth + '/'

print 'The path and current working directory are: ', colored(cwd, 'green')
print

samp_table = fr.rc(stack_samp)

write_term_file("THE STACKED SAMPLE'S IDs ARE GIVEN IN THE FITS FILE: "+stack_samp)
write_term_file('THE '+str(ncomp)+' BOOTSTRAP SAMPLES FOR WHICH COMPOSITE SPECTRA WILL BE MADE')
write_term_file('STACKING METHOD: '+stack_meth)
write_term_file('EMISSION LINE TO NORMALIZE BY: '+norm_eline)
write_term_file('CORRECTING FOR DUST EXTINCTION: '+dust_corr)
write_term_file('COSMIC VARIANCE INCLUDED IN THE UNCERTAINTY ESTIMATE: '+uncert_comp+'\n\n\n')



eline_lum_table = pd.read_csv(fpath + 'sample_parameters_' + norm_eline + '_' + stack_meth + '.txt', delim_whitespace=True, header=0, index_col=0, \
                              usecols=['ID', 'Mask', norm_eline+'_Lum', norm_eline+'_Lum_Err'], \
                              dtype={'ID': np.string_, 'Mask': np.string_, norm_eline+'_Lum': np.float64, norm_eline+'_Lum_Err': np.float64} \
                             )[['Mask', norm_eline+'_Lum', norm_eline+'_Lum_Err']]

eline_lum_table = eline_lum_table[eline_lum_table[norm_eline+'_Lum'].notna()]

write_term_file('EMISSION-LINE LUMINOSITY TABLE:\n')
write_term_file(eline_lum_table)
write_term_file('\n\n\n')

resamp_wave_params = pd.read_csv(fpath + 'resampled_wavelength_parameters.txt', delim_whitespace=True, header=None, \
                                 names=['Min Wavelength','Max Wavelength','Rest-Frame Dispersion (A/pix)'], index_col=False, \
                                 dtype={'Min Wavelength': np.float64, 'Max Wavelength': np.float64, 'Rest-Frame Dispersion (A/pix)': np.float64}, comment='#' \
                                )[['Min Wavelength', 'Max Wavelength', 'Rest-Frame Dispersion (A/pix)']]

resamp_wave_params.set_index(pd.Index(['YJ', 'JH', 'HK']), inplace=True)

write_term_file('RESAMPLED WAVELENGTH PARAMETERS TO BE USED FOR ALL COMPOSITE STACKS:\n')
write_term_file(resamp_wave_params)
write_term_file('\n\n\n')


resampled_spectra = OrderedDict.fromkeys(['YJ', 'JH', 'HK'])

for key in resampled_spectra.keys():
    resampled_spectra[key] = OrderedDict.fromkeys(['New_Wavelengths', 'New_Luminosities', 'BS_Luminosities'])

    resampled_spectra[key]['New_Wavelengths'] = np.arange(resamp_wave_params.loc[key,'Min Wavelength'], \
                                                          resamp_wave_params.loc[key,'Max Wavelength'] + resamp_wave_params.loc[key,'Rest-Frame Dispersion (A/pix)'], \
                                                          resamp_wave_params.loc[key,'Rest-Frame Dispersion (A/pix)'] \
                                                         )
    
    resampled_spectra[key]['New_Luminosities'] = np.zeros((len(samp_table), len(resampled_spectra[key]['New_Wavelengths'])))
    resampled_spectra[key]['BS_Luminosities']  = np.zeros((ncomp, len(resampled_spectra[key]['New_Wavelengths'])))

##############################################################

for iter_ in range(ncomp):

    boot_samp_ind   = np.random.randint(0, len(samp_table), size=len(samp_table))

    boot_samp_ids   = samp_table['ID'][boot_samp_ind]
    boot_samp_masks = samp_table['Mask'][boot_samp_ind]

    if len(boot_samp_ids) != len(samp_table):
        raise ValueError('A bootstrap sample should have the same number of galaxies in it as the original sample')
    

    stacking_sample = OrderedDict.fromkeys(['fpath','mask','id','filt'])

    for key in stacking_sample.keys():
        stacking_sample[key] = np.array([])

    for idx, id_num in enumerate(boot_samp_ids):

        mask = boot_samp_masks[idx]
        
        file_names = sorted(glob(fpath + mask + '.?.' + id_num + '.rest-frame.lum.norm-lum.not-resampled.txt'))

        if id_num == '1197_370':
            file_names = file_names + [fpath + 'a1689_z1_1.H.370.rest-frame.lum.norm-lum.not-resampled.txt']

        for file_ in file_names:
           
            fname = file_[len(fpath):]
            
            print 'Parsing '+colored(fname,'white')+' into stacking sample by filter'

            filt  = fname[len(mask)+1]
            ID    = fname[len(mask)+3: -42]
            
            stacking_sample['fpath'] = np.append(stacking_sample['fpath'], file_)
            stacking_sample['mask']  = np.append(stacking_sample['mask'], mask)
            stacking_sample['id']    = np.append(stacking_sample['id'], ID)
            stacking_sample['filt']  = np.append(stacking_sample['filt'], filt)
            

    stacking_sample_DF = pd.DataFrame.from_dict(stacking_sample, orient='columns')

    print
    print

    write_term_file('SAMPLE: '+str(iter_ + 1)+'\n')
    write_term_file(stacking_sample_DF)
    write_term_file('\n\n\n')
            
    print
    print
    print
    print 'Number of galaxies that should be stacked: ', colored(len(samp_table), 'green')
    print 'Number of galaxies with spectral data found: ', colored(len(stacking_sample['fpath']) / 3, 'green')
    print
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print
    
    #sys.exit()
##############################################################


    sample_params = pd.DataFrame(index=samp_table['ID'], columns=['Mask', norm_eline+'_Lum', norm_eline+'_Lum_Err', norm_eline+'_Lum_Sample'])

    seen_idmask = []
    
    yj_idx, jh_idx, hk_idx = 0, 0, 0


    for i, file_path in enumerate(stacking_sample['fpath']):

        mask   = stacking_sample['mask'][i]
        id_num = stacking_sample['id'][i]
        filt   = stacking_sample['filt'][i]

        fname  = file_path[len(fpath):]

        print colored('--> ','cyan',attrs=['bold'])+'Preparing spectrum in file '+colored(fname,'white')+' for resampling...'
        print
        print 'Mask: ', colored(mask,'green')
        print 'ID: ', colored(id_num,'green')
        print 'Filter: ', colored(filt,'green')
        print

   
        rest_waves, luminosities, lum_errs = np.loadtxt(file_path, comments='#', usecols=(1,4,5), dtype='float', unpack=True)

        if len(luminosities) != len(lum_errs) or len(luminosities) != len(rest_waves):
            raise Exception('Rest-frame wavelength array, luminosity array, and luminosity error array must all be the same length')
        

        pert_lums = np.add(luminosities, np.multiply(lum_errs, np.random.randn(len(lum_errs))))  ##I have perturbed the spectra

        if (id_num, mask) not in seen_idmask and (id_num, mask) != ('370', 'a1689_z1_1'):
            eline_lum = eline_lum_table.loc[id_num, norm_eline+'_Lum']
            eline_lum_error = eline_lum_table.loc[id_num, norm_eline+'_Lum_Err']

            pert_eline_lum  = eline_lum_error * np.random.randn() + eline_lum ##I have perturbed the normalizing emission line luminosity

            print colored('-> ','magenta')+'Writing perturbed emission-line luminosity to PANDAS DataFrame to be considered later...'

            sample_params.loc[id_num] = pd.Series([mask, eline_lum, eline_lum_error, pert_eline_lum], index=sample_params.columns)

            seen_idmask = seen_idmask + [(id_num,mask)]

        elif (id_num, mask) in seen_idmask:
            pert_eline_lum = sample_params.loc[id_num, norm_eline+'_Lum_Sample']

        elif (id_num, mask) == ('370', 'a1689_z1_1'):
            pert_eline_lum = sample_params.loc['1197_370', norm_eline+'_Lum_Sample']

        print 'Emission line with which the spectrum will be normalized: ', colored(norm_eline,'green')
        print 'Measured emission-line luminosity (NOT dust-corrected): ', colored('%.5e' % eline_lum,'green'), '+/-', colored('%.5e' % eline_lum_error,'green')
        print 'Perturbed emission-line luminosity: ', colored('%.5e' % pert_eline_lum,'green')
        print


        pert_lum_norm = sf.normalize_spectra(pert_lums, norm_eline, pert_eline_lum)  ## I have normalized the perturbed spectrum with the perturbed emission-line luminosity
        

        if (mask == 'a1689_z1_1' and filt == 'Y') or (mask != 'a1689_z1_1' and filt == 'J'):

            resampled = sf.resample_spectra(resampled_spectra['YJ']['New_Wavelengths'], rest_waves, pert_lum_norm, lum_errors=None, fill=0., verbose=True)
            resampled_spectra['YJ']['New_Luminosities'][yj_idx] = resampled[:,1]



            
            yj_idx += 1

        elif (mask == 'a1689_z1_1' and filt == 'J') or (mask != 'a1689_z1_1' and filt == 'H'):

            resampled = sf.resample_spectra(resampled_spectra['JH']['New_Wavelengths'], rest_waves, pert_lum_norm, lum_errors=None, fill=0., verbose=True)
            resampled_spectra['JH']['New_Luminosities'][jh_idx] = resampled[:,1]

            jh_idx += 1

        elif (mask == 'a1689_z1_1' and filt == 'H') or (mask != 'a1689_z1_1' and filt == 'K'):

            resampled = sf.resample_spectra(resampled_spectra['HK']['New_Wavelengths'], rest_waves, pert_lum_norm, lum_errors=None, fill=0., verbose=True)
            resampled_spectra['HK']['New_Luminosities'][hk_idx] = resampled[:,1]

            hk_idx += 1


    if stack_meth == 'average':
        sample_eline_lum = sample_params[norm_eline+'_Lum_Sample'].mean()

    elif stack_meth == 'median':
        sample_eline_lum = sample_params[norm_eline+'_Lum_Sample'].median()
        
    print
    print
    
    write_term_file('LUMINOSITY SAMPLES DRAWN FOR THE EMISSION LINE USED TO NORMALIZE THE SPECTRA:\n')
    write_term_file(sample_params)
    write_term_file('\n\nThe '+stack_meth+' perturbed '+norm_eline+' luminosity of the bootstrap sample is: '+str(sample_eline_lum)+'\n\n\n')
    

    for bands in resampled_spectra.keys():

        print colored('--> ','cyan',attrs=['bold'])+'Stacking the spectra of this bootstrap sample and finalizing the stacks...'
        print

        fname_out = 'bootstrap_sample_'+str(iter_+1)+'_stacked_spectrum_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_noDC.txt'

        stacked_luminosities = sf.combine_spectra(resampled_spectra[bands]['New_Luminosities'], stack_meth, axis=0)

        final_luminosities = sf.multiply_stack_by_eline(stacked_luminosities, stack_meth, norm_eline, sample_eline_lum)
        final_wavelengths  = resampled_spectra[bands]['New_Wavelengths']

        resampled_spectra[bands]['BS_Luminosities'][iter_] = final_luminosities

        stacked_spectrum_vals = np.array([final_wavelengths, final_luminosities]).T

        np.savetxt(bs_path + fname_out, stacked_spectrum_vals, fmt=['%10.5f','%6.5e'], delimiter='\t', newline='\n', comments='#', \
                   header=fname_out+'\n'+stack_meth+' Lum: '+str('%.5e' % sample_eline_lum)+'\n'+ \
                   'Rest-frame wavelength (A) | Luminosity (erg/s/A)'+'\n' \
                  )

        print
        print colored(fname_out,'green')+' written!'
        print
        
print
print 

        
for bands in resampled_spectra.keys():

    print colored('--> ','cyan', attrs=['bold'])+'Calculating the standard deviation of luminosities in each pixel for the '+colored(bands,'magenta')+'-band bootstrap composites'
    print

    fname_out = 'bootstrap_std_by_pixel_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_noDC.txt'

    std_arr = np.std(resampled_spectra[bands]['BS_Luminosities'], axis=0, dtype=np.float64)

    wavelengths = resampled_spectra[bands]['New_Wavelengths']

    if len(wavelengths) != len(std_arr):
        raise Exception('STD array is not the same length as the array of wavelengths. "Axis" keyword in "np.std" call is likely wrong')
    
    comp_err_spectra = np.array([wavelengths, std_arr]).T

    np.savetxt(fname_out, comp_err_spectra, fmt=['%10.5f', '%6.5e'], delimiter='\t', newline='\n', comments='#', \
               header=fname_out+'\n\n'+'Rest-frame Wavelength (A) | Luminosity Uncertainty (erg/s/A)'+'\n' \
              )

    print
    print colored(fname_out,'green')+' '+colored('written!','red',attrs=['bold'])
    print
    print
    print

end_time = time.time()
tot_time = end_time - start_time

print 'Total run-time for '+colored(ncomp,'cyan')+' bootstrap samples:  ',colored('--- %.1f seconds ---' % (tot_time),'cyan'),'===>',colored('--- %.1f minutes ---' % (tot_time / 60.),'cyan')
print
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print
print
