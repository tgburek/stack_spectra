#! /usr/bin/env python

import os
import re
import sys
import time
import numpy as np
import pandas as pd
import stacking_functions as sf
from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
from glob import glob
from collections import OrderedDict
from termcolor import colored
import fits_readin as fr


print()

class HelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    pass


parser = ArgumentParser(formatter_class=HelpFormatter, description=(
    
"""ESTIMATE UNCERTAINTY SPECTRUM OF STACK.
UNCERTAINTY CAN BE DESIGNATED TO BE STATISTICAL ONLY OR INCLUDE COSMIC VARIANCE THROUGH BOOTSTRAP RESAMPLING.

THIS INVOLVES STACKING INDIVIDUAL, PERTURBED, SLIT-LOSS-CORRECTED SPECTRA.
- Spectra have already been shifted to the rest frame.
- Spectra have had their flux densities converted to luminosity densities.
- Spectra MAY be dust-corrected. This option is specified in the call to this script (Option present but not currently usable).
- Spectra will be normalized by either perturbed continuum luminosity density or perturbed integrated luminosity of an emission line given in the call to this script.
- Spectra will be resampled onto a wavelength grid with a dispersion (A/pix) equal to the MOSFIRE dispersion de-redshifted by the {median, average} sample redshift (by filter).
- Spectra will be combined via the method (median, average) given in the call to this script.
- Stacked spectrum will be multiplied by the sample {median, average} perturbed continuum luminosity density or perturbed integrated line luminosity (depending on normalization choice).

FOR MORE INFO ON THE PROCEDURE IN THIS SCRIPT, SEE THE README (NOT YET CREATED)."""  ## Have not made the README yet

))


parser.add_argument('-n', '--N_Comp', metavar='int', type=int, default=500, \
                    help='The number of composite spectra to generate')  ##Typically 500

parser.add_argument('-c', '--Cosmic_Var', action='store_true', \
                    help='If called, cosmic variance will be included through bootstrap resampling\n'
                         'Otherwise, the uncertainty spectrum will be statistical only')  ##If this option is called, this argument will be True.  Otherwise it's False

parser.add_argument('-d', '--Dust_Correct', action='store_true', \
                    help='If called, each individual spectrum will be dust-corrected (not currently supported)')

parser.add_argument('Norm_Feature', choices=['OIII5007', 'H-alpha', 'Lum_Density'], \
                    help='The feature used to normalize each spectrum\n'
                         "Current options include an emission-line's integrated luminosity\n"
                         "  or a continuum's luminosity density at a given wavelength")

parser.add_argument('Stacking_Method', choices=['median', 'average'], \
                    help='The method with which the spectra will be stacked')

parser.add_argument('Stacking_Sample', \
                    help='The FITS file with the spectroscopic IDs to be stacked')

parser.add_argument('Instrument', choices=['MOSFIRE', 'LRIS'], \
                    help='Instrument data were collected with')



args = parser.parse_args()

ncomp       = args.N_Comp
inc_cos_var = args.Cosmic_Var
dust_corr   = args.Dust_Correct
norm_feat   = args.Norm_Feature
stack_meth  = args.Stacking_Method
stack_samp  = args.Stacking_Sample
Instrument  = args.Instrument


if inc_cos_var == True:
    samples_type = 'bootstrap'
else:
    samples_type = 'statistical'

emiss_lines_to_norm_by = ['OIII5007', 'H-alpha'] ##All current emission-line options to use for normalization

if norm_feat == 'Lum_Density':
    lum_density_wave = str(raw_input('Enter the wavelength, in Angstroms, at which the luminosity density is being derived: '))
    print()


if norm_feat in emiss_lines_to_norm_by:
    norm_feature = norm_feat+'_Lum'
    norm_feature_descr = norm_feat + ' integrated luminosity'


elif norm_feat == 'Lum_Density':
    norm_feature = 'Lum_Density_'+lum_density_wave
    norm_feature_descr = 'continuum luminosity density at '+lum_density_wave+' Angstroms'

norm_feature_err  = norm_feature+'_Err'
norm_feature_pert = norm_feature+'_Pert'


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


def write_term_file(output, filename = 'stack_uncertainty_est_'+norm_feat+'_'+stack_meth+'_'+samples_type):
    term_only  = sys.stdout
    sys.stdout = Logger(logname=cwd+'/logfiles/'+filename, mode='a')
    print(output)
    sys.stdout = term_only

    return


def create_samp_cat(ids, masks, dirpath, return_DF=False):
    samp_dict = OrderedDict.fromkeys(['fpath', 'mask', 'id', 'filt'])

    for key in samp_dict.keys():
        samp_dict[key] = np.array([])

    for id_num, mask in zip(ids, masks):
        print(dirpath)
        print(mask)
        print(id_num)
        mask = list(mask)
        mask[0] = 'A'
        mask = "".join(mask)
        if Instrument == 'MOSFIRE':file_names = sorted(glob(str(dirpath) + str(mask) + '.?.' + str(id_num) + '.rest-frame.lum.norm-lum.not-resampled.txt'))
        else: file_names = sorted(glob(str(dirpath) + str(mask) + '.rest_UV.' + str(id_num) + '.rest-frame.lum.norm-lum.not-resampled.txt'))
        print(file_names)
        for file_ in file_names:
            print(file_)
            fname = file_[len(dirpath):]
            if Instrument == "MOSFIRE":filt  = fname[len(mask)+1]
            else: filt = fname.split('.')[1]
            ID    = id_num
            print(ID)
    
            samp_dict['fpath'] = np.append(samp_dict['fpath'], file_)
            samp_dict['mask']  = np.append(samp_dict['mask'], mask)
            samp_dict['id']    = np.append(samp_dict['id'], ID)
            samp_dict['filt']  = np.append(samp_dict['filt'], filt)

    stacking_sample_DF = pd.DataFrame.from_dict(samp_dict, orient='columns')

    write_term_file(stacking_sample_DF)
    write_term_file('\n\n\n')

    exp_stack_sample_size = len(ids)
    if Instrument == 'MOSFIRE': gals_with_data_found  = len(samp_dict['fpath']) / 3
    else: gals_with_data_found  = len(samp_dict['fpath'])
            
    print()
    print()
    print()
    print('Number of galaxies that should be stacked: ', colored(exp_stack_sample_size, 'green'))
    print('Number of galaxies with spectral data found: ', colored(gals_with_data_found, 'green'))
    print()
    print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print()

    if exp_stack_sample_size != gals_with_data_found:
        raise ValueError(('The number of galaxies that should have their spectra stacked does not match the number of spectra found\n'
                          '(number of spectra found divided by 3 to account for the three filters considered)'))

    if return_DF == True:
        return stacking_sample_DF

    else:
        del stacking_sample_DF
        return samp_dict



start_time = time.time()

cwd = os.getcwd()

logfile = cwd + '/logfiles/stack_uncertainty_est_'+norm_feat+'_'+stack_meth+'_'+samples_type+'_'+time.strftime('%m-%d-%Y')+'.log'
f = open(logfile, 'w')


write_term_file('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
write_term_file(("This program will estimate the stack's uncertainty spectrum\n"
                 "(either purely statistically or with cosmic variance included\n"
                 "through bootstrap resampling)\n"
                 "THIS CODE IS IN DEVELOPMENT."))
write_term_file('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'+'\n\n')



write_term_file('Review of options called and arguments given to this script:')
write_term_file('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'+'\n')

write_term_file('OPTIONS:')
write_term_file('-> THE NUMBER OF COMPOSITE SPECTRA TO BE GENERATED: ', colored(ncomp, 'green'))
write_term_file('-> COSMIC VARIANCE INCLUDED IN THE UNCERTAINTY ESTIMATE: ', colored(inc_cos_var, 'green'))
write_term_file('-> CORRECTING FOR DUST EXTINCTION: ', colored(dust_corr, 'green'), '\n')

write_term_file('ARGUMENTS:')
write_term_file("-> THE STACKED SAMPLE'S IDs ARE GIVEN IN THE FITS FILE: ", colored(stack_samp, 'green'))
write_term_file('-> STACKING METHOD: ', colored(stack_meth, 'green'))
write_term_file('-> SPECTRA NORMALIZED BY: ', colored(norm_feat, 'green'))
if norm_feat == 'Lum_Density':
    write_term_file('---> LUMINOSITY DENSITY TAKEN AT WAVELENGTH '+colored(lum_density_wave, 'green')+' ANGSTROMS')
write_term_file('-> THE INSTRUMENT USED TO OBTAIN THE SPECTRA: ', colored(Instrument, 'green'), '\n\n\n')




print('The path and current working directory are: ', colored(cwd, 'green'))
print()

filepath = cwd + '/intermed_stacking_output_' + norm_feat + '_' + stack_meth + '/tables/'
tab_stacks_opath = cwd + '/' + samples_type + '_samples_' + norm_feat + '_' + stack_meth + '/'

if os.path.isdir(tab_stacks_opath) == False:
    os.mkdir(tab_stacks_opath)
    print('Created directory: '+colored(tab_stacks_opath, 'white'))
    print()


samp_table = fr.rc(stack_samp)


norm_feat_table = pd.read_csv(filepath + 'sample_parameters_' + norm_feat + '_' + stack_meth + '.txt', delim_whitespace=True, header=0, index_col=0, \
                              usecols=['ID', 'Mask'] + [norm_feature, norm_feature_err], \
                              dtype={'ID': np.string_, 'Mask': np.string_, norm_feature: np.float64, norm_feature_err: np.float64} \
                             )[['Mask'] + [norm_feature, norm_feature_err]]

norm_feat_table = norm_feat_table[norm_feat_table[norm_feature].notna()]

norm_feat_table.index = norm_feat_table.index.astype('str')

write_term_file('NORMALIZATION FEATURE TABLE:\n')
write_term_file(norm_feat_table)
write_term_file('\n\n\n')

resamp_wave_params = pd.read_csv(filepath + 'resampled_wavelength_parameters.txt', delim_whitespace=True, header=None, comment='#', \
                                 names=['Min Wavelength','Max Wavelength','RF Dispersion'], index_col=False, \
                                 dtype={'Min Wavelength': np.float64, 'Max Wavelength': np.float64, 'RF Dispersion': np.float64} \
                                )[['Min Wavelength', 'Max Wavelength', 'RF Dispersion']]

if Instrument == 'MOSFIRE': resamp_wave_params.set_index(pd.Index(['YJ', 'JH', 'HK'], name='Filters'), inplace=True)
else: resamp_wave_params.set_index(pd.Index(['rest_UV'], name='Filters'), inplace=True)

write_term_file('RESAMPLED WAVELENGTH PARAMETERS TO BE USED FOR ALL COMPOSITE STACKS:\n')
write_term_file(resamp_wave_params)
write_term_file('\n\n\n')


if Instrument == 'MOSFIRE': resampled_spectra = OrderedDict.fromkeys(['YJ', 'JH', 'HK'])
else: resampled_spectra = OrderedDict.fromkeys(['rest_UV'])

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
                                 columns=['ID', 'Mask', norm_feature, norm_feature_err, norm_feature_pert])
    
    seen_galaxy = []

    prev_id = ''
    gal_num, filt_cons = 1, 1
    yj_idx, jh_idx, hk_idx, lris_idx = 0, 0, 0, 0


    for i, file_path in enumerate(stacking_sample['fpath']):

        mask   = stacking_sample['mask'][i]
        id_num = stacking_sample['id'][i]
        filt   = stacking_sample['filt'][i]
        print(colored(stacking_sample['filt'],'red'))

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
                gal_num  += 1
                filt_cons = 1
                

        print(colored('--> ','cyan',attrs=['bold'])+'Preparing spectrum in file '+colored(fname,'white')+' for resampling...')
        print()
        print('ID: ', colored(id_num, 'green'))
        print('Mask: ', colored(mask, 'green'))
        print('Filter: ', colored(filt, 'green'))
        print('Galaxy in stack: ', colored(str(gal_num)+'/'+str(len(samp_ids)), 'green'))
        print()

   
        rest_waves, luminosities, lum_errs = np.loadtxt(file_path, comments='#', usecols=(1,4,5), dtype='float', unpack=True)

        if len(luminosities) != len(lum_errs) or len(luminosities) != len(rest_waves):
            raise ValueError('Rest-frame wavelength array, luminosity array, and luminosity error array must all be the same length')
        

        pert_lums = np.add(luminosities, np.multiply(lum_errs, np.random.randn(len(lum_errs))))  ##I have perturbed the spectra

        if gal_num not in seen_galaxy:
            print(id_num, norm_feature)
            print(norm_feat_table[norm_feature].keys())

            norm_factor     = norm_feat_table.loc[id_num, norm_feature]
            norm_factor_err = norm_feat_table.loc[id_num, norm_feature_err]

            pert_norm_fact  = norm_factor_err * np.random.randn() + norm_factor ##I have perturbed the normalizing emission line luminosity


            print(colored('-> ','magenta')+'Writing perturbed normalization factor to PANDAS DataFrame to be considered later...')
            print()

            sample_params.loc[gal_num] = pd.Series([id_num, mask, norm_factor, norm_factor_err, pert_norm_fact], index=sample_params.columns)

            seen_galaxy.append(gal_num)



        print('Spectral feature with which the spectrum will be normalized: ', colored(norm_feature_descr,'green'))
        print('Normalization factor from file (NOT dust-corrected or de-magnified): ', colored('%.5e' % norm_factor,'green'), '+/-', colored('%.5e' % norm_factor_err,'green'))
        print('Perturbed normalization factor: ', colored('%.5e' % pert_norm_fact,'green'))
        print()


        pert_lum_norm = sf.normalize_spectra(pert_lums, norm_feature_descr, pert_norm_fact)  ## I have normalized the perturbed spectrum with the perturbed emission-line luminosity
        print(colored(filt,'red'))

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

        elif (mask == 'a1689_z1_1' and filt == 'rest_UV') or (mask != 'a1689_z1_1' and filt == 'rest_UV'):
            resampled = sf.resample_spectra(resampled_spectra['rest_UV']['New_Wavelengths'], rest_waves, pert_lum_norm, fill=0., verbose=True)
            print(resampled)
            resampled_spectra['rest_UV']['New_Luminosities'][lris_idx] = resampled[:,1]
            lris_idx += 1
            

        prev_id = id_num

        

    if stack_meth == 'average':
        sample_norm_fact = sample_params[norm_feature_pert].mean()

    elif stack_meth == 'median':
        sample_norm_fact = sample_params[norm_feature_pert].median()
        
    print
    print
    
    write_term_file('PERTURBED MEASUREMENTS OF THE SPECTRAL FEATURE USED TO NORMALIZE THE SPECTRA:\n')
    write_term_file(sample_params)
    write_term_file('\n\nThe '+stack_meth+' perturbed '+norm_feature_descr+' of the sample is: '+str(sample_norm_fact)+'\n\n')
    

    for bands in resampled_spectra.keys():

        print(colored('--> ','cyan',attrs=['bold'])+'Stacking the spectra of this sample and finalizing the stack...')
        print()

        fname_out = 'sample_'+str(iter_+1)+'_stacked_spectrum_'+bands+'-bands_'+stack_meth+'_'+norm_feat+'_noDC.txt'

        print(colored(resampled_spectra[bands]['New_Luminosities'],'green'))
        print(np.all(resampled_spectra[bands]['New_Luminosities'][0] == 0))
        print(np.all(resampled_spectra[bands]['New_Luminosities'][1] == 0))
        print(np.all(resampled_spectra[bands]['New_Luminosities'][2] == 0))

        stacked_luminosities = sf.combine_spectra(resampled_spectra[bands]['New_Luminosities'], stack_meth, axis=0)

        if len(stacked_luminosities) != len(resampled_spectra[bands]['New_Wavelengths']):
            raise ValueError(('Array of stacked luminosity values is not the same length as the array of wavelengths.\n'
                             '"Axis" keyword in "sf.combine_spectra" call is likely wrong'))

        final_luminosities = sf.multiply_stack_by_sfeat(stacked_luminosities, stack_meth, norm_feature_descr, sample_norm_fact)

        print(colored(final_luminosities,'red'))
        print(colored(stacked_luminosities,'cyan'))

        final_wavelengths  = resampled_spectra[bands]['New_Wavelengths']

        resampled_spectra[bands]['CS_Luminosities'][iter_] = final_luminosities

        stacked_spectrum_vals = np.array([final_wavelengths, final_luminosities]).T

        np.savetxt(tab_stacks_opath + fname_out, stacked_spectrum_vals, fmt=['%20.5f','%20.5e'], delimiter='\t', newline='\n', comments='#', \
                   header=fname_out+'\n'+stack_meth.capitalize()+' '+norm_feature_descr+' of sample: '+str('%.5e' % sample_norm_fact)+'\n'+ \
                   'Rest-frame wavelength (A) | Luminosity (erg/s/A)'+'\n' \
                  )

        print()
        print(colored(fname_out,'green')+' written!')
        print()
        
print()
print() 

        
for bands in resampled_spectra.keys():

    print(colored('--> ','cyan', attrs=['bold'])+'Calculating the standard deviation of luminosities in each pixel for the '+colored(bands,'magenta')+'-band composites')
    print()

    fname_out = samples_type+'_std_by_pixel_'+bands+'-bands_'+stack_meth+'_'+norm_feat+'_noDC.txt'

    std_arr = np.std(resampled_spectra[bands]['CS_Luminosities'], axis=0, dtype=np.float64)

    wavelengths = resampled_spectra[bands]['New_Wavelengths']

    if len(wavelengths) != len(std_arr):
        raise ValueError('STD array is not the same length as the array of wavelengths. "Axis" keyword in "np.std" call is likely wrong')
    
    comp_unc_spectra = np.array([wavelengths, std_arr]).T

    np.savetxt(fname_out, comp_unc_spectra, fmt=['%10.5f', '%6.5e'], delimiter='\t', newline='\n', comments='#', \
               header=fname_out+'\n\n'+'Rest-frame Wavelength (A) | Luminosity 1-Sigma Uncertainty (erg/s/A)'+'\n' \
              )

    print()
    print(colored(fname_out,'green')+' '+colored('written!','red',attrs=['bold']))
    print()
    print()
    print()

end_time = time.time()
tot_time = end_time - start_time

print('Total run-time for '+colored(ncomp,'cyan')+' samples:  ',colored('--- %.1f seconds ---' % (tot_time),'cyan'),'===>',colored('--- %.1f minutes ---' % (tot_time / 60.),'cyan'))
print()
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print()
print()
print()
