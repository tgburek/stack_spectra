#! /usr/bin/env python

import os
import re
import sys
import time
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as g
import seaborn as sns
import sns_setstyle
import fits_readin as fr
from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from glob import glob
from termcolor import colored
from matplotlib.backends.backend_pdf import PdfPages

print

class HelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    pass

parser = ArgumentParser(formatter_class=HelpFormatter, description=(

"""PLOT THE NEWLY-CREATED COMPOSITE SPECTRA (WITH BEST-FIT MODELS IF DESIRED). 
The entire wavelength coverage will be plotted as well as sections around emission lines of interest."""
    
))


parser.add_argument('-m', '--Multiple_Image_IDs', metavar='str', \
                    help='The IDs corresponding to the individual spectra of a multiply-imaged object\n'
                         '(ex. "ID1_ID2_ID3_...")')

parser.add_argument('-f', '--Plot_Fit', action='store_true', \
                    help='Plot best-fit model on top of stacked spectrum')

parser.add_argument('Norm_ELine',  choices=['OIII5007','H-alpha'], \
                    help='The emission-line name of the line used to normalize')

parser.add_argument('Stacking_Method', choices=['median','average','weighted-average'], \
                    help='The method with which the spectra were stacked')

parser.add_argument('Uncertainty', choices=['bootstrap', 'statistical'], \
                    help='How the uncertainty spectrum was calculated\n'
                         '(i.e. including cosmic variance or just statistically)')


args = parser.parse_args()

mult_imgs  = args.Multiple_Image_IDs
plot_fit   = args.Plot_Fit
norm_eline = args.Norm_ELine
stack_meth = args.Stacking_Method
uncert     = args.Uncertainty


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


def linear_cont(x, slope, yint):
    return slope * (x - min(x)) + yint
        
def gaussian(x, mean, amplitude, width, c=2.998e5):  ## "c" in km/s
    return amplitude * np.exp(-0.5*(np.square(x - mean) / np.square(width*mean/c)))


def plot_spectra(wavelengths, luminosities, lum_errors, eline_waves, eline_names, pdf, offset=None, disp_names=False, norm_fact=1., plt_ylim_top=None, plt_ylim_bot=None, leg_loc='best', \
                 plot_fit_model=False, fit_params=None, save_model_txt=False, stack_meth='', norm_eline='', uncert='', bands='', save_pickle=False, opath=''):

    fig, ax = plt.subplots()

    ax.step(wavelengths, np.divide(luminosities, norm_fact), where='mid', color='xkcd:sea blue', linewidth=0.7, label='Stacked Spectrum')

    if stack_meth == 'weighted-average':
        ax.step(wavelengths, lum_errors, where='mid', color='xkcd:gunmetal', linewidth=0.7, alpha=0.5, label='Error Spectrum')

        ylimits = ax.get_ylim()
        
        if ylimits[0] < -5.0e41:
            ax.set_ylim(bottom = -5.0e41)
            
    else:
            
        if offset is None or offset == 0.:
            raise ValueError('For the "median" or "average" stacking method, an offset from the x-axis must be supplied to plot the composite error spectrum')

        ax.fill_between(wavelengths, -offset, np.subtract(np.divide(lum_errors, norm_fact), offset), \
                        step='mid', facecolor='xkcd:gunmetal', linewidth=0.7, edgecolor='xkcd:gunmetal', alpha=0.5, label='Error Spectrum' \
                       )


    within_plotted_wrange = np.where((eline_waves >= min(wavelengths)) & (eline_waves <= max(wavelengths)))[0]
    eline_waves_to_plot   = eline_waves[within_plotted_wrange]
    eline_names_to_plot   = eline_names[within_plotted_wrange]

        
    if plot_fit_model:

        if fit_params is None or bands == '':
            raise ValueError('Values must be passed to "fit_params" and "bands" keywords if "plot_fit_model=True"')
        
        filt_idx  = int(np.where(fit_params['Filters'] == bands)[0])

        fit_waves = np.linspace(min(wavelengths), max(wavelengths), len(wavelengths)*5, endpoint=True)

        fit_model_hr = linear_cont(fit_waves, fit_params['Slope'][filt_idx], fit_params['y-Int'][filt_idx])
        fit_model_lr = linear_cont(wavelengths, fit_params['Slope'][filt_idx], fit_params['y-Int'][filt_idx])
        
        fit_amps_narrow = np.array([])

        for name, ewave in itertools.izip(eline_names_to_plot, eline_waves_to_plot):
            if name == 'OIII4959':
                fit_amps_narrow = np.append(fit_amps_narrow, fit_params['OIII5007_Narrow_Amp'][filt_idx] / 2.98)
            elif name == 'NII6548':
                fit_amps_narrow = np.append(fit_amps_narrow,fit_params['NII6583_Narrow_Amp'][filt_idx]/2.95)
            else:
                fit_amps_narrow = np.append(fit_amps_narrow,fit_params[name+'_Narrow_Amp'][filt_idx])
        for j, ewave in enumerate(eline_waves_to_plot):
            fit_model_hr += gaussian(fit_waves, ewave, fit_amps_narrow[j] / fit_params['Amplitude_Ratio'][filt_idx], fit_params['Sigma_Broad'][filt_idx]) + \
                            gaussian(fit_waves, ewave, fit_amps_narrow[j], fit_params['Sigma_Narrow'][filt_idx]) 
            fit_model_lr += gaussian(wavelengths, ewave, fit_amps_narrow[j] / fit_params['Amplitude_Ratio'][filt_idx], fit_params['Sigma_Broad'][filt_idx]) + \
                            gaussian(wavelengths, ewave, fit_amps_narrow[j], fit_params['Sigma_Narrow'][filt_idx])
        
        residuals = np.subtract(luminosities, fit_model_lr)

        if save_model_txt:
            G2model = np.array([wavelengths, fit_model_lr, residuals]).T
            fname_out = 'multiple_gaussian_fit_model_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_'+uncert+'.txt'
            np.savetxt(opath+fname_out, G2model, fmt=['%10.5f','%6.5e','%6.5e'], delimiter='\t', newline='\n', comments='#', \
                       header=fname_out+'\n'+'Rest Wave. (A) | Model Luminosity (erg/s/A) | Residuals (Data - Model)'+'\n' \
                      )
            print('-> '+colored(fname_out, 'green')+' written at location\n'+colored(opath,'white'))
            print()
 
        ax.plot(fit_waves, np.divide(fit_model_hr, norm_fact), color='red', linewidth=0.7, alpha=0.7, label='Model Spectrum')

        ax.step(wavelengths, np.subtract(np.divide(residuals, norm_fact), 1.5*offset), where='mid', color='xkcd:grass green', linewidth=0.7, label='Residuals')
        ax.axhline(y = -1.5*offset, color='xkcd:gunmetal', linewidth=0.5)
        
        handles, labels = ax.get_legend_handles_labels()
        # new_order = [1, 0, 2]
        new_order = [1, 0, 2, 3]  #With residuals plotted

        handles[:] = [handles[i] for i in new_order]  ## Re-orders the list in-place instead of creating a new variable
        labels[:]  = [labels[i] for i in new_order]

    ax.axhline(y = 0., color='xkcd:gunmetal', linewidth=0.5)
        
    if plt_ylim_top is not None:
        ax.set_ylim(top = plt_ylim_top)
    if plt_ylim_bot is not None:
        ax.set_ylim(bottom = plt_ylim_bot)

    for ewave, ename in zip(eline_waves_to_plot, eline_names_to_plot):
        ax.axvline(x = ewave, color='black', linestyle='--', linewidth=0.5, alpha=0.7)

        if disp_names:
            ylimits_final = ax.get_ylim()
            bbox_props = dict(boxstyle='square', fc='w', ec='w')
            ax.text(ewave, 0.79*ylimits_final[1], ename, ha='center', va='bottom', rotation=90, size=6., bbox=bbox_props) 

    norm_fact_exp = str('%e' % norm_fact)[str('%e' % norm_fact).find('e')+1:].strip('+')


    ax.minorticks_on()
    ax.tick_params(which='both', left=True, right=True, bottom=True, top=True)
    if plt_ylim_top == full_filt_ylim_top:
        ax.set_yticks(np.arange(plt_ylim_bot,full_filt_ylim_top+2.,2.))
    ax.legend(handles, labels, loc=leg_loc, fontsize='x-small', fancybox=True, frameon=True, framealpha=0.8, edgecolor='black')  ## If plot_fit_model is False, "handles" and "labels" is currently undefined
    ax.set_xlabel(r'Rest-Frame Wavelength ($\AA$)')
    ax.set_ylabel(r'$L_\lambda$ ($erg\ s^{-1}\ \AA^{-1}$) ($\times10^{'+norm_fact_exp+'}$)')
    ax.set_title('Stacked Spectra via '+stack_meth+' - Norm by '+norm_eline+' - Uncertainty: '+uncert.capitalize())

    plt.tight_layout()
    if save_pickle:
        pickle_fname = 'stacked_spectra_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_'+uncert+'_two_gaussian_model.fig.pickle'
        pickle.dump(fig,open(opath+pickle_fname,'wb'))
    pdf.savefig()
    plt.close(fig)

    return fit_waves, fit_model_hr, pdf

cwd = os.getcwd()

sys.stdout = Logger(logname=cwd+'/logfiles/plotting_stacked_spectra_'+stack_meth+'_'+norm_eline+'_plot_fit-'+str(plot_fit), mode='w')

print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print (colored(('This script will plot the newly-created stacked spectra.\n'
               'The entire wavelength coverage of a spectrum will be plotted\n'
               'as well as sections around emission lines of interest.\n'
               'Best-fit models of the spectra can also be overlaid.'
              ), 'cyan',attrs=['bold']))
print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print ()
print ()

print ('Review of options called and arguments given to this script:')
print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print ()
print ('Options:')
print ('-> Multiple-Image IDs: ', colored(mult_imgs,'cyan'))
print ('-> Plot best-fit spectral models: ', colored(plot_fit,'cyan'))
print ()
print ('Arguments:')
print ('-> Spectra normalized by: ', colored(norm_eline,'cyan'))
print ('-> Stacking method used: ', colored(stack_meth,'cyan'))
print ('-> Uncertainty calculation method: ', colored(uncert,'cyan'))
print ()
print ()

###################################
c = 2.998e5 ## km/s

if norm_eline == 'H-alpha':
    full_filt_ylim_top = 14.
    o2_ylim_top = 7.5
    paper_plot_ylim_top = 11.

elif norm_eline == 'OIII5007':
    full_filt_ylim_top = 12.
    o2_ylim_top = 6.5
    paper_plot_ylim_top = 10.
###################################


if mult_imgs is None:
    stacked_fnames = sorted(glob('stacked_spectrum_*-bands_'+stack_meth+'_'+norm_eline+'_noDC.txt'))[::-1]
    pp_name = 'stacked_spectra_by_'+stack_meth+'_'+norm_eline+'_'+uncert+'_two_gaussian_model.pdf'

else:
    stacked_fnames = sorted(glob('stacked_spectrum_*-bands_'+stack_meth+'_'+norm_eline+'_noDC_'+mult_imgs+'.txt'))[::-1]
    pp_name = 'stacked_spectra_by_'+stack_meth+'_'+norm_eline+'_'+mult_imgs+'.pdf'

print (colored('The stacked spectra to plot:','green'))
print (colored('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','green'))

for fname in stacked_fnames:
    print (fname)

print ()
print ()
    

if stack_meth == 'average' or stack_meth == 'median':
    uncert_fnames = sorted(glob(uncert+'_std_by_pixel_*-bands_'+stack_meth+'_'+norm_eline+'_noDC.txt'))[::-1]

    print (colored('The composite error spectra to plot:','green'))
    print (colored('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','green'))

    for fname in uncert_fnames:
        print (fname)
        
    print ()
    print ()

    offset = 2.0
    
else:
    uncert_fnames = None
    offset = 0.
    

if plot_fit == True:
    print (colored('The fit parameters that will be used to plot the fit model:','green'))
    print (colored('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','green'))

    output_path = cwd + '/uncertainty_'+uncert+'_fitting_analysis/' + norm_eline+'_norm/'
    
    fit_params = fr.rc(output_path + 'fw_full_spectrum/two_gaussian_fit_parameters_stacked_spectra_'+stack_meth+'_'+norm_eline+'_no_offset_fw_full_spectrum.fits')

    print ()
    print ()

else:
    output_path = cwd + '/'
    fit_params = None
    

eline_list, eline_rwave, eline_nmsu = np.loadtxt('loi.txt', comments='#', usecols=(0,2,3), dtype='str', unpack=True)
eline_rwave = eline_rwave.astype(float)

####
filter_dict = OrderedDict.fromkeys(['YJ', 'JH', 'HK'])
for filt in filter_dict.keys():
    if filt != 'JH':
        filter_dict[filt] = OrderedDict.fromkeys(['Wavelength', 'Luminosity', 'Lum_Error', 'Fit_Waves', 'Fit'])
    else:
        filter_dict[filt] = OrderedDict.fromkeys(['Wavelength_Blue', 'Wavelength_Red', 'Luminosity_Blue', 'Luminosity_Red', 'Lum_Error_Blue', \
                                                  'Lum_Error_Red', 'Fit_Waves_Blue', 'Fit_Waves_Red', 'Fit_Blue', 'Fit_Red'])
####
    
pp = PdfPages(output_path+pp_name)


for i, fname in enumerate(stacked_fnames):

    print ('Plotting the spectrum in file '+colored(fname,'white')+'...')
    print ()

    stacked_bands = fname[len('stacked_spectrum_') : len('stacked_spectrum_')+2]

    print ('The '+colored(stacked_bands,'green')+' band spectrum will be plotted...')

    if mult_imgs is None:
        rest_waves, luminosities = np.loadtxt(fname, comments='#', usecols=(0,1), dtype='float', unpack=True)
    else:
        rest_waves, luminosities, lum_errs = np.loadtxt(fname, comments='#', usecols=(0,1,2), dtype='float', unpack=True)

    nans_zeros = np.where((np.isnan(luminosities) == True) | (luminosities == 0.))[0]
    rest_waves, luminosities = np.delete(rest_waves, nans_zeros), np.delete(luminosities, nans_zeros)

    if stacked_bands == 'YJ':
        below_bb = np.where(rest_waves < 3660.)[0] #3660
        rest_waves, luminosities = np.delete(rest_waves, below_bb), np.delete(luminosities, below_bb)

    if mult_imgs is not None:
        lum_errs = np.delete(lum_errs, nans_zeros)

        if stacked_bands == 'YJ':
            lum_errs = np.delete(lum_errs, below_bb)

        if len(luminosities) != len(lum_errs):
            raise ValueError('The luminosity array is not the same length as the luminosity error array')
        

    if uncert_fnames is not None:
        uncert_bands = uncert_fnames[i][len(uncert+'_std_by_pixel_') : len(uncert+'_std_by_pixel_')+2]

        print ('The '+colored(uncert_bands,'green')+' band composite error spectrum will be plotted...')

        if stacked_bands != uncert_bands:
            raise ValueError('The bands of the stacked spectrum and composite error spectrum do not match!')

        uncert_waves, lum_errs = np.loadtxt(uncert_fnames[i], comments='#', usecols=(0,1), dtype='float', unpack=True)

        uncert_waves, lum_errs = np.delete(uncert_waves, nans_zeros), np.delete(lum_errs, nans_zeros)

        if stacked_bands == 'YJ':
            uncert_waves, lum_errs = np.delete(uncert_waves, below_bb), np.delete(lum_errs, below_bb)

        equiv_waves = rest_waves == uncert_waves
        
        if np.any(equiv_waves == False):
            raise ValueError('The wavelength values in the stacked spectrum are not the same as the values in the composite error spectrum!')

    if plot_fit:
        print ('The '+colored(stacked_bands,'green')+' fit model spectrum will be plotted...')
        
    print

    kwargs = dict(norm_fact=10.**41, disp_names=True, plt_ylim_bot=-4., plot_fit_model=plot_fit, fit_params=fit_params, stack_meth=stack_meth, \
                  norm_eline=norm_eline, uncert=uncert, bands=stacked_bands, offset=offset)

    
    fit_waves, fit, pp = plot_spectra(rest_waves, luminosities, lum_errs, eline_rwave, eline_list, pp, plt_ylim_top=full_filt_ylim_top, \
                                      save_model_txt=True, leg_loc='upper center', opath=output_path, save_pickle=True, **kwargs)

    if stacked_bands == 'YJ':
        filter_dict['YJ']['Wavelength'] = rest_waves
        filter_dict['YJ']['Luminosity'] = luminosities
        filter_dict['YJ']['Lum_Error']  = lum_errs
        filter_dict['YJ']['Fit_Waves']  = fit_waves
        filter_dict['YJ']['Fit'] = fit
        _,_,pp = plot_spectra(rest_waves, luminosities, lum_errs, eline_rwave, eline_list, pp, plt_ylim_top=o2_ylim_top, **kwargs)
    


    if stacked_bands == 'JH':
        lte_4430 = np.where(rest_waves <= 4430.)[0] #4430
        gte_4800 = np.where(rest_waves >= 4800.)[0] #4800

        
        filter_dict['JH']['Wavelength_Blue'] = rest_waves[lte_4430]
        filter_dict['JH']['Luminosity_Blue'] = luminosities[lte_4430]
        filter_dict['JH']['Lum_Error_Blue']  = lum_errs[lte_4430]
        fit_waves, fit, pp = plot_spectra(rest_waves[lte_4430], luminosities[lte_4430], lum_errs[lte_4430], eline_rwave, eline_list, pp, plt_ylim_top=full_filt_ylim_top, **kwargs)
        _, _, pp = plot_spectra(rest_waves[lte_4430], luminosities[lte_4430], lum_errs[lte_4430], eline_rwave, eline_list, pp, plt_ylim_top=3.6, **kwargs)
        filter_dict['JH']['Fit_Waves_Blue']  = fit_waves
        filter_dict['JH']['Fit_Blue'] = fit
        
        filter_dict['JH']['Wavelength_Red'] = rest_waves[gte_4800]
        filter_dict['JH']['Luminosity_Red'] = luminosities[gte_4800]
        filter_dict['JH']['Lum_Error_Red']  = lum_errs[gte_4800]
        fit_waves, fit, pp = plot_spectra(rest_waves[gte_4800], luminosities[gte_4800], lum_errs[gte_4800], eline_rwave, eline_list, pp, plt_ylim_top=full_filt_ylim_top, **kwargs)
        filter_dict['JH']['Fit_Waves_Red']  = fit_waves
        filter_dict['JH']['Fit_Red'] = fit


    elif stacked_bands == 'HK':
        gte_6450   = np.where(rest_waves >= 6450.)[0] #6450
        heI_wrange = np.where((rest_waves >= 5800.) & (rest_waves <= 5950.))[0]


        filter_dict['HK']['Wavelength'] = rest_waves[gte_6450]
        filter_dict['HK']['Luminosity'] = luminosities[gte_6450]
        filter_dict['HK']['Lum_Error']  = lum_errs[gte_6450]
        fit_waves, fit, pp = plot_spectra(rest_waves[gte_6450], luminosities[gte_6450], lum_errs[gte_6450], eline_rwave, eline_list, pp, plt_ylim_top=full_filt_ylim_top, leg_loc='upper left', **kwargs)
        filter_dict['HK']['Fit_Waves']  = fit_waves
        filter_dict['HK']['Fit'] = fit

        _, _, pp = plot_spectra(rest_waves[heI_wrange], luminosities[heI_wrange], lum_errs[heI_wrange], eline_rwave, eline_list, pp, plt_ylim_top=full_filt_ylim_top, **kwargs)
        _, _, pp = plot_spectra(rest_waves[heI_wrange], luminosities[heI_wrange], lum_errs[heI_wrange], eline_rwave, eline_list, pp, plt_ylim_top=2.6, **kwargs)


pp.close()

print (colored(pp_name,'green')+colored(' written!','red'))
print ()
print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print ()
print ()
print ()

upper_ylim = paper_plot_ylim_top
bbox_props = dict(boxstyle='square', fc='w', ec='w')

yj_med_idx = (np.abs(filter_dict['YJ']['Wavelength'] - 3727.42)).argmin()
jh_blue_med_idx = (np.abs(filter_dict['JH']['Wavelength_Blue'] - 4351.83)).argmin()
jh_red_med_idx = (np.abs(filter_dict['JH']['Wavelength_Red'] - 4910.12)).argmin()
hk_med_idx = (np.abs(filter_dict['HK']['Wavelength'] - 6565.75)).argmin()

o2_range   = np.arange(yj_med_idx-23, yj_med_idx+23, 1)
hgo3_range = np.arange(jh_blue_med_idx-46, jh_blue_med_idx+46, 1)
hbo3_range = np.arange(jh_red_med_idx-115, jh_red_med_idx+115, 1)
n2ha_range = np.arange(hk_med_idx-46, hk_med_idx+46, 1)


yj_min_wave, yj_max_wave = filter_dict['YJ']['Wavelength'][min(o2_range)], filter_dict['YJ']['Wavelength'][max(o2_range)]
hgo3_min_wave, hgo3_max_wave = filter_dict['JH']['Wavelength_Blue'][min(hgo3_range)], filter_dict['JH']['Wavelength_Blue'][max(hgo3_range)]
hbo3_min_wave, hbo3_max_wave = filter_dict['JH']['Wavelength_Red'][min(hbo3_range)], filter_dict['JH']['Wavelength_Red'][max(hbo3_range)]
hk_min_wave, hk_max_wave = filter_dict['HK']['Wavelength'][min(n2ha_range)], filter_dict['HK']['Wavelength'][max(n2ha_range)]


o2_fit   = np.where((filter_dict['YJ']['Fit_Waves'] >= yj_min_wave) & (filter_dict['YJ']['Fit_Waves'] <= yj_max_wave))[0]
hgo3_fit = np.where((filter_dict['JH']['Fit_Waves_Blue'] >= hgo3_min_wave) & (filter_dict['JH']['Fit_Waves_Blue'] <= hgo3_max_wave))[0]
hbo3_fit = np.where((filter_dict['JH']['Fit_Waves_Red'] >= hbo3_min_wave) & (filter_dict['JH']['Fit_Waves_Red'] <= hbo3_max_wave))[0]
n2ha_fit = np.where((filter_dict['HK']['Fit_Waves'] >= hk_min_wave) & (filter_dict['HK']['Fit_Waves'] <= hk_max_wave))[0]


fig = plt.figure()

gs = g.GridSpec(1, 4, width_ratios=[1,2,5,2])
gs.update(bottom=0.257, wspace=0.03)

ax1 = fig.add_subplot(gs[0,0])
ax1.step(filter_dict['YJ']['Wavelength'][o2_range], np.divide(filter_dict['YJ']['Luminosity'][o2_range], 10.**41), where='mid', color='xkcd:sea blue', linewidth=0.7)
ax1.fill_between(filter_dict['YJ']['Wavelength'][o2_range], -offset, np.subtract(np.divide(filter_dict['YJ']['Lum_Error'][o2_range], 10.**41), offset), \
                        step='mid', facecolor='xkcd:gunmetal', linewidth=0.7, edgecolor='xkcd:gunmetal', alpha=0.5 \
                       )
ax1.plot(filter_dict['YJ']['Fit_Waves'][o2_fit], np.divide(filter_dict['YJ']['Fit'][o2_fit], 10.**41), color='red', linewidth=0.7, alpha=0.7)
ax1.axvline(x = 3726.032, color='black', linestyle='--', linewidth=0.5, alpha=0.6)
ax1.axvline(x = 3728.815, color='black', linestyle='--', linewidth=0.5, alpha=0.6)
ax1.axhline(y = 0., color='xkcd:gunmetal', linewidth=0.5)

ax1.minorticks_on()
ax1.set_xticks([3730])
ax1.set_xticks([3720, 3725, 3735], minor=True)
ax1.tick_params(axis='both', which='both', left=True, right=True, bottom=True, top=True)
ax1.set_ylim([-2.2, upper_ylim])
ax1.set_ylabel(r'$L_\lambda$ ($\times10^{41}$) ($erg\ s^{-1}\ \AA^{-1}$)')
ax1.text(3726.032 - 3., 0.8*upper_ylim, '[OII]3726', ha='center', va='bottom', rotation=90, size=7, bbox=bbox_props)  
ax1.text(3728.815 + 3., 0.8*upper_ylim, '[OII]3729', ha='center', va='bottom', rotation=90, size=7, bbox=bbox_props)


ax2 = fig.add_subplot(gs[0,1])
ax2.step(filter_dict['JH']['Wavelength_Blue'][hgo3_range], np.divide(filter_dict['JH']['Luminosity_Blue'][hgo3_range], 10.**41), where='mid', color='xkcd:sea blue', linewidth=0.7)
ax2.fill_between(filter_dict['JH']['Wavelength_Blue'][hgo3_range], -offset, np.subtract(np.divide(filter_dict['JH']['Lum_Error_Blue'][hgo3_range], 10.**41), offset), \
                        step='mid', facecolor='xkcd:gunmetal', linewidth=0.7, edgecolor='xkcd:gunmetal', alpha=0.5 \
                       )
ax2.plot(filter_dict['JH']['Fit_Waves_Blue'][hgo3_fit], np.divide(filter_dict['JH']['Fit_Blue'][hgo3_fit], 10.**41), color='red', linewidth=0.7, alpha=0.7)
ax2.axvline(x = 4340.459, color='black', linestyle='--', linewidth=0.5, alpha=0.6)
ax2.axvline(x = 4363.209, color='black', linestyle='--', linewidth=0.5, alpha=0.6)
ax2.axhline(y = 0., color='xkcd:gunmetal', linewidth=0.5)

ax2.minorticks_on()
ax2.set_xticks([4340, 4360])
ax2.tick_params(axis='both', which='both', left=True, right=True, bottom=True, top=True, labelleft=False)
ax2.set_ylim([-2.2, upper_ylim])
ax2.text(4340.459, 0.8*upper_ylim, r'H$\gamma$', ha='center', va='bottom', rotation=90, size=7, bbox=bbox_props, zorder=100)  
ax2.text(4363.209, 0.8*upper_ylim, '[OIII]4363', ha='center', va='bottom', rotation=90, size=7, bbox=bbox_props, zorder=100)

ax3 = fig.add_subplot(gs[0,2])
ax3.step(filter_dict['JH']['Wavelength_Red'][hbo3_range], np.divide(filter_dict['JH']['Luminosity_Red'][hbo3_range], 10.**41), where='mid', color='xkcd:sea blue', linewidth=0.7, label='Stacked Spectrum')
ax3.fill_between(filter_dict['JH']['Wavelength_Red'][hbo3_range], -offset, np.subtract(np.divide(filter_dict['JH']['Lum_Error_Red'][hbo3_range], 10.**41), offset), \
                        step='mid', facecolor='xkcd:gunmetal', linewidth=0.7, edgecolor='xkcd:gunmetal', alpha=0.5, label='Error Spectrum' \
                       )
ax3.plot(filter_dict['JH']['Fit_Waves_Red'][hbo3_fit], np.divide(filter_dict['JH']['Fit_Red'][hbo3_fit], 10.**41), color='red', linewidth=0.7, alpha=0.7, label='Model Spectrum')
ax3.axvline(x = 4861.321, color='black', linestyle='--', linewidth=0.5, alpha=0.6)
ax3.axvline(x = 4958.910, color='black', linestyle='--', linewidth=0.5, alpha=0.6)
ax3.axhline(y = 0., color='xkcd:gunmetal', linewidth=0.5)

ax3.minorticks_on()
ax3.tick_params(axis='both', which='both', left=True, right=True, bottom=True, top=True, labelleft=False)
ax3.set_ylim([-2.2, upper_ylim])
ax3.text(4861.321, 0.8*upper_ylim, r'H$\beta$', ha='center', va='bottom', rotation=90, size=7, bbox=bbox_props, zorder=100)  
ax3.text(4958.910 - 6., 0.8*upper_ylim, '[OIII]4959', ha='center', va='bottom', rotation=90, size=7, bbox=bbox_props)

handles, labels = ax3.get_legend_handles_labels()
new_order = [1, 0, 2]
handles[:] = [handles[i] for i in new_order]  ## Re-orders the list in-place instead of creating a new variable
labels[:]  = [labels[i] for i in new_order]

ax3.legend(handles, labels, loc='upper center', fontsize='x-small', fancybox=True, frameon=True, framealpha=0.8, edgecolor='black', borderaxespad=1)

ax4 = fig.add_subplot(gs[0,3])
ax4.step(filter_dict['HK']['Wavelength'][n2ha_range], np.divide(filter_dict['HK']['Luminosity'][n2ha_range], 10.**41), where='mid', color='xkcd:sea blue', linewidth=0.7)
ax4.fill_between(filter_dict['HK']['Wavelength'][n2ha_range], -offset, np.subtract(np.divide(filter_dict['HK']['Lum_Error'][n2ha_range], 10.**41), offset), \
                        step='mid', facecolor='xkcd:gunmetal', linewidth=0.7, edgecolor='xkcd:gunmetal', alpha=0.5 \
                       )
ax4.plot(filter_dict['HK']['Fit_Waves'][n2ha_fit], np.divide(filter_dict['HK']['Fit'][n2ha_fit], 10.**41), color='red', linewidth=0.7, alpha=0.7)
ax4.axvline(x = 6548.048, color='black', linestyle='--', linewidth=0.5, alpha=0.6)
ax4.axvline(x = 6562.794, color='black', linestyle='--', linewidth=0.5, alpha=0.6)
ax4.axvline(x = 6583.448, color='black', linestyle='--', linewidth=0.5, alpha=0.6)
ax4.axhline(y = 0., color='xkcd:gunmetal', linewidth=0.5)

ax4.minorticks_on()
ax4.set_xticks([6550, 6570, 6590])
ax4.tick_params(axis='both', which='both', left=True, right=True, bottom=True, top=True, labelleft=False)
ax4.set_ylim([-2.2, upper_ylim])
ax4.text(6548.048, 0.8*upper_ylim, '[NII]6548', ha='center', va='bottom', rotation=90, size=7, bbox=bbox_props, zorder=100)
ax4.text(6562.794 + 7., 0.8*upper_ylim, r'H$\alpha$', ha='center', va='bottom', rotation=90, size=7, bbox=bbox_props)  
ax4.text(6583.448, 0.8*upper_ylim, '[NII]6583', ha='center', va='bottom', rotation=90, size=7, bbox=bbox_props, zorder=100)

plt.annotate(r'Rest-Frame Wavelength ($\AA$)', xy=(0.41,0.16), xytext=(0.41,0.16), xycoords='figure fraction', \
             textcoords='figure fraction')

fig.savefig(output_path + 'Stacked_Spectrum_All_Filters_'+stack_meth+'_'+norm_eline+'_'+uncert+'.pdf')

