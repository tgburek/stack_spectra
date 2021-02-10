#! /usr/bin/env python

import os
import re
import sys
import time
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

"""PLOT THE NEWLY-CREATED COMPOSITE SPECTRA. 
The entire wavelength coverage will be plotted as well as sections around emission lines of interest."""
    
))


parser.add_argument('-m', '--Multiply_Imaged_IDs', metavar='str', \
                    help='The IDs corresponding to the individual spectra of a multiply-imaged object (ex. "123_456_789_...")')

parser.add_argument('-f', '--Plot_Fit', action='store_true', \
                    help='Plot fit model on top of stacked spectrum')

parser.add_argument('Normalizing_ELine',  choices=['OIII5007','H-alpha','no-norm'], \
                    help='The emission line name of the line used to normalize')

parser.add_argument('Stacking_Method', choices=['median','average','weighted-average'], \
                    help='The method with which the spectra were stacked')

parser.add_argument('Uncertainty', choices=['bootstrap', 'statistical'], \
                    help='How the uncertainty spectrum was calculated (including cosmic variance or just statistically)')


args = parser.parse_args()

mult_imgs  = args.Multiply_Imaged_IDs
plot_fit   = args.Plot_Fit
norm_eline = args.Normalizing_ELine
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


def plot_spectra(wavelengths, luminosities, lum_errors, eline_waves, eline_names, disp_names=False, norm_fact=1., plot_fit_model=False, plt_ylim_top=None, \
                 plt_ylim_bot=None, save_model_txt=False, fit_params=None, stack_meth='', norm_eline='', uncert='', offset=None, bands='', leg_loc='best', pp=None):

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

        
    if plot_fit_model == True:

        if fit_params is None or bands == '':
            raise ValueError('Values must be passed to "fit_params" and "bands" keywords if "plot_fit_model=True"')
        
        filt_idx  = int(np.where(fit_params['Filters'] == bands)[0])

        fit_waves = np.linspace(min(wavelengths), max(wavelengths), len(wavelengths)*5, endpoint=True)

        fit_model_hr = linear_cont(fit_waves, fit_params['Slope'][filt_idx], fit_params['y-Int'][filt_idx])
        fit_model_lr = linear_cont(wavelengths, fit_params['Slope'][filt_idx], fit_params['y-Int'][filt_idx])
        
        fit_amps_broad, fit_amps_narrow, fit_amps_single = np.array([]), np.array([]), np.array([])
        two_comp_lines, one_comp_lines = np.array([]), np.array([])

        for name, ewave in itertools.izip(eline_names_to_plot, eline_waves_to_plot):
            if name == 'OIII4959':
                fit_amps_broad  = np.append(fit_amps_broad, fit_params['OIII5007_Broad_Amp'][filt_idx] / 2.98)
                fit_amps_narrow = np.append(fit_amps_narrow, fit_params['OIII5007_Narrow_Amp'][filt_idx] / 2.98)
                two_comp_lines  = np.append(two_comp_lines, ewave)
            elif name == 'NII6548':
                fit_amps_single = np.append(fit_amps_single, fit_params['NII6583_Single_Amp'][filt_idx] / 2.95)
                one_comp_lines  = np.append(one_comp_lines, ewave)
            else:
                if name+'_Broad_Amp' in fit_params.columns.names:
                    fit_amps_broad  = np.append(fit_amps_broad, fit_params[name+'_Broad_Amp'][filt_idx])
                    fit_amps_narrow = np.append(fit_amps_narrow, fit_params[name+'_Narrow_Amp'][filt_idx])
                    two_comp_lines  = np.append(two_comp_lines, ewave)
                else:
                    fit_amps_single = np.append(fit_amps_single, fit_params[name+'_Single_Amp'][filt_idx])
                    one_comp_lines  = np.append(one_comp_lines, ewave)

        reordered_eline_waves = np.append(two_comp_lines, one_comp_lines)

        for j, ewave in enumerate(reordered_eline_waves):

            if j < len(two_comp_lines):
                fit_model_hr += gaussian(fit_waves, ewave, fit_amps_broad[j], fit_params['Sigma_Broad'][filt_idx]) + \
                                gaussian(fit_waves, ewave, fit_amps_narrow[j], fit_params['Sigma_Narrow'][filt_idx]) 
            
                fit_model_lr += gaussian(wavelengths, ewave, fit_amps_broad[j], fit_params['Sigma_Broad'][filt_idx]) + \
                                gaussian(wavelengths, ewave, fit_amps_narrow[j], fit_params['Sigma_Narrow'][filt_idx])

            else:
                fit_model_hr += gaussian(fit_waves, ewave, fit_amps_single[j-len(two_comp_lines)], fit_params['Sigma_Single'][filt_idx])

                fit_model_lr += gaussian(wavelengths, ewave, fit_amps_single[j-len(two_comp_lines)], fit_params['Sigma_Single'][filt_idx])

        if save_model_txt == True:
            G2model = np.array([wavelengths, fit_model_lr]).T
            fname_out = 'multiple_gaussian_fit_model_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'.txt'
            np.savetxt(fname_out, G2model, fmt=['%10.5f','%6.5e'], delimiter='\t', newline='\n', comments='#', \
                       header=fname_out+'\n'+'Rest Wave. (A) | Model Luminosity (erg/s/A)'+'\n' \
                      )
            print '-> '+colored(fname_out, 'green')+' written'
            print

        ax.plot(fit_waves, np.divide(fit_model_hr, norm_fact), color='red', linewidth=0.7, alpha=0.7, label='Model Spectrum')
        #ax.step(wavelengths, np.subtract(np.subtract(luminosities, fit_model_lr), 2.*offset), where='mid', color='xkcd:burnt orange', linewidth=0.7, label='Residuals')
        #ax.axhline(y = -2.*offset, color='xkcd:gunmetal', linewidth=0.5)
        
        handles, labels = ax.get_legend_handles_labels()
        new_order = [1, 0, 2]
        #new_order = [1, 0, 2, 3]  #With residuals plotted

        handles[:] = [handles[i] for i in new_order]  ## Re-orders the list in-place instead of creating a new variable
        labels[:]  = [labels[i] for i in new_order]

    ax.axhline(y = 0., color='xkcd:gunmetal', linewidth=0.5)
        
    if plt_ylim_top is not None:
        ax.set_ylim(top = plt_ylim_top)
    if plt_ylim_bot is not None:
        ax.set_ylim(bottom = plt_ylim_bot)

    for ewave, ename in zip(eline_waves_to_plot, eline_names_to_plot):
        ax.axvline(x = ewave, color='black', linestyle='--', linewidth=0.5, alpha=0.7)

        if disp_names == True:
            ylimits_final = ax.get_ylim()
            bbox_props = dict(boxstyle='square', fc='w', ec='w')
            ax.text(ewave - 5., 0.75*ylimits_final[1], ename, ha='center', va='bottom', rotation=90, size=8, bbox=bbox_props) 

    norm_fact_exp = str('%e' % norm_fact)[str('%e' % norm_fact).find('e')+1:]

    if norm_fact_exp[1] == '0':
        norm_fact_exp = norm_fact_exp.replace('0', '', 1)
    if norm_fact_exp[0] == '+':
        norm_fact_exp = norm_fact_exp.replace('+', '')

    ax.minorticks_on()
    ax.tick_params(which='both', left=True, right=True, bottom=True, top=True)
    ax.legend(handles, labels, loc=leg_loc, fontsize='x-small', fancybox=True, frameon=True, framealpha=0.8, edgecolor='black')  ## If plot_fit_model is False, "handles" and "labels" is currently undefined
    ax.set_xlabel(r'Rest-Frame Wavelength ($\AA$)')
    ax.set_ylabel(r'$L_\lambda$ ($erg\ s^{-1}\ \AA^{-1}$) ($\times10^{'+norm_fact_exp+'}$)')
    ax.set_title('Stacked Spectra via '+stack_meth+' - Norm by '+norm_eline+' - Uncertainty: '+uncert.capitalize())

    plt.tight_layout()
    if pp is not None:
        pp.savefig()
    else:
        plt.show()
    plt.close(fig)

    return fit_waves, fit_model_hr, pp

cwd = os.getcwd()

c = 2.998e5 ## km/s

sys.stdout = Logger(logname=cwd+'/logfiles/plotting_stacked_spectra_'+stack_meth+'_'+norm_eline+'_plot_fit-'+str(plot_fit), mode='w')

print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print colored(('This script will plot the newly-created stacked spectra.\n'
               'The entire wavelength coverage of a spectrum will be plotted\n'
               'as well as sections around emission lines of interest.\n'
               'Models of the spectra can also be overlaid.'
              ), 'cyan',attrs=['bold'])
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print


if mult_imgs is None:
    stacked_fnames = sorted(glob('stacked_spectrum_*-bands_'+stack_meth+'_'+norm_eline+'_noDC.txt'))[::-1]
    pp_name = 'stacked_spectra_by_'+stack_meth+'_'+norm_eline+'_'+uncert+'_two_gaussian_model.pdf'
else:
    stacked_fnames = sorted(glob('stacked_spectrum_*-bands_'+stack_meth+'_'+norm_eline+'_noDC_'+mult_imgs+'.txt'))[::-1]
    pp_name = 'stacked_spectra_by_'+stack_meth+'_'+norm_eline+'_'+mult_imgs+'.pdf'

print colored('The stacked spectra to plot:','green')
print colored('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','green')

for fname in stacked_fnames:
    print fname

print
print
    

if stack_meth == 'average' or stack_meth == 'median':
    uncert_fnames = sorted(glob(uncert+'_std_by_pixel_*-bands_'+stack_meth+'_'+norm_eline+'_noDC.txt'))[::-1]

    print colored('The composite error spectra to plot:','green')
    print colored('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','green')

    for fname in uncert_fnames:
        print fname
        
    print
    print

    offset = 2.0
    
else:
    uncert_fnames = None
    offset = 0.
    

if plot_fit == True:
    print colored('The fit parameters that will be used to plot the fit model:','green')
    print colored('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','green')
    
    fit_params = fr.rc('two_gaussian_fit_parameters_stacked_spectra_'+stack_meth+'_'+norm_eline+'_no_offset_fw_full_spectrum.fits')

    print
    print

else:
    fit_params = None
    

eline_list, eline_rwave = np.loadtxt('loi.txt', comments='#', usecols=(0,2), dtype='str', unpack=True)
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
    
pp = PdfPages(pp_name)


for i, fname in enumerate(stacked_fnames):

    print 'Plotting the spectrum in file '+colored(fname,'white')+'...'
    print

    stacked_bands = fname[len('stacked_spectrum_') : len('stacked_spectrum_')+2]

    print 'The '+colored(stacked_bands,'green')+' band spectrum will be plotted...'

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
            raise Exception('The luminosity array is not the same length as the luminosity error array')
        

    if uncert_fnames is not None:
        uncert_bands = uncert_fnames[i][len(uncert+'_std_by_pixel_') : len(uncert+'_std_by_pixel_')+2]

        print 'The '+colored(uncert_bands,'green')+' band composite error spectrum will be plotted...'

        if stacked_bands != uncert_bands:
            raise ValueError('The bands of the stacked spectrum and composite error spectrum do not match!')

        uncert_waves, lum_errs = np.loadtxt(uncert_fnames[i], comments='#', usecols=(0,1), dtype='float', unpack=True)

        uncert_waves, lum_errs = np.delete(uncert_waves, nans_zeros), np.delete(lum_errs, nans_zeros)

        if stacked_bands == 'YJ':
            uncert_waves, lum_errs = np.delete(uncert_waves, below_bb), np.delete(lum_errs, below_bb)

        equiv_waves = rest_waves == uncert_waves
        
        if np.any(equiv_waves == False):
            raise ValueError('The wavelength values in the stacked spectrum are not the same as the values in the composite error spectrum!')

    if plot_fit == True:
        print 'The '+colored(stacked_bands,'green')+' fit model spectrum will be plotted...'
        
    print

    kwargs = dict(norm_fact=10.**41, plot_fit_model=plot_fit, fit_params=fit_params, stack_meth=stack_meth, \
                  norm_eline=norm_eline, uncert=uncert, offset=offset, bands=stacked_bands, pp=pp
                 )

    
    fit_waves, fit, pp = plot_spectra(rest_waves, luminosities, lum_errs, eline_rwave, eline_list, \
                                      plt_ylim_top=12., save_model_txt=True, leg_loc='upper center', **kwargs)

    if stacked_bands == 'YJ':
        filter_dict['YJ']['Wavelength'] = rest_waves
        filter_dict['YJ']['Luminosity'] = luminosities
        filter_dict['YJ']['Lum_Error']  = lum_errs
        filter_dict['YJ']['Fit_Waves']  = fit_waves
        filter_dict['YJ']['Fit'] = fit
    


    if stacked_bands == 'JH':
        lte_4430 = np.where(rest_waves <= 4430.)[0] #4430
        gte_4800 = np.where(rest_waves >= 4800.)[0] #4800

        
        filter_dict['JH']['Wavelength_Blue'] = rest_waves[lte_4430]
        filter_dict['JH']['Luminosity_Blue'] = luminosities[lte_4430]
        filter_dict['JH']['Lum_Error_Blue']  = lum_errs[lte_4430]
        fit_waves, fit, pp = plot_spectra(rest_waves[lte_4430], luminosities[lte_4430], lum_errs[lte_4430], eline_rwave, eline_list, \
                                          plt_ylim_top=12., disp_names=True, **kwargs)
        _, _, pp = plot_spectra(rest_waves[lte_4430], luminosities[lte_4430], lum_errs[lte_4430], eline_rwave, eline_list, \
                                plt_ylim_top=3., disp_names=True, **kwargs)
        filter_dict['JH']['Fit_Waves_Blue']  = fit_waves
        filter_dict['JH']['Fit_Blue'] = fit
        
        filter_dict['JH']['Wavelength_Red'] = rest_waves[gte_4800]
        filter_dict['JH']['Luminosity_Red'] = luminosities[gte_4800]
        filter_dict['JH']['Lum_Error_Red']  = lum_errs[gte_4800]
        fit_waves, fit, pp = plot_spectra(rest_waves[gte_4800], luminosities[gte_4800], lum_errs[gte_4800], eline_rwave, eline_list, \
                                          plt_ylim_top=12., disp_names=True, **kwargs)
        filter_dict['JH']['Fit_Waves_Red']  = fit_waves
        filter_dict['JH']['Fit_Red'] = fit


    elif stacked_bands == 'HK':
        gte_6450   = np.where(rest_waves >= 6450.)[0] #6450
        heI_wrange = np.where((rest_waves >= 5800.) & (rest_waves <= 5950.))[0]


        filter_dict['HK']['Wavelength'] = rest_waves[gte_6450]
        filter_dict['HK']['Luminosity'] = luminosities[gte_6450]
        filter_dict['HK']['Lum_Error']  = lum_errs[gte_6450]
        fit_waves, fit, pp = plot_spectra(rest_waves[gte_6450], luminosities[gte_6450], lum_errs[gte_6450], eline_rwave, eline_list, \
                                          plt_ylim_top=12., disp_names=True, **kwargs)
        filter_dict['HK']['Fit_Waves']  = fit_waves
        filter_dict['HK']['Fit'] = fit

        _, _, pp = plot_spectra(rest_waves[heI_wrange], luminosities[heI_wrange], lum_errs[heI_wrange], eline_rwave, eline_list, \
                                plt_ylim_top=12., disp_names=True, **kwargs)
        _, _, pp = plot_spectra(rest_waves[heI_wrange], luminosities[heI_wrange], lum_errs[heI_wrange], eline_rwave, eline_list, \
                                plt_ylim_top=2., disp_names=True, **kwargs)


pp.close()

print colored(pp_name,'green')+colored(' written!','red')
print
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print
print


# yj_med_idx = (np.abs(filter_dict['YJ']['Wavelength'] - 3727.42)).argmin()
# jh_blue_med_idx = (np.abs(filter_dict['JH']['Wavelength_Blue'] - 4351.83)).argmin()
# jh_red_med_idx = (np.abs(filter_dict['JH']['Wavelength_Red'] - 4910.12)).argmin()
# hk_med_idx = (np.abs(filter_dict['HK']['Wavelength'] - 6565.75)).argmin()

# o2_range   = np.arange(yj_med_idx-23, yj_med_idx+23, 1)
# hgo3_range = np.arange(jh_blue_med_idx-46, jh_blue_med_idx+46, 1)
# hbo3_range = np.arange(jh_red_med_idx-115, jh_red_med_idx+115, 1)
# n2ha_range = np.arange(hk_med_idx-46, hk_med_idx+46, 1)


# yj_min_wave, yj_max_wave = filter_dict['YJ']['Wavelength'][min(o2_range)], filter_dict['YJ']['Wavelength'][max(o2_range)]
# hgo3_min_wave, hgo3_max_wave = filter_dict['JH']['Wavelength_Blue'][min(hgo3_range)], filter_dict['JH']['Wavelength_Blue'][max(hgo3_range)]
# hbo3_min_wave, hbo3_max_wave = filter_dict['JH']['Wavelength_Red'][min(hbo3_range)], filter_dict['JH']['Wavelength_Red'][max(hbo3_range)]
# hk_min_wave, hk_max_wave = filter_dict['HK']['Wavelength'][min(n2ha_range)], filter_dict['HK']['Wavelength'][max(n2ha_range)]


# o2_fit   = np.where((filter_dict['YJ']['Fit_Waves'] >= yj_min_wave) & (filter_dict['YJ']['Fit_Waves'] <= yj_max_wave))[0]
# hgo3_fit = np.where((filter_dict['JH']['Fit_Waves_Blue'] >= hgo3_min_wave) & (filter_dict['JH']['Fit_Waves_Blue'] <= hgo3_max_wave))[0]
# hbo3_fit = np.where((filter_dict['JH']['Fit_Waves_Red'] >= hbo3_min_wave) & (filter_dict['JH']['Fit_Waves_Red'] <= hbo3_max_wave))[0]
# n2ha_fit = np.where((filter_dict['HK']['Fit_Waves'] >= hk_min_wave) & (filter_dict['HK']['Fit_Waves'] <= hk_max_wave))[0]


# fig = plt.figure()

# gs = g.GridSpec(1, 4, width_ratios=[1,2,5,2])
# gs.update(bottom=0.257, wspace=0.03)

# ax1 = fig.add_subplot(gs[0,0])
# ax1.step(filter_dict['YJ']['Wavelength'][o2_range], np.divide(filter_dict['YJ']['Luminosity'][o2_range], 10.**41), where='mid', color='xkcd:sea blue', linewidth=0.7)
# ax1.fill_between(filter_dict['YJ']['Wavelength'][o2_range], -offset, np.subtract(np.divide(filter_dict['YJ']['Lum_Error'][o2_range], 10.**41), offset), \
#                         step='mid', facecolor='xkcd:gunmetal', linewidth=0.7, edgecolor='xkcd:gunmetal', alpha=0.5 \
#                        )
# ax1.plot(filter_dict['YJ']['Fit_Waves'][o2_fit], np.divide(filter_dict['YJ']['Fit'][o2_fit], 10.**41), color='red', linewidth=0.7, alpha=0.7)
# ax1.axvline(x = 3726.032, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
# ax1.axvline(x = 3728.815, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
# ax1.axhline(y = 0., color='xkcd:gunmetal', linewidth=0.5)

# ax1.minorticks_on()
# #ax1.set_xticks([3725])
# ax1.tick_params(axis='both', which='both', left=True, right=True, bottom=True, top=True)
# ax1.set_ylim([-2.2, 10.])
# ax1.set_ylabel(r'$L_\lambda$ ($\times10^{41}$) ($erg\ s^{-1}\ \AA^{-1}$)')


# ax2 = fig.add_subplot(gs[0,1])
# ax2.step(filter_dict['JH']['Wavelength_Blue'][hgo3_range], np.divide(filter_dict['JH']['Luminosity_Blue'][hgo3_range], 10.**41), where='mid', color='xkcd:sea blue', linewidth=0.7)
# ax2.fill_between(filter_dict['JH']['Wavelength_Blue'][hgo3_range], -offset, np.subtract(np.divide(filter_dict['JH']['Lum_Error_Blue'][hgo3_range], 10.**41), offset), \
#                         step='mid', facecolor='xkcd:gunmetal', linewidth=0.7, edgecolor='xkcd:gunmetal', alpha=0.5 \
#                        )
# ax2.plot(filter_dict['JH']['Fit_Waves_Blue'][hgo3_fit], np.divide(filter_dict['JH']['Fit_Blue'][hgo3_fit], 10.**41), color='red', linewidth=0.7, alpha=0.7)
# ax2.axvline(x = 4340.459, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
# ax2.axvline(x = 4363.209, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
# ax2.axhline(y = 0., color='xkcd:gunmetal', linewidth=0.5)

# ax2.minorticks_on()
# #ax2.set_xticks([4340, 4360])
# ax2.tick_params(axis='both', which='both', left=True, right=True, bottom=True, top=True, labelleft=False)
# ax2.set_ylim([-2.2, 10])

# ax3 = fig.add_subplot(gs[0,2])
# ax3.step(filter_dict['JH']['Wavelength_Red'][hbo3_range], np.divide(filter_dict['JH']['Luminosity_Red'][hbo3_range], 10.**41), where='mid', color='xkcd:sea blue', linewidth=0.7, label='Stacked Spectrum')
# ax3.fill_between(filter_dict['JH']['Wavelength_Red'][hbo3_range], -offset, np.subtract(np.divide(filter_dict['JH']['Lum_Error_Red'][hbo3_range], 10.**41), offset), \
#                         step='mid', facecolor='xkcd:gunmetal', linewidth=0.7, edgecolor='xkcd:gunmetal', alpha=0.5, label='Error Spectrum' \
#                        )
# ax3.plot(filter_dict['JH']['Fit_Waves_Red'][hbo3_fit], np.divide(filter_dict['JH']['Fit_Red'][hbo3_fit], 10.**41), color='red', linewidth=0.7, alpha=0.7, label='Model Spectrum')
# ax3.axvline(x = 4861.321, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
# ax3.axvline(x = 4958.910, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
# ax3.axhline(y = 0., color='xkcd:gunmetal', linewidth=0.5)

# ax3.minorticks_on()
# ax3.tick_params(axis='both', which='both', left=True, right=True, bottom=True, top=True, labelleft=False)
# ax3.set_ylim([-2.2, 10])

# handles, labels = ax3.get_legend_handles_labels()
# new_order = [1, 0, 2]
# handles[:] = [handles[i] for i in new_order]  ## Re-orders the list in-place instead of creating a new variable
# labels[:]  = [labels[i] for i in new_order]

# ax3.legend(handles, labels, loc='upper center', fontsize='x-small', fancybox=True, frameon=True, framealpha=0.8, edgecolor='black', borderaxespad=1)

# ax4 = fig.add_subplot(gs[0,3])
# ax4.step(filter_dict['HK']['Wavelength'][n2ha_range], np.divide(filter_dict['HK']['Luminosity'][n2ha_range], 10.**41), where='mid', color='xkcd:sea blue', linewidth=0.7)
# ax4.fill_between(filter_dict['HK']['Wavelength'][n2ha_range], -offset, np.subtract(np.divide(filter_dict['HK']['Lum_Error'][n2ha_range], 10.**41), offset), \
#                         step='mid', facecolor='xkcd:gunmetal', linewidth=0.7, edgecolor='xkcd:gunmetal', alpha=0.5 \
#                        )
# ax4.plot(filter_dict['HK']['Fit_Waves'][n2ha_fit], np.divide(filter_dict['HK']['Fit'][n2ha_fit], 10.**41), color='red', linewidth=0.7, alpha=0.7)
# ax4.axvline(x = 6548.048, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
# ax4.axvline(x = 6562.794, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
# ax4.axvline(x = 6583.448, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
# ax4.axhline(y = 0., color='xkcd:gunmetal', linewidth=0.5)

# ax4.minorticks_on()
# #ax4.set_xticks([6550, 6570])
# ax4.tick_params(axis='both', which='both', left=True, right=True, bottom=True, top=True, labelleft=False)
# ax4.set_ylim([-2.2, 10])

# plt.annotate(r'Rest Wavelength ($\rm \AA$)', xy=(0.41,0.16), xytext=(0.41,0.16), xycoords='figure fraction', \
#              textcoords='figure fraction')

# fig.savefig('Stacked_Spectrum_All_Filters.pdf')

# plt.show()

