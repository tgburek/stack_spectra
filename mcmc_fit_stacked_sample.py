#!/usr/bin/env python

import os
import re
import sys
import time
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
import plotting_functions as pf
import seaborn as sns
import sns_setstyle
import fits_readin as fr
from corner import corner
from astropy.io import fits
from collections import OrderedDict
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.interpolate import interp1d
from termcolor import colored
from matplotlib.backends.backend_pdf import PdfPages
from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentDefaultsHelpFormatter

print

class HelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    pass

parser = ArgumentParser(formatter_class=HelpFormatter, \
                        description = 'Markov Chain Monte Carlo (MCMC) spectral fitting code for composite spectrum of stacked galaxy sample')

parser.add_argument('-c', '--Plot_Corner', action='store_true', \
                    help='Create corner plot from output of MCMC routine')

parser.add_argument('-q', '--Quicklook_Fit', action='store_true', \
                    help='Overplot best-fit model on stacked spectrum immediately following MCMC routine\n'
                         'This precedes generation of marginalized posterior and confidence interval plots\n'
                         '(Useful for testing fitting code at an intermediate step)')
                    
parser.add_argument('-w', '--Walkers', metavar='int', type=int, default=300, \
                    help='The number of walkers to be used in the MCMC routine')

parser.add_argument('-b', '--Burnin', metavar='int', type=int, default=1500, \
                    help='The number of burn-in iterations to be done in the MCMC routine')

parser.add_argument('-i', '--Iterations', metavar='int', type=int, default=25000, \
                    help='The number of post-burn-in iterations to be done in the MCMC routine')

parser.add_argument('-s', '--Sigma_Broad', metavar='float', type=float, \
                    help='A fixed broad (1-sigma) width (km/s) to use for the broad component\n'
                         'when fitting a narrow component width and N/B amplitude ratio')

parser.add_argument('-p', '--Parameter_File', metavar='str', \
                    help='The FITS file with the ALREADY-fit "narrow" and "broad" component line widths (km/s)\n'
                         'and amplitude ratios (N/B) to use and fix here')

parser.add_argument('Normalizing_Eline', choices=['OIII5007', 'H-alpha'], \
                    help='The emission-line name of the line used to normalize the spectra during stacking')

parser.add_argument('Stacking_Method', choices=['median', 'average'], \
                    help='The method with which the spectra in the sample were stacked')

parser.add_argument('Uncertainty', choices=['bootstrap', 'statistical'], \
                    help='How the uncertainty spectrum was calculated\n'
                         '(including cosmic variance or just statistically)')

args = parser.parse_args()

plt_corner  = args.Plot_Corner
quick_fit   = args.Quicklook_Fit
nwalkers    = args.Walkers
burn        = args.Burnin
iterations  = args.Iterations
prov_sbroad = args.Sigma_Broad
param_file  = args.Parameter_File
norm_eline  = args.Normalizing_Eline
stack_meth  = args.Stacking_Method
uncert      = args.Uncertainty

if param_file is None: ## I am fitting the widths in this version of the run
    fixed_widths = False
    run_descr = 'fitting_widths'
else:
    fixed_widths = True
    run_descr = 'fw_full_spectrum'
    

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

class prior_bounds:
    def __init__(self, broad_sigma, narrow_sigma, amplitude_ratio, narrow_amplitudes):
        self.bsig   = broad_sigma
        self.nsig   = narrow_sigma
        self.aratio = amplitude_ratio
        self.namps  = narrow_amplitudes


def linear_cont(x, slope, yint):
    return slope * (x - min(x)) + yint
        
def model_gaussian(x, mean, amplitude, width):
    return amplitude * np.exp(-0.5*(np.power(x - mean, 2) / np.power(width*mean/c, 2)))


def gaussian(x, mu, sigma):
    return (1./(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*(np.power(x - mu, 2) / np.power(sigma, 2)))


def make_model(emiss_waves, width_broad=None, width_narrow=None, amplitude_ratio=None, amp_begin=None, two_gauss=False):
    def gauss_model(wavelengths, *pars):
        
        m = pars[0]
        b = pars[1]
        
        if two_gauss:

            if width_broad is not None:
                bsig = width_broad
            else:
                bsig = pars[2]


            if width_narrow is not None:
                nsig = width_narrow
            elif width_narrow is None and width_broad is not None:
                nsig = pars[2]
            else:
                nsig = pars[3]


            if amplitude_ratio is not None:
                amp_ratio = amplitude_ratio
            else:
                amp_ratio = pars[amp_begin-1]

            
            amps = np.array(pars[amp_begin:])             


        else:
            sig  = pars[2]
            amps = pars[3:]

            
        if len(amps) != len(emiss_waves):
            raise ValueError('The number of amplitude free-parameters should equal the number of emission lines being fit')
        
        model = linear_cont(wavelengths, m, b)

        if two_gauss:
            for j, ewave in enumerate(emiss_waves):
                model += model_gaussian(wavelengths, ewave, amps[j] / amp_ratio, bsig) + model_gaussian(wavelengths, ewave, amps[j], nsig)

        else:
            for j, ewave in enumerate(emiss_waves):
                model += model_gaussian(wavelengths, ewave, amps[j], sig)

        return model
    
    return gauss_model
        
    


##Establishing priors in order to prevent runaway walkers
def lnprior(pos, pbounds, eline, amp_begin, fixed_widths, broad_width, nar_width, amplitude_ratio):

    ## pos is a 1D array of length ndim and type float64
  
    m = pos[0]
    b = pos[1]

    if broad_width is not None:
        bsig = broad_width
    else:
        bsig = pos[2]


    if nar_width is not None:
        nsig = nar_width
    elif nar_width is None and broad_width is not None:
        nsig = pos[2]
    else:
        nsig = pos[3]


    if amplitude_ratio is not None:
        ratio = amplitude_ratio
    else:
        ratio = pos[amp_begin-1]


    namps = pos[amp_begin:]


    if np.any(eline == '[OIII-]') and np.any(eline == '[OIII+]'):
        midx  = int(np.where(eline == '[OIII-]')[0])
        namps = np.insert(namps, midx, namps[-1]/o3_ratio)
        
    elif np.any(eline == '[NII-]') and np.any(eline == '[NII+]'):
        midx  = int(np.where(eline == '[NII-]')[0])
        namps = np.insert(namps, midx, namps[-1]/n2_ratio)

    within_priors = False
    
    if fixed_widths == False:
        if pbounds.nsig[0] <= nsig < pbounds.nsig[1] and \
           pbounds.aratio[0] < ratio <= pbounds.aratio[1] and \
           np.all(namps >= pbounds.namps[0]) and \
           np.all(namps <= pbounds.namps[1]) and \
           (broad_width is not None or (pbounds.bsig[0] <= bsig <= pbounds.bsig[1])):
           
            within_priors = True
       
    elif fixed_widths == True and nar_width is None:
        if pbounds.nsig[0] <= nsig < pbounds.nsig[1] and \
           np.all(namps >= pbounds.namps[0]) and \
           np.all(namps <= pbounds.namps[1]):

            within_priors = True

    else:
        if np.all(namps >= pbounds.namps[0]) and \
           np.all(namps <= pbounds.namps[1]):

            within_priors = True

    if within_priors:
        return m, b, bsig, nsig, ratio, namps, 0.0

    return m, b, bsig, nsig, ratio, namps, -np.inf



##Establishing the likelihood and product of likelihood and priors.  "pos" is again the position of a walker
def lnprob(pos, wavelengths, luminosities, lum_errors, pbounds, eline, rest_wave, amp_begin, fixed_widths, broad_width, nar_width, amplitude_ratio):

    ##pos is the initial position of a walker in the parameter space -> will be an array of values
    ##Call the prior function

    m, b, bsig, nsig, ratio, namps, lnp = lnprior(pos, pbounds, eline, amp_begin, fixed_widths, broad_width, nar_width, amplitude_ratio)        

    if np.isinf(lnp):
        return lnp
    
    ##The continuum part (linear) of the flux calculation.  The wavelength range is shifted by the minimum of the range in order to establish an intercept at the minimum value
    model = linear_cont(wavelengths, m, b)

    ##The rest of the flux calculation (Guassian --> Following Class Activity 6)
    for i, ewave in enumerate(rest_wave):
        model += model_gaussian(wavelengths, ewave, namps[i], nsig) + model_gaussian(wavelengths, ewave, namps[i] / ratio, bsig)
        
    ##Calculating the posterior at each wavelength
    post = -np.log(np.sqrt(2.*np.pi)*lum_errors) + (-0.5*(np.power(model - luminosities, 2) / np.power(lum_errors, 2)))

    ##Adding all the individual posterior measurements together (They are in log space and are independent of each other)
    tot_prob = np.sum(post)

    ##Return the posterior value (up to a constant)
    return tot_prob


def plot_model(wavelengths, luminosities, luminosity_errors, broad_component, narrow_component, total_model, ewaves, \
               which='', bands='', stack_meth='', norm_eline='', run_description='', opath='', pp=None):

    fig, ax = plt.subplots()

    ax.plot(wavelengths, luminosities, linewidth=0.5, color='black', label='Stacked Spectrum')
    ax.plot(wavelengths, luminosity_errors, linewidth=0.5, color='xkcd:purplish grey', alpha=0.7, label='Error Spect')
    ax.plot(wavelengths, broad_component, linewidth=0.5, color='green', label='Model Broad Comp')
    ax.plot(wavelengths, narrow_component, linewidth=0.5, color='blue', label='Model Narrow Comp')
    ax.plot(wavelengths, total_model, linewidth=0.7, color='red', alpha=0.7, label='Total Model', zorder=100)
    for line in ewaves:
        ax.axvline(x = line, linestyle='--', linewidth=0.5, color='black', alpha=0.6)

    ax.minorticks_on()
    ax.tick_params(which='both', left=True, right=True, bottom=True, top=True)
    
    if which == 'Initial':
        ax.set_title('Initial Guess Model --- Filters: '+bands+'   Stacked via: '+stack_meth+'   Norm by: '+norm_eline)
        pickle_fname = opath + 'Initial_Fit_Guess_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_description+'.fig.pickle'
        
    elif which == 'Final':
        ax.set_title('Fit Model --- Filters: '+bands+'   Stacked via: '+stack_meth+'   Norm by: '+norm_eline)
        pickle_fname = opath + 'Fit_Model_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_description+'.fig.pickle'
        
    else:
        ax.set_title('Stacked Spectrum with Spectral Model and Model Components')
        pickle_fname = opath + 'Stack_w_overplotted_model_comps_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_description+'_'+which+'.fig.pickle'
        
    ax.set_xlabel(r'Rest-Frame Wavelength ($\AA$)')
    ax.set_ylabel(r'$\rm L_\lambda$ ($erg\ s^{-1}\ \AA^{-1}$)')
    ax.legend(loc='upper left', fontsize='small', fancybox=True, frameon=True, framealpha=0.8, edgecolor='black')
    plt.tight_layout()
    pickle.dump(fig, open(pickle_fname, 'wb'))
    if pp is not None:
        pp.savefig()
    else:
        plt.show()
    plt.close(fig)

    return


def plot_hist_icof(param_chain, binsize, color='xkcd:grey', xlabel='Parameter', ylabel='Number of Instances', title='', filename='hist.pdf'):
    ## icof = in case of failure - for when a fit runs up against its prior bounds - the usual reason for failure of this script
    fig, ax = plt.subplots()
    ax.hist(param_chain, bins=np.arange(min(param_chain), max(param_chain)+binsize, binsize), color=color, alpha=0.5, edgecolor='grey')
    ax.minorticks_on()
    ax.tick_params(which='both', left=True, right=True, bottom=True, top=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename) 
    plt.close(fig)

    print colored(filename, 'yellow')+' created'
    print

    return


cwd = os.getcwd()

terminal_only = sys.stdout
logname_base  = cwd+'/logfiles/fitting_spectra_'+norm_eline+'_'+stack_meth+'_'+uncert+'_no_offset_'+run_descr
sys.stdout    = Logger(logname=logname_base, mode='w')

print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print '- '+colored(('This program is a Markov Chain Monte Carlo (MCMC) spectral fitting code\n'
                    'for fitting the composite spectrum of a stacked galaxy sample.\n'
                    'THIS CODE IS IN DEVELOPMENT.'
                   ), 'cyan',attrs=['bold']),
print ' -'
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print

##################################### Global fixed variables
c = 2.998e5  ## km/s

o3_ratio = 2.98
n2_ratio = 2.95

width = 50

std_div = 10.

scale_fact = 1.0e40

if fixed_widths == False:
    #comp_bands = ['YJ']
    comp_bands = ['JH', 'HK']
else:
    comp_bands = ['YJ', 'JH', 'HK']

pbounds = prior_bounds(broad_sigma=(80., 150.), narrow_sigma=(30., 80.), \
                       amplitude_ratio=(1.0, 15.), narrow_amplitudes=(0., 1.5e42))
####################################

print 'Review of options called and arguments given to this script:'
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print 'Options:'
print '-> Create corner plot: ', colored(plt_corner, 'cyan')
print '-> Create "quicklook" fit plot: ', colored(quick_fit, 'cyan')
print '-> MCMC Parameters:'
print '---> Number of walkers: ', colored(nwalkers, 'cyan')
print '---> Burn-in iterations: ', colored(burn, 'cyan')
print '---> Iterations following burn-in: ', colored(iterations, 'cyan')
print '-> Fixed broad component (1-sigma) width provided with the "Sigma_Broad" option (km/s): ', colored(prov_sbroad, 'cyan')
print '-> Parameter file containing already-fit "broad" and "narrow" line widths to be used here: ', colored(param_file, 'cyan')
print
print 'Arguments:'
print '-> Spectra normalized by: ', colored(norm_eline, 'cyan')
print '-> Stacking method used: ', colored(stack_meth, 'cyan')
print '-> Uncertainty calculation method: ', colored(uncert, 'cyan')
print
print
print

print 'The current working directory and path are: '+colored(cwd, 'cyan')
print

parent_out_path = cwd + '/' + 'uncertainty_'+uncert+'_fitting_analysis/'
child_out_path  = parent_out_path + norm_eline + '_norm/'
output_path     = child_out_path  + run_descr  + '/'

if os.path.isdir(parent_out_path) == False:
    os.mkdir(parent_out_path)
    print 'Created directory: '+colored(parent_out_path, 'white')

if os.path.isdir(child_out_path) == False:
    os.mkdir(child_out_path)
    print 'Created directory: '+colored(child_out_path, 'white')

if os.path.isdir(output_path) == False:
    os.mkdir(output_path)
    os.mkdir(output_path + 'param_hists')
    os.mkdir(output_path + 'param_hists/two_gaussian_fits')
    print 'Created directories: '
    print '- ' + colored(output_path,  'white')
    print '- ' + colored(output_path + 'param_hists', 'white')
    print '- ' + colored(output_path + 'param_hists/two_gaussian_fits', 'white')


elines_restframe = pd.read_csv('loi.txt', comment='#', delim_whitespace=True, names=['Eline','Eline_SH','Rest_Lambda'], index_col='Eline', \
                               dtype={'Eline': np.string_, 'Eline_SH': np.string_, 'Rest_Lambda': np.float64}, usecols=[0, 1, 2] \
                              )[['Eline_SH', 'Rest_Lambda']]  ## SH = shorthand
print
print    
print '- '+colored('Select emission lines to fit if within spectroscopic coverage', 'magenta', attrs=['bold'])+' -'
print
print elines_restframe
print
print elines_restframe.dtypes
print
print
print

if param_file is not None:
    params = fr.rc(child_out_path + 'fitting_widths/' + param_file)

    
pp  = PdfPages(output_path + 'Two_Gaussian_Initial_Fit_Guess_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_descr+'.pdf')
pp3 = PdfPages(output_path + 'Two_Gaussian_Spectral_Fits_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_descr+'.pdf')

fit_dict_keys = ['Filters', 'Stacking_Method', 'Norm_Eline', 'Uncertainty', 'Slope', 'Slope_sig', 'y-Int', 'y-Int_sig', 'Sigma_Broad', 'Sigma_Narrow', 'Amplitude_Ratio']

fit_dict = OrderedDict.fromkeys(fit_dict_keys)

for key in fit_dict_keys:
    if key == 'Filters' or key == 'Stacking_Method' or key == 'Norm_Eline' or key == 'Uncertainty':
        fit_dict[key] = np.array([])
    else:
        fit_dict[key] = np.zeros(len(comp_bands))


for count, bands in enumerate(comp_bands):

    fname = 'stacked_spectrum_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_noDC.txt'
    uncertainty_fname = uncert+'_std_by_pixel_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_noDC.txt'

    print 'Reading in the spectrum wavelength and luminosity table: '+colored(fname, 'white')
    print 'Reading in composite error spectrum table: '+colored(uncertainty_fname, 'white')
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print
    print

    wavelengths, luminosities = np.loadtxt(fname, comments='#', usecols=(0,1), dtype='float', unpack=True)
    boot_waves, lum_errors    = np.loadtxt(uncertainty_fname, comments='#', usecols=(0,1), dtype='float', unpack=True)

    nans_zeros = np.where((np.isnan(luminosities) == True) | (luminosities == 0.))[0]
    wavelengths, boot_waves  = np.delete(wavelengths, nans_zeros), np.delete(boot_waves, nans_zeros)
    luminosities, lum_errors = np.delete(luminosities, nans_zeros), np.delete(lum_errors, nans_zeros)

    if fixed_widths == False:
        if bands == 'YJ':
            no_fit = np.where(wavelengths < 3660.)[0]
        elif bands == 'JH':
            # no_fit = np.array([])
            no_fit = np.where(wavelengths < 4800.)[0]
        elif bands == 'HK':
            # no_fit = np.array([])
            # no_fit = np.where(wavelengths < 6450.)[0]
            no_fit = np.where((wavelengths <= 6551.) | (wavelengths >= 6580.))[0]
    else:
        if bands == 'YJ':
            no_fit = np.where(wavelengths < 3660.)[0]  ## To cut out the portion of the spectrum with the Balmer break (~3646 A)
        else:
            no_fit = np.array([])

    wavelengths, boot_waves  = np.delete(wavelengths, no_fit), np.delete(boot_waves, no_fit)
    luminosities, lum_errors = np.delete(luminosities, no_fit), np.delete(lum_errors, no_fit) 

    if len(wavelengths) != len(boot_waves) or len(luminosities) != len(lum_errors):
        raise ValueError('The wavelength and/or luminosity arrays of the two spectral tables read in are different lengths!')

    equiv_waves = wavelengths == boot_waves

    if equiv_waves.any() == False:
        raise ValueError('The wavelength values in the stacked spectrum are not the same as the values in the composite error spectrum!')

    del boot_waves


    elines_restframe['Within_SCov'] = elines_restframe['Rest_Lambda'].between(min(wavelengths), max(wavelengths), inclusive=True)

    fit_lines = elines_restframe[elines_restframe['Within_SCov']].index.to_numpy()  ## Returns index labels where "Within_SCov" column is "True"
    eline     = elines_restframe.loc[fit_lines, 'Eline_SH'].to_numpy()
    rest_wave = elines_restframe.loc[fit_lines, 'Rest_Lambda'].to_numpy()


    print '- '+colored('PRIOR to any removals:', 'magenta', attrs=['bold'])+' -'
    print 'Emission lines to fit: '+colored(fit_lines, 'green')
    print 'In shorthand: '+colored(eline, 'green')
    print 'Rest wavelengths of these lines: '+colored(rest_wave, 'green')
    print

    #Guessing initial conditions
    m = 0.
    b = 0.
    amp_nar = np.array([])

    if fixed_widths == False:

        if prov_sbroad is not None:
            bwidth = prov_sbroad
            amp_begin = 4
        else:
            bwidth = 100.  #75
            amp_begin = 5

        nwidth = 45.
        amp_ratio = 5. 

    else:
        if bands != 'YJ':
            bwidth = params['Sigma_Broad'][params['Filters'] == bands][0]
            nwidth = params['Sigma_Narrow'][params['Filters'] == bands][0]
            amp_ratio = params['Amplitude_Ratio'][params['Filters'] == bands][0]
            amp_begin = 2
        else:
            bwidth = params['Sigma_Broad'][params['Filters'] == 'JH'][0]
            nwidth = 45.
            amp_ratio = params['Amplitude_Ratio'][params['Filters'] == 'JH'][0]
            amp_begin = 3

    for lambda_ in rest_wave:
        search_ind = np.where((wavelengths >= lambda_ - 2.) & (wavelengths <= lambda_ + 2.))[0]
        amp_nar = np.append(amp_nar, 0.8 * max(luminosities[search_ind]) / scale_fact)

    luminosities_scaled, lum_errors_scaled = np.divide(luminosities, scale_fact), np.divide(lum_errors, scale_fact)


    if fixed_widths == False:

        if prov_sbroad is not None:
            popt, pcov = curve_fit(make_model(rest_wave, width_broad=bwidth, amp_begin=amp_begin, two_gauss=True), \
                                   wavelengths, luminosities_scaled, sigma=lum_errors_scaled, \
                                   p0=[m, b, nwidth, amp_ratio] + list(amp_nar), \
                                   bounds=([-np.inf, -np.inf, pbounds.nsig[0], pbounds.aratio[0]] + list(np.full(len(amp_nar), pbounds.namps[0])), \
                                           [np.inf, np.inf, pbounds.nsig[1], pbounds.aratio[1]] + list(np.full(len(amp_nar), pbounds.namps[1]/scale_fact))))
        else:
            popt, pcov = curve_fit(make_model(rest_wave, amp_begin=amp_begin, two_gauss=True), \
                                   wavelengths, luminosities_scaled, sigma=lum_errors_scaled, \
                                   p0=[m, b, bwidth, nwidth, amp_ratio] + list(amp_nar), \
                                   bounds=([-np.inf, -np.inf, pbounds.bsig[0], pbounds.nsig[0], pbounds.aratio[0]] + list(np.full(len(amp_nar), pbounds.namps[0])), \
                                           [np.inf, np.inf, pbounds.bsig[1]-10., pbounds.nsig[1], pbounds.aratio[1]] + list(np.full(len(amp_nar), pbounds.namps[1]/scale_fact))))

    else:
        if bands == 'YJ':
            popt, pcov = curve_fit(make_model(rest_wave, width_broad=bwidth, amplitude_ratio=amp_ratio, amp_begin=amp_begin, two_gauss=True), \
                                   wavelengths, luminosities_scaled, sigma=lum_errors_scaled, \
                                   p0=[m, b, nwidth] + list(amp_nar), \
                                   bounds=([-np.inf, -np.inf, pbounds.nsig[0]] + list(np.full(len(amp_nar), pbounds.namps[0])), \
                                           [np.inf, np.inf, pbounds.nsig[1]] + list(np.full(len(amp_nar), pbounds.namps[1]/scale_fact))))
        else:
            popt, pcov = curve_fit(make_model(rest_wave, width_broad=bwidth, width_narrow=nwidth, amplitude_ratio=amp_ratio, amp_begin=amp_begin, two_gauss=True), \
                                   wavelengths, luminosities_scaled, sigma=lum_errors_scaled, \
                                   p0=[m, b] + list(amp_nar), \
                                   bounds=([-np.inf, -np.inf] + list(np.full(len(amp_nar), pbounds.namps[0])), \
                                           [np.inf, np.inf] + list(np.full(len(amp_nar), pbounds.namps[1]/scale_fact))))

    psigs = np.sqrt(np.diag(pcov))

    m, m_isig = popt[0]*scale_fact, psigs[0]*scale_fact
    b, b_isig = popt[1]*scale_fact, psigs[1]*scale_fact
    namps, namps_isig = popt[amp_begin:]*scale_fact, psigs[amp_begin:]*scale_fact

    if fixed_widths == False:
        if prov_sbroad is None:
            bwidth, bwidth_isig = popt[2], psigs[2]
            
        nwidth, nwidth_isig = popt[amp_begin-2], psigs[amp_begin-2]
        amp_ratio, amp_ratio_isig = popt[amp_begin-1], psigs[amp_begin-1]

    elif fixed_widths == True and bands == 'YJ':
        nwidth, nwidth_isig = popt[2], psigs[2]

    bamps = np.divide(namps, amp_ratio)

    print
    print '- '+colored('Initial fit guesses from non-linear least squares curve fitting:', 'magenta', attrs=['bold'])+' -'
    print

    snf  = '{: >11.4e}'.format
    ff   = '{: >11.4f}'.format
    istf = '{: >30}'.format

    print istf('Slope = '),       colored( snf(m), 'green'), ' +/-', colored( snf(m_isig), 'green')
    print istf('y-Intercept = '), colored( snf(b), 'green'), ' +/-', colored( snf(b_isig), 'green')
    print

    if fixed_widths == False:
        if prov_sbroad is None:
            print istf('Width - Broad (km/s) = '),  colored( ff(bwidth), 'green'), ' +/-', colored( ff(bwidth_isig), 'green')
        else:
            print istf('Width - Broad (km/s) = '),  colored( ff(bwidth), 'green')
        print istf('Width - Narrow (km/s) = '), colored( ff(nwidth), 'green'), ' +/-', colored( ff(nwidth_isig), 'green')
        print
        print istf('Amplitude Ratio = '), colored( ff(amp_ratio), 'green'), ' +/-', colored( ff(amp_ratio_isig), 'green')

    else:
        print istf('Width - Broad (km/s) = '),  colored( ff(bwidth), 'green')
        if bands == 'YJ':
            print istf('Width - Narrow (km/s) = '), colored( ff(nwidth), 'green'), ' +/-', colored( ff(nwidth_isig), 'green')
        else:
            print istf('Width - Narrow (km/s) = '), colored( ff(nwidth), 'green')
        print
        print istf('Amplitude Ratio = '), colored( ff(amp_ratio), 'green')

    print

    for j, ename in enumerate(fit_lines):
        print istf(ename + ' Amplitude - Broad = '),  colored( snf(bamps[j]), 'green')            
        print istf(ename + ' Amplitude - Narrow = '), colored( snf(namps[j]), 'green'), ' +/-', colored( snf(namps_isig[j]), 'green')
        print

    print

    # with np.printoptions(formatter={'float': '{: .6f}'.format}):
    #     print popt
    # print
    # with np.printoptions(formatter={'float': '{: .6f}'.format}):
    #     print np.sqrt(np.diag(pcov))
    # print

    print '-> Plotting the initial guess model on top of the stacked spectrum'

    G1model = make_model(rest_wave, two_gauss=False)

    if fixed_widths == False:
        G2model = make_model(rest_wave, width_broad=prov_sbroad, amp_begin=amp_begin, two_gauss=True)

    else:
        if bands == 'YJ':
            G2model = make_model(rest_wave, width_broad=bwidth, amplitude_ratio=amp_ratio, amp_begin=amp_begin, two_gauss=True)
        else:
            G2model = make_model(rest_wave, width_broad=bwidth, width_narrow=nwidth, amplitude_ratio=amp_ratio, amp_begin=amp_begin, two_gauss=True)
        

    broad_comp  = G1model(wavelengths, m, b, bwidth, *bamps)
    narrow_comp = G1model(wavelengths, m, b, nwidth, *namps)
    total_model = G2model(wavelengths, *popt) * scale_fact

    model_kwargs = dict(bands=bands, stack_meth=stack_meth, norm_eline=norm_eline, run_description=run_descr, opath=output_path)

    plot_model(wavelengths, luminosities, lum_errors, broad_comp, narrow_comp, total_model, rest_wave, which='Initial', pp=pp, **model_kwargs)

    print '-> Plot saved to PDF and binary file for later interactive use'
    print
    print


    if np.any(eline == '[OIII-]') and np.any(eline == '[OIII+]'):
        print '-> Removing [OIII]4959 from directly being fit'
        print '-> This line will still help constrain [OIII]5007 via their intrinsic flux ratio'

        midx = int(np.where(eline == '[OIII-]')[0])
        namps, namps_isig = np.delete(namps, midx), np.delete(namps_isig, midx)
        fit_lines = np.delete(fit_lines, midx)

    elif np.any(eline == '[NII-]') and np.any(eline == '[NII+]'):
        print '-> Removing [NII]6548 from directly being fit'
        print '-> This line will still help constrain [NII]6583 via their intrinsic flux ratio'

        midx = int(np.where(eline == '[NII-]')[0])
        namps, namps_isig = np.delete(namps, midx), np.delete(namps_isig, midx)
        fit_lines = np.delete(fit_lines, midx)

    print
    print
    print '- '+colored('AFTER any removals:', 'magenta', attrs=['bold'])+' -'
    print 'Emission lines to fit: '+colored(fit_lines, 'green')
    print "In shorthand (shouldn't change): "+colored(eline, 'green')
    print "Rest wavelengths of these lines (shouldn't change): "+colored(rest_wave, 'green')
    print
    print


    if count == len(comp_bands)-1:
        pp.close()

    
        
# sys.exit()
# for i in range(3):

    ndim = len(namps) + amp_begin
        
    if fixed_widths == False:
        nar_width = None 
        amplitude_ratio = None
        broad_width = prov_sbroad
            
    else:
        broad_width = bwidth
        amplitude_ratio = amp_ratio
        if bands == 'YJ':
            nar_width = None
        else:
            nar_width = nwidth
        
    while True:
        try:
            p0 = np.zeros((nwalkers, ndim))

            p0[:,0] = m + np.random.randn(nwalkers) * (m / 3.)
            p0[:,1] = b + np.random.randn(nwalkers) * (b / 3.)
            
            if fixed_widths == False:
                if prov_sbroad is None:
                    p0[:,2] = bwidth + np.random.randn(nwalkers) * 2.5
   
                p0[:,amp_begin-2] = nwidth + np.random.randn(nwalkers) * 1.5
                p0[:,amp_begin-1] = amp_ratio + np.random.randn(nwalkers) * 0.5

            elif fixed_widths == True and bands == 'YJ':
                p0[:,2] = nwidth + np.random.randn(nwalkers) * 1.5
                
            p0[:,amp_begin:] = namps + np.random.randn(nwalkers, len(namps)) * np.divide(namps, 3.)
            

            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
                                            args=(wavelengths, luminosities, lum_errors, pbounds, eline, rest_wave, amp_begin, fixed_widths, broad_width, nar_width, amplitude_ratio), \
                                            live_dangerously=False)

            print '- '+colored('FITTING SPECTRA:', 'cyan')+' -'
            print
            print '\nRUNNING A BURN-IN OF ' + colored(burn, 'cyan') + ' STEPS:\n'
            
            sys.stdout = terminal_only

            for k, result in enumerate(sampler.sample(p0, iterations=burn)):
                position, state = result[0], result[2]
                n = int((width+1) * float(k) / burn)
                sys.stdout.write("\r<{0}{1}]".format('#' * n, ' ' * (width - n)))
                if (k+1) % 100 == 0:
                    print("{0:5.1%}".format(float(k) / burn)),
            sys.stdout.write("\n")

            sampler.reset()

            ##Get total run-time of sampler
            start_time = time.time()

            sys.stdout = Logger(logname=logname_base, mode='a')
            
            print '\nRUNNING FULL MCMC RUN OF ' + colored(iterations, 'cyan') + ' STEPS:\n'

            sys.stdout = terminal_only

            for k, result in enumerate(sampler.sample(position, iterations=iterations, rstate0=state)):
                n = int((width+1) * float(k) / iterations)
                sys.stdout.write("\r<{0}{1}]".format('#' * n, ' ' * (width - n)))
                if (k+1) % 100 == 0:
                    print("{0:5.1%}".format(float(k) / iterations)),
            sys.stdout.write("\n")

            end_time = time.time()
            tot_time = end_time - start_time

            sys.stdout = Logger(logname=logname_base, mode='a')

            print '\nTotal sampler run-time (not including burn-in)\n','--- %.5s seconds ---' % (tot_time),'===> --- %.5s minutes ---' % (tot_time / 60.),'\n'


            acceptance = sampler.acceptance_fraction
            print 'Acceptance fraction for ' + colored(nwalkers, 'cyan') + ' walkers:\n', acceptance
            print
            print 'The average of the acceptance fractions: ', '%.4f' % np.mean(acceptance)
            print 'The median of the acceptance fractions: ',  '%.4f' % np.median(acceptance)
            print 'The minimum and maximum acceptance fractions: ', np.amin(acceptance), np.amax(acceptance)
            print

            flatchain = sampler.chain[:,:,:].reshape((-1, ndim))
            print 'Shape of flatchain: ', flatchain.shape, '\n'

            if plt_corner or quick_fit:
                
                pp_quicklook = PdfPages(output_path + 'corner_model_plots_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_descr+'.pdf')

                max_prob_fit_vals = np.array([])
                corner_hist_nbins = []

                for i in range(ndim):
                    binsize = np.std(flatchain[:,i])/std_div
                    bincounts, binedges = np.histogram(flatchain[:,i], bins=np.arange(min(flatchain[:,i]), max(flatchain[:,i])+binsize, binsize), density=False)
                    bincenters = np.add(binedges, binsize/2.)[0:-1]
                    most_prob_val = bincenters[bincounts == max(bincounts)]
                    max_prob_fit_vals = np.append(max_prob_fit_vals, most_prob_val)
                    corner_hist_nbins.append(len(bincounts))
                
                if quick_fit:
                    
                    slope = max_prob_fit_vals[0]
                    y_intercept = max_prob_fit_vals[1]
                    narrow_amps = max_prob_fit_vals[amp_begin:]

                    if fixed_widths == False:
                        if prov_sbroad is None:
                            bwidth = max_prob_fit_vals[2]
                            
                        nwidth = max_prob_fit_vals[amp_begin-2]
                        amp_ratio = max_prob_fit_vals[amp_begin-1]

                    elif fixed_widths == True and bands == 'YJ':
                        nwidth = max_prob_fit_vals[2]

                    if np.any(eline == '[OIII-]') and np.any(eline == '[OIII+]'):
                        narrow_amps = np.insert(narrow_amps, midx, narrow_amps[-1]/o3_ratio)
                        all_params  = np.insert(max_prob_fit_vals, midx+amp_begin, narrow_amps[-1]/o3_ratio)                        
        
                    elif np.any(eline == '[NII-]') and np.any(eline == '[NII+]'):
                        narrow_amps = np.insert(narrow_amps, midx, narrow_amps[-1]/n2_ratio)
                        all_params  = np.insert(max_prob_fit_vals, midx+amp_begin, narrow_amps[-1]/n2_ratio)
                    
                    else:
                        all_params  = max_prob_fit_vals.copy()

                    narrow_comp_params = np.append(np.array([slope, y_intercept, nwidth]), narrow_amps)
                    broad_comp_params  = np.append(np.array([slope, y_intercept, bwidth]), np.divide(narrow_amps, amp_ratio))

                    broad_comp  = G1model(wavelengths, *broad_comp_params)
                    narrow_comp = G1model(wavelengths, *narrow_comp_params)
                    total_model = G2model(wavelengths, *all_params)

                    plot_model(wavelengths, luminosities, lum_errors, broad_comp, narrow_comp, total_model, rest_wave, which='quicklook', pp=pp_quicklook, **model_kwargs)

                if plt_corner:
                    
                    if fixed_widths == False:
                        param_names = ['Slope', 'y-Int', r'$\sigma_{\rm B}$', r'$\sigma_{\rm N}$', 'ARatio'] + [line+' NA' for line in fit_lines]

                        if prov_sbroad is not None:
                            param_names.remove(r'$\sigma_{\rm B}$')

                    else:
                        param_names = ['Slope', 'y-Int'] + [line+' NA' for line in fit_lines]

                        if bands == 'YJ':
                            param_names.insert(2, r'$\sigma_{\rm N}$')

                    if len(param_names) != ndim:
                        raise ValueError('The number of named parameters for the corner plot does not match the number of dimensions fit with the MCMC method')

                    flatchain_scaled  = flatchain.copy()
                    max_prob_scaled   = max_prob_fit_vals.copy()
                    scale_exp_str     = str(scale_fact)[str(scale_fact).find('e')+1:].strip('+')

                    for i in range(ndim):
                        if param_names[i] not in [r'$\sigma_{\rm B}$', r'$\sigma_{\rm N}$', 'ARatio']:
                            flatchain_scaled[:,i] /= scale_fact
                            max_prob_scaled[i]    /= scale_fact
                            param_names[i]        += r' ($\times 10^{'+scale_exp_str+'}$)'

                    param_name_titles = param_names[:]

                    for i in range(ndim):
                        if i < amp_begin:
                            param_name_titles[i] += ' = '+str('%.4f' % max_prob_scaled[i])
                        else:
                            param_name_titles[i] += '\n'+str('%.4f' % max_prob_scaled[i])


                    sns.set_style({"xtick.direction": "out","ytick.direction": "out"})

                    fig = corner(flatchain_scaled, bins=corner_hist_nbins, labels=param_names, titles=param_name_titles, title_fmt=None, \
                                 plot_contours=True, use_math_text=True, show_titles=True, hist_kwargs={'color': 'blue'})

                    corner_axes = np.array(fig.axes).reshape((ndim, ndim))

                    for i in range(ndim):
                        ax = corner_axes[i,i]
                        ax.axvline(max_prob_scaled[i], color='xkcd:red', alpha=0.8)

                    for yi in range(ndim):
                        for xi in range(yi):
                            ax = corner_axes[yi, xi]
                            ax.axvline(max_prob_scaled[xi], color='xkcd:red', alpha=0.5)
                            ax.axhline(max_prob_scaled[yi], color='xkcd:red', alpha=0.5)
                            #ax.plot(max_prob_scaled[xi], max_prob_scaled[yi], color='xkcd:scarlet', marker='s')

                    fig.suptitle(('Priors:\n'
                                  r'$\sigma_{\rm B}$: '+str('%.1f' % pbounds.bsig[0])+' - '+str('%.1f' % pbounds.bsig[1])+' (km/s)\n'
                                  r'$\sigma_{\rm N}$: '+str('%.1f' % pbounds.nsig[0])+' - '+str('%.1f' % pbounds.nsig[1])+' (km/s)\n'
                                  'Amp Ratio (N/B): '+str('%.1f' % pbounds.aratio[0])+' - '+str('%.1f' % pbounds.aratio[1])+'\n'
                                  'Narrow Amps: '+str('%.1f' % pbounds.namps[0])+' - '+str('%.1e' % pbounds.namps[1])+r' (erg/s/$\rm \AA$)'+'\n\n'
                                  'Fixed Values:\n'
                                  r'$\sigma_{\rm B}$ Provided: '+'%.5s' % str(broad_width)+' (km/s)\n'
                                  r'$\sigma_{\rm N}$ Provided: '+'%.4s' % str(nar_width)+' (km/s)\n'
                                  'Amp Ratio (N/B) Provided: '+'%.4s' % str(amplitude_ratio)), \
                                  x=0.5, ha='left') 
                    
                    # fig.savefig(output_path + 'corner_plot_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_descr+'.pdf')
                    pp_quicklook.savefig(fig)
                    plt.close(fig)

                    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

                    del flatchain_scaled

                pp_quicklook.close()

            actime, window_size = sampler.get_autocorr_time(low=10, high=5000, step=1, c=10)

            break

        except emcee.autocorr.AutocorrError as err:
            del flatchain
            print '\n',colored(err,'yellow'),'\n\n',colored('RE-RUNNING FIT (WITH DIFFERENT INITIAL GUESSES)','yellow'),'\n'

    del sampler

    print 'Auto-correlation times for each parameter\n',actime,'\n'
    print 'The final window size that found the auto-correlation time\n',window_size,'\n'
    print '10x the auto-correlation times for each parameter\n',np.multiply(actime,10.),'\n'

# sys.exit()
# for i in range(3):

    print
    print '-> Generating fit parameter posterior probability distributions'
    print


    pp2 = PdfPages(output_path + 'param_hists/two_gaussian_fits/'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_parameter_hists_no_offset_'+run_descr+'.pdf')

    kwargs = dict(title='Filters: '+bands+'   Stacked via: '+stack_meth+'   Norm by: '+norm_eline, ylabel='Number of Instances', std_div=std_div, pp=pp2)


    fit_dict['Filters'] = np.append(fit_dict['Filters'], bands)
    fit_dict['Stacking_Method'] = np.append(fit_dict['Stacking_Method'], stack_meth)
    fit_dict['Norm_Eline']  = np.append(fit_dict['Norm_Eline'], norm_eline)
    fit_dict['Uncertainty'] = np.append(fit_dict['Uncertainty'], uncert)

    added_BN_key_endings = ['_Narrow_Amp', '_Broad_Lum', '_Broad_Lum_sig', '_Narrow_Lum', '_Narrow_Lum_sig', '_Total_Lum', '_Total_Lum_sig']
    
    fstf = '{: <53}'.format

    if fixed_widths:
        fit_dict['Sigma_Broad'][count] = bwidth
        fit_dict['Amplitude_Ratio'][count] = amp_ratio

        print fstf('The fixed broad width for the filter is: '),  colored( ff(bwidth), 'green')

        if bands == 'YJ':
            nwidth = flatchain[:,2]
            
            fit_dict['Sigma_Narrow'][count], _, _, narrow_sigma_sig, pp2 = pf.posterior_gen(nwidth, xlabel=r'$\sigma_{\rm N}$ ($km\ s^{-1}$)', color='xkcd:cement', **kwargs)
            print fstf('The most probable narrow width for the filter is: '), colored( ff(fit_dict['Sigma_Narrow'][count]),'green'),    ' +/-', colored( ff(narrow_sigma_sig),'green')

        else:
            fit_dict['Sigma_Narrow'][count] = nwidth
            print fstf('The fixed narrow width for the filter is: '), colored( ff(nwidth), 'green')
        
        print
        print fstf('The fixed component amplitude ratio (N/B) is: '),   colored( ff(amp_ratio), 'green')

        
    else:
        if prov_sbroad is None:
            bwidth = flatchain[:,2]

        nwidth    = flatchain[:,amp_begin-2]
        amp_ratio = flatchain[:,amp_begin-1]

        try:
            if prov_sbroad is None:
                fit_dict['Sigma_Broad'][count], _, _, broad_sigma_sig, pp2 = pf.posterior_gen(bwidth, xlabel=r'$\sigma_{\rm B}$ ($km\ s^{-1}$)', color='xkcd:wheat', **kwargs)
                
                print fstf('The most probable broad width for the filter is: '),  colored( ff(fit_dict['Sigma_Broad'][count]),'green'),     ' +/-', colored( ff(broad_sigma_sig),'green')

            else:
                fit_dict['Sigma_Broad'][count] = bwidth
                
                print fstf('The fixed broad width is: '),  colored( ff(bwidth), 'green')

            fit_dict['Sigma_Narrow'][count], _, _, narrow_sigma_sig, pp2 = pf.posterior_gen(nwidth, xlabel=r'$\sigma_{\rm N}$ ($km\ s^{-1}$)', color='xkcd:cement', **kwargs)
            fit_dict['Amplitude_Ratio'][count], _, _, amp_ratio_sig, pp2 = pf.posterior_gen(amp_ratio, xlabel='Narrow Amplitude / Broad Amplitude', color='xkcd:aqua blue', **kwargs)

            print fstf('The most probable narrow width for the filter is: '), colored( ff(fit_dict['Sigma_Narrow'][count]),'green'),    ' +/-', colored( ff(narrow_sigma_sig),'green')
            print fstf('The most probable fit amplitude ratio (N/B) is: '),   colored( ff(fit_dict['Amplitude_Ratio'][count]),'green'), ' +/-', colored( ff(amp_ratio_sig),'green')

        except Exception as error:

            if prov_sbroad is None:
                plot_hist_icof(bwidth, np.std(bwidth)/std_div, color='xkcd:wheat', xlabel=r'$\sigma_{\rm B}$ ($km\ s^{-1}$)', title=norm_eline+'  ---  '+stack_meth+'  ---  '+bands, \
                               filename=output_path + 'broad_width_histogram_'+bands+'-bands_'+norm_eline+'_'+stack_meth+'_icof.pdf')

            plot_hist_icof(nwidth, np.std(nwidth)/std_div, color='xkcd:cement', xlabel=r'$\sigma_{\rm N}$ ($km\ s^{-1}$)', title=norm_eline+'  ---  '+stack_meth+'  ---  '+bands, \
                           filename=output_path + 'narrow_width_histogram_'+bands+'-bands_'+norm_eline+'_'+stack_meth+'_icof.pdf')

            plot_hist_icof(amp_ratio, np.std(amp_ratio)/std_div, color='xkcd:aqua blue', xlabel='Narrow Amplitude / Broad Amplitude', title=norm_eline+'  ---  '+stack_meth+'  ---  '+bands, \
                           filename=output_path + 'amp_ratio_histogram_'+bands+'-bands_'+norm_eline+'_'+stack_meth+'_icof.pdf')

            raise error
        

    print

    fit_narrow_amps = np.array([])

    for j, line in enumerate(fit_lines):

        namp_chain = flatchain[:,amp_begin+j]
        bamp_chain = np.divide(namp_chain, amp_ratio)

        if eline[-1] == '[OIII-]' and line == 'OIII4959':
            print '-> Multiplying the broad and narrow amplitude chains of '+colored(line, 'magenta')+' by '+colored(o3_ratio, 'magenta')
            print
            bamp_chain = np.multiply(bamp_chain, o3_ratio)
            namp_chain = np.multiply(namp_chain, o3_ratio)
            line = 'OIII5007'

        # bamp_max, _, _, bamp_sig, pp2 = pf.posterior_gen(bamp_chain, xlabel=line+' Broad Amplitude', axis_sn='x', color='xkcd:tomato red', **kwargs)
        namp_max, _, _, namp_sig, pp2 = pf.posterior_gen(namp_chain, xlabel=line+' Narrow Amplitude', axis_sn='x', color='xkcd:tangerine', **kwargs)

        fit_narrow_amps = np.append(fit_narrow_amps, namp_max)        

        print '-> Converting the width chains from velocity (km/s) to Angstroms using '+colored(line, 'magenta'),
        print ' at '+colored(elines_restframe.loc[line, 'Rest_Lambda'], 'magenta')+' A'
        print

        kms_to_a = elines_restframe.loc[line, 'Rest_Lambda'] / c

        broad_lum  = np.multiply(np.multiply(bamp_chain, np.multiply(bwidth, kms_to_a)), np.sqrt(2. * np.pi))
        narrow_lum = np.multiply(np.multiply(namp_chain, np.multiply(nwidth, kms_to_a)), np.sqrt(2. * np.pi))

        tot_lum = np.add(broad_lum, narrow_lum)

        blum_max, _, _, blum_sig, pp2 = pf.posterior_gen(broad_lum, xlabel=line+' Broad Luminosity', axis_sn='x', color='xkcd:dark sky blue', **kwargs)
        nlum_max, _, _, nlum_sig, pp2 = pf.posterior_gen(narrow_lum, xlabel=line+' Narrow Luminosity', axis_sn='x', color='xkcd:steel', **kwargs)

        tot_lum_max, _, _, tot_lum_sig, pp2 = pf.posterior_gen(tot_lum, xlabel=line+' Total Luminosity', axis_sn='x', color='xkcd:light plum', **kwargs)

        added_fit_dict_values = [namp_max, blum_max, blum_sig, nlum_max, nlum_sig, tot_lum_max, tot_lum_sig]

        if len(added_BN_key_endings) != len(added_fit_dict_values):
            raise ValueError('For line '+line+', the number of added keys to the dictionary of fit values does not match the number of added values')

        for idx, key_ending in enumerate(added_BN_key_endings):
            fit_dict[line+key_ending] = np.full(len(comp_bands), np.nan)
            fit_dict[line+key_ending][count] = added_fit_dict_values[idx]

        # print fstf('The most probable broad amplitude for '+line+' is: '),   colored( snf(bamp_max),'green'), ' +/-', colored( snf(bamp_sig),'green')
        print fstf('The most probable narrow amplitude for '+line+' is: '),  colored( snf(namp_max),'green'), ' +/-', colored( snf(namp_sig),'green')
        print
        print fstf('The most probable broad luminosity for '+line+' is: '),  colored( snf(blum_max),'green'), ' +/-', colored( snf(blum_sig),'green')
        print fstf('The most probable narrow luminosity for '+line+' is: '), colored( snf(nlum_max),'green'), ' +/-', colored( snf(nlum_sig),'green')
        print
        print fstf('The most probable TOTAL luminosity for '+line+' is: '),  colored( snf(tot_lum_max),'green'), ' +/-', colored( snf(tot_lum_sig),'green')
        print

    print
                                    
    fit_dict['y-Int'][count], _, _, fit_dict['y-Int_sig'][count], pp2 = pf.posterior_gen(flatchain[:,1], xlabel='y-Intercept', axis_sn='x', color='xkcd:bright violet', **kwargs)
    fit_dict['Slope'][count], _, _, fit_dict['Slope_sig'][count], pp2 = pf.posterior_gen(flatchain[:,0], xlabel='Slope', axis_sn='x', color='xkcd:dark hot pink', **kwargs)

    print fstf('The most probable y-intercept for the continuum is: '), colored( snf(fit_dict['y-Int'][count]),'green'), ' +/-', colored( snf(fit_dict['y-Int_sig'][count]),'green')
    print fstf('The most probable slope for the continuum is: '),       colored( snf(fit_dict['Slope'][count]),'green'), ' +/-', colored( snf(fit_dict['Slope_sig'][count]),'green')
    print

    del flatchain
    
    pp2.close()

    print '-> Plotting the fit model on top of the stacked spectrum'
    print
    print

    if np.any(eline == '[OIII-]') and np.any(eline == '[OIII+]'):
        fit_narrow_amps = np.insert(fit_narrow_amps, midx, fit_narrow_amps[-1]/o3_ratio)
        
    elif np.any(eline == '[NII-]') and np.any(eline == '[NII+]'):
        fit_narrow_amps = np.insert(fit_narrow_amps, midx, fit_narrow_amps[-1]/n2_ratio)

    elif eline[-1] == '[OIII-]':
        fit_narrow_amps[-1] /= o3_ratio

    narrow_comp_params = np.append(np.array([fit_dict['Slope'][count], fit_dict['y-Int'][count], fit_dict['Sigma_Narrow'][count]]), \
                                   fit_narrow_amps)
    broad_comp_params  = np.append(np.array([fit_dict['Slope'][count], fit_dict['y-Int'][count], fit_dict['Sigma_Broad'][count]]), \
                                   np.divide(fit_narrow_amps, fit_dict['Amplitude_Ratio'][count]))

    if fixed_widths == False:
        all_fit_params = np.append(np.array([fit_dict['Slope'][count], fit_dict['y-Int'][count], fit_dict['Sigma_Broad'][count], \
                                             fit_dict['Sigma_Narrow'][count], fit_dict['Amplitude_Ratio'][count]]), \
                                   fit_narrow_amps)

        if prov_sbroad is not None:
            all_fit_params = np.delete(all_fit_params, 2)

    else:
        all_fit_params = np.append(np.array([fit_dict['Slope'][count], fit_dict['y-Int'][count]]), fit_narrow_amps)

        if bands == 'YJ':
            all_fit_params = np.insert(all_fit_params, 2, fit_dict['Sigma_Narrow'][count])
            
    broad_comp   = G1model(wavelengths, *broad_comp_params)
    narrow_comp  = G1model(wavelengths, *narrow_comp_params)
    total_model  = G2model(wavelengths, *all_fit_params)

    plot_model(wavelengths, luminosities, lum_errors, broad_comp, narrow_comp, total_model, rest_wave, which='Final', pp=pp3, **model_kwargs)
    

pp3.close()


cols = [fits.Column(name='Filters', format='3A', array=fit_dict['Filters']), \
        fits.Column(name='Stacking_Method', format='15A', array=fit_dict['Stacking_Method']), \
        fits.Column(name='Norm_Eline', format='10A', array=fit_dict['Norm_Eline']), \
        fits.Column(name='Uncertainty', format='11A', array=fit_dict['Uncertainty'])]

for key in fit_dict.keys()[4:]:
    cols = cols + [fits.Column(name=key, format='D', array=fit_dict[key])]

hdu = fits.BinTableHDU.from_columns(cols)
hdu.writeto(output_path + 'two_gaussian_fit_parameters_stacked_spectra_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_descr+'.fits', overwrite=True)

print '-> '+colored('two_gaussian_fit_parameters_stacked_spectra_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_descr+'.fits', 'green')+' written'
print 'at location: '+colored(output_path, 'white')
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print
