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

print


parser = argparse.ArgumentParser(description = 'Markov Chain Monte Carlo (MCMC) spectral fitting code for composite spectrum of stacked galaxy sample')

parser.add_argument('-p', '--Parameter_File', metavar='str', \
                    help='The FITS file with the ALREADY-fit "narrow" and "broad" component line widths (km/s) to use and fix here')

parser.add_argument('Normalizing_Eline', choices=['OIII5007', 'H-alpha'], \
                    help='The emission-line name of the line used to normalize the spectra during stacking')

parser.add_argument('Stacking_Method', choices=['median', 'average'], \
                    help='The method with which the spectra in the sample were stacked')

parser.add_argument('Uncertainty', choices=['bootstrap', 'statistical'], \
                    help='How the uncertainty spectrum was calculated (including cosmic variance or just statistically)')

args = parser.parse_args()

param_file = args.Parameter_File
norm_eline = args.Normalizing_Eline
stack_meth = args.Stacking_Method
uncert     = args.Uncertainty

if param_file is None: ## I am fitting the widths in this version of the run
    fixed_widths = False
    run_descr = 'fitting_widths'
    amp_begin = 4
else:
    fixed_widths = True
    run_descr = 'fw_full_spectrum'
    amp_begin = 2

    

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
        
def model_gaussian(x, mean, amplitude, width):
    return amplitude * np.exp(-0.5*(np.power(x - mean, 2) / np.power(width*mean/c, 2)))


def gaussian(x, mu, sigma):
    return (1./(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*(np.power(x - mu, 2) / np.power(sigma, 2)))


def make_model(emiss_waves, fixed_widths=False, width1=None, width2=None, width3=None, amp_begin=None, MC_lines=None, two_gauss=False):
    def gauss_model(wavelengths, *pars):
        
        m = pars[0]
        b = pars[1]
        
        if two_gauss == True:
            
            if fixed_widths == False:
                bsig = pars[2]
                nsig = pars[3]

                num_MC_lines = len(emiss_waves)
                
            else:
                if width1 == None or width2 == None:
                    raise Exception('''For a model with 2 Gaussian components with fixed widths, 
                                       "width1" and "width2" should be specified (in km/s)'''
                                   )
                bsig = width1
                nsig = width2

                if width3 != None:
                    ssig = width3

                if MC_lines == None:
                    raise ValueError('''For a model with both multi-Gaussian and single-Gaussian line profiles,
                                        the number of multi-Gaussian lines must be provided'''
                                    )

                num_MC_lines = MC_lines

            bamps = np.array(pars[amp_begin : amp_begin+num_MC_lines])
            namps = np.array(pars[amp_begin+num_MC_lines : amp_begin+2*num_MC_lines])
            samps = np.array(pars[amp_begin+2*num_MC_lines:])


            if len(bamps) != len(namps):
                raise ValueError('''The number of amplitude free parameters should be 
                                    the same for both the "broad" and "narrow" Gaussian fits'''
                                )

            
            elif len(namps)+len(samps) != len(emiss_waves):
                raise ValueError('''The number of amplitude free parameters in either the "broad" or "narrow" 
                                    Gaussian fits combined with the number of amplitude free parameters for single-Gaussian fits
                                    should equal the number of emission lines to fit'''
                                )

        else:
            if fixed_widths == False:
                sig  = pars[2]
                amps = pars[3:]

            else:
                sig  = width1
                amps = pars[2:]
            
            if len(amps) != len(emiss_waves):
                raise ValueError('The number of amplitude free parameters should equal the number of emission lines to fit')
        

        model = linear_cont(wavelengths, m, b)
        
        for j, ewave in enumerate(emiss_waves):
            
            if two_gauss == True:
                if j < num_MC_lines:
                    model += model_gaussian(wavelengths, ewave, bamps[j], bsig) + model_gaussian(wavelengths, ewave, namps[j], nsig)
                else:
                    model += model_gaussian(wavelengths, ewave, samps[j-num_MC_lines], ssig)

            else:
                model += model_gaussian(wavelengths, ewave, amps[j], sig)

        return model
    
    return gauss_model
        
    


##Establishing priors in order to prevent runaway walkers
def lnprior(pos, eline, rest_wave, amp_begin, fixed_widths, MC_lines):
    
    m = pos[0]
    b = pos[1]

    if fixed_widths == False:
        bsig = pos[2]
        nsig = pos[3]

    bamps = pos[amp_begin : amp_begin+MC_lines]
    namps = pos[amp_begin+MC_lines : amp_begin+2*MC_lines]
    samps = pos[amp_begin+2*MC_lines:]

    if np.any(eline == '[OIII-]') and np.any(eline == '[OIII+]'):
        midx  = int(np.where(rest_wave == 4958.910)[0])
        bamps = np.insert(bamps, midx, bamps[-1]/o3_ratio)
        namps = np.insert(namps, midx, namps[-1]/o3_ratio)
        
    elif np.any(eline == '[NII-]') and np.any(eline == '[NII+]'):
        midx  = int(np.where(rest_wave == 6548.048)[0])
        samps = np.insert(samps, midx - MC_lines, samps[-1]/n2_ratio)


    if fixed_widths == False:
        if 30. <= nsig < 60. and 60. <= bsig <= 90. and \
           np.all(namps >= 0.) and np.all(bamps >= 0.) and np.all(namps <= 1.5e42) and np.all(bamps <= 1.5e42):

            return m, b, bsig, nsig, bamps, namps, 0.0
    
        return m, b, bsig, nsig, bamps, namps, -np.inf

    else:
        if np.all(namps >= 0.) and np.all(bamps >= 0.) and np.all(samps >= 0.) \
           and np.all(namps <= 1.5e42) and np.all(bamps <= 1.5e42) and np.all(samps <= 1.5e42):

            return m, b, bamps, namps, samps, 0.0

        return m, b, bamps, namps, samps, -np.inf



##Establishing the likelihood and product of likelihood and priors.  "pos" is again the position of a walker
def lnprob(pos, wavelengths, luminosities, lum_errors, eline, rest_wave, amp_begin, fixed_widths, broad_width, nar_width, single_width, MC_lines):

    ##pos is the initial position of a walker in the parameter space -> will be an array of values
    ##Call the prior function

    if fixed_widths == False:
        m, b, bsig, nsig, bamps, namps, lnp = lnprior(pos, eline, rest_wave, amp_begin, False, MC_lines)
    else:
        m, b, bamps, namps, samps, lnp = lnprior(pos, eline, rest_wave, amp_begin, True, MC_lines)

        if broad_width is None or nar_width is None:
            raise Exception('''For a model with 2 Gaussian components with fixed widths, 
                               a width for each component should be specified (in km/s)'''
                           )
        bsig = broad_width
        nsig = nar_width
        ssig = single_width

    if np.isinf(lnp):
        return lnp
    
    ##The continuum part (linear) of the flux calculation.  The wavelength range is shifted by the minimum of the range in order to establish an intercept at the minimum value
    model = linear_cont(wavelengths, m, b)

    ##The rest of the flux calculation (Guassian --> Following Class Activity 6)
    for i, ewave in enumerate(rest_wave):

        if i < MC_lines:
            model += model_gaussian(wavelengths, ewave, namps[i], nsig) + model_gaussian(wavelengths, ewave, bamps[i], bsig)
        else:
            model += model_gaussian(wavelengths, ewave, samps[i-MC_lines], ssig)
        
    ##Calculating the posterior at each wavelength
    post = -np.log(np.sqrt(2.*np.pi)*lum_errors) + (-0.5*(np.power(model - luminosities, 2) / np.power(lum_errors, 2)))

    ##Adding all the individual posterior measurements together (They are in log space and are independent of each other)
    tot_prob = np.sum(post)

    ##Return the posterior value (up to a constant)
    return tot_prob


def plot_model(wavelengths, luminosities, luminosity_errors, broad_component, narrow_component, total_model, ewaves, which='', bands='', stack_meth='', norm_eline='', run_description='', pp=None):

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
        pickle_fname = 'Initial_Fit_Guess_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_description+'.fig.pickle'
        
    elif which == 'Final':
        ax.set_title('Fit Model --- Filters: '+bands+'   Stacked via: '+stack_meth+'   Norm by: '+norm_eline)
        pickle_fname = 'Fit_Model_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_description+'.fig.pickle'
        
    else:
        ax.set_title('Stacked Spectrum with Spectral Model and Model Components')
        pickle_fname = 'Stacked_spectrum_with_overplotted_model_and_model_comps.fig.pickle'
        
    ax.set_xlabel(r'Rest-Frame Wavelength ($\AA$)')
    ax.set_ylabel(r'$\rm L_\lambda$ ($erg\ s^{-1}\ \AA^{-1}$)')
    ax.legend(loc='upper left', fontsize='small', fancybox=True, frameon=True, framealpha=0.8, edgecolor='black')
    plt.tight_layout()
    pickle.dump(fig, open(pickle_fname, 'wb'))
    if pp != None:
        pp.savefig()
    else:
        plt.show()
    plt.close(fig)

    return


cwd = os.getcwd()

sys.stdout = Logger(logname=cwd+'/logfiles/fitting_spectra_'+norm_eline+'_'+stack_meth+'_no_offset_'+run_descr, mode='w')

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

nwalkers = 300
burn = 1500  #1500
iterations = 25000  ##25000
width = 50

if fixed_widths == False:
    comp_bands = ['JH', 'HK']
else:
    comp_bands = ['YJ', 'JH', 'HK']
####################################

print 'Review of options called and arguments given to this script:'
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print 'Options:'
print '-> Parameter file containing already-fit "broad" and "narrow" line widths to be used here: ', param_file
print
print 'Arguments:'
print '-> Spectra normalized by: ', norm_eline
print '-> Stacking method used: ', stack_meth
print '-> Uncertainty calculation method: ', uncert
print
print 'MCMC Parameters (currently defined in-script):'
print '-> Number of walkers: ', nwalkers
print '-> Burn-in iterations: ', burn
print '-> Iterations following burn-in: ', iterations
print
print

elines_restframe = pd.read_csv('loi.txt', comment='#', delim_whitespace=True, names=['Eline','Eline_SH','Rest_Lambda'], index_col='Eline', \
                               dtype={'Eline': np.string_, 'Eline_SH': np.string_, 'Rest_Lambda': np.float64}, usecols=[0, 1, 2] \
                              )[['Eline_SH', 'Rest_Lambda']]  ## SH = shorthand


print 'The current working directory and path are: '+colored(cwd, 'cyan')
print

if os.path.isdir(cwd+'/param_hists') == False:
    os.mkdir(cwd + '/param_hists')
    os.mkdir(cwd + '/param_hists/two_gaussian_fits')
    print 'Created directory '+colored(cwd+'/param_hists', 'white')+' and subdirectory '+colored('/two_gaussian_fits', 'white')+' therein'
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
    params = fr.rc(param_file)
    
#stack_meth = fname[len('stacked_spectrum_'+bands+'-bands_') : fname.find('_', len('stacked_spectrum_'+bands+'-bands_'))]
#norm_eline = fname[len('stacked_spectrum_'+bands+'-bands_'+stack_meth+'_') : -len('_noDC.txt')]

pp  = PdfPages('Two_Gaussian_Initial_Fit_Guess_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_descr+'.pdf')
pp3 = PdfPages('Two_Gaussian_Spectral_Fits_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_descr+'.pdf')

fit_dict_keys = ['Filters', 'Stacking_Method', 'Norm_Eline', 'Slope', 'Slope_sig', 'y-Int', 'y-Int_sig', 'Sigma_Broad', 'Sigma_Narrow', 'Sigma_Single']

fit_dict = OrderedDict.fromkeys(fit_dict_keys)

for key in fit_dict_keys:
    if key == 'Filters' or key == 'Stacking_Method' or key == 'Norm_Eline':
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
        if bands == 'JH':
            no_fit = np.where(wavelengths < 4800.)[0]
        elif bands == 'HK':
            #no_fit = np.where(wavelengths < 6450.)[0]
            no_fit = np.where((wavelengths <= 6551.) | (wavelengths >= 6580.))[0]
    else:
        if bands == 'YJ':
            no_fit = np.where(wavelengths < 3660.)[0]  ## To cut out the portion of the spectrum with the Balmer break (~3646 A)
        else:
            no_fit = np.array([])
        
    wavelengths, boot_waves  = np.delete(wavelengths, no_fit), np.delete(boot_waves, no_fit)
    luminosities, lum_errors = np.delete(luminosities, no_fit), np.delete(lum_errors, no_fit) 

    if len(wavelengths) != len(boot_waves) or len(luminosities) != len(lum_errors):
        raise Exception('The wavelength and/or luminosity arrays of the two spectral tables read in are different lengths!')

    equiv_waves = wavelengths == boot_waves

    if equiv_waves.any() == False:
        raise Exception('The wavelength values in the stacked spectrum are not the same as the values in the composite error spectrum!')

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

    scale_fact = 1.0e40
    
    #Guessing initial conditions
    m = 0.
    b = 0.
    amp_nar = np.array([])

    if fixed_widths == False:
        bwidth = 70.
        nwidth = 45.

        ewaves = rest_wave

    else:
        if bands != 'YJ':
            bwidth = params['Sigma_Broad'][params['Filters'] == bands][0]
            nwidth = params['Sigma_Narrow'][params['Filters'] == bands][0]
        else:
            bwidth = params['Sigma_Broad'][params['Filters'] == 'JH'][0]
            nwidth = params['Sigma_Narrow'][params['Filters'] == 'JH'][0]

        weak_lines = ['Hg', '[OIII]', 'HeI', '[NII-]', '[NII+]']

        one_comp_lines = np.array([])  ## Wavelengths of these lines
        two_comp_lines = np.array([])  ## Wavelengths of these lines

        amp_single = np.array([])
        
        if bands == 'JH' or bands == 'HK':
            x = np.linspace(-300., 299., 10000)
            comp_profile = gaussian(x, 0., bwidth) + gaussian(x, 0., nwidth)

            CI = pf.Confidence_Interval(*pf.conf_int(x, comp_profile, 68.))

            swidth = CI.approximate_sigma

            fig, ax = plt.subplots()
            ax.plot(x, gaussian(x, 0., bwidth), color='green', linewidth=0.7, label='Broad Comp')
            ax.plot(x, gaussian(x, 0., nwidth), color='blue', linewidth=0.7, label='Narrow Comp')
            ax.plot(x, comp_profile, color='red', linewidth=0.7, label='Composite', zorder=100)
            ax.fill_between(x[CI.range_minimum_idx : CI.range_maximum_idx+1], 0., comp_profile[CI.range_minimum_idx : CI.range_maximum_idx+1], \
                            alpha=0.3, facecolor='xkcd:charcoal', label='%.2f' % CI.percentage_of_total_area+' %\n'+r'$\sigma$ ~ '+'%.4f' % swidth)
            ax.minorticks_on()
            ax.tick_params(which='both', left=True, right=True, top=True, bottom=True)
            ax.legend(loc='upper left', fontsize='small', fancybox=True, frameon=True, framealpha=0.8, edgecolor='black')
            plt.tight_layout()
            plt.savefig('single_gaussian_est_fixed_width_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_descr+'.pdf')
            plt.close(fig)
            
        else:
            swidth = None

    for lambda_, ename_sh in itertools.izip(rest_wave, eline):
        search_ind = np.where((wavelengths >= lambda_ - 2.) & (wavelengths <= lambda_ + 2.))[0]

        if fixed_widths == False:
            amp_nar = np.append(amp_nar, 0.8 * max(luminosities[search_ind]) / scale_fact)
        else:
            if ename_sh not in weak_lines:
                amp_nar = np.append(amp_nar, 0.8 * max(luminosities[search_ind]) / scale_fact)
                two_comp_lines = np.append(two_comp_lines, round(lambda_, 3))
            else:
                amp_single = np.append(amp_single, max(luminosities[search_ind]) / scale_fact)
                one_comp_lines = np.append(one_comp_lines, round(lambda_, 3))

    if fixed_widths == True:
        print 'Multiple component fit: ', two_comp_lines
        print 'Single component fit: ',   one_comp_lines

        ewaves = np.append(two_comp_lines, one_comp_lines)

        fw_amp_dict = OrderedDict.fromkeys(list(ewaves.astype(str)))

    amp_bro = np.multiply(amp_nar, 0.2)

    luminosities_scaled, lum_errors_scaled = np.divide(luminosities, scale_fact), np.divide(lum_errors, scale_fact)
    

    if fixed_widths == False:

        popt, pcov = curve_fit(make_model(ewaves, amp_begin=amp_begin, two_gauss=True), \
                               wavelengths, luminosities_scaled, sigma=lum_errors_scaled, \
                               p0=[m, b, bwidth, nwidth]+list(amp_bro)+list(amp_nar), \
                               bounds=([-np.inf,-np.inf,60.,30.]+list(np.full(2*len(amp_bro), 0.)), \
                                       [np.inf,np.inf,90.,60.]+list(np.full(2*len(amp_bro), 150.))) \
                              )

    else:
        popt, pcov = curve_fit(make_model(ewaves, fixed_widths=True, width1=bwidth, width2=nwidth, width3=swidth, amp_begin=amp_begin, MC_lines=len(two_comp_lines), two_gauss=True), \
                               wavelengths, luminosities_scaled, sigma=lum_errors_scaled, \
                               p0=[m, b]+list(amp_bro)+list(amp_nar)+list(amp_single), \
                               bounds=([-np.inf,-np.inf]+list(np.full(2*len(amp_bro)+len(amp_single), 0.)), \
                                       [np.inf,np.inf]+list(np.full(2*len(amp_bro)+len(amp_single), 150.))) \
                              )

    psigs = np.sqrt(np.diag(pcov))

    m, m_isig = popt[0]*scale_fact, psigs[0]*scale_fact
    b, b_isig = popt[1]*scale_fact, psigs[1]*scale_fact
    bamps, bamps_isig = popt[amp_begin:amp_begin+len(amp_bro)]*scale_fact, psigs[amp_begin:amp_begin+len(amp_bro)]*scale_fact
    namps, namps_isig = popt[amp_begin+len(amp_bro):amp_begin+2*len(amp_bro)]*scale_fact, psigs[amp_begin+len(amp_bro):amp_begin+2*len(amp_bro)]*scale_fact

    if fixed_widths == False:
        bwidth = popt[2]
        nwidth = popt[3]

    else:    
        samps, samps_isig = popt[amp_begin+2*len(amp_bro):]*scale_fact, psigs[amp_begin+2*len(amp_bro):]*scale_fact

        for i, key in enumerate(fw_amp_dict.keys()):
            if i < len(two_comp_lines):
                fw_amp_dict[key] = (bamps[i], namps[i], bamps_isig[i], namps_isig[i])
            else:
                fw_amp_dict[key] = (samps[i-len(two_comp_lines)], samps_isig[i-len(two_comp_lines)])


    print
    print '- '+colored('Initial fit guesses from non-linear least squares curve fitting:', 'magenta', attrs=['bold'])+' -'
    print
    
    snf  = '{: >11.4e}'.format
    ff   = '{: >11.4f}'.format
    istf = '{: >30}'.format
    
    print istf('Slope = '),       colored( snf(m), 'green'), ' +/-', colored( snf(m_isig), 'green')
    print istf('y-Intercept = '), colored( snf(b), 'green'), ' +/-', colored( snf(b_isig), 'green')
    print
    print istf('Width - Broad (km/s) = '),  colored( ff(bwidth), 'green')
    print istf('Width - Narrow (km/s) = '), colored( ff(nwidth), 'green')

    if fixed_widths == True and (bands == 'JH' or bands == 'HK'):
        print istf('Width - Single (km/s) = '), colored( ff(swidth), 'green')

    print
    
    if fixed_widths == False:
        for j, ename in enumerate(fit_lines):
            print istf(ename + ' Amplitude - Broad = '),  colored( snf(bamps[j]), 'green'), ' +/-', colored( snf(bamps_isig[j]), 'green')            
            print istf(ename + ' Amplitude - Narrow = '), colored( snf(namps[j]), 'green'), ' +/-', colored( snf(namps_isig[j]), 'green')
            print

    else:
        for wave, ename in itertools.izip(sorted(fw_amp_dict.keys()), fit_lines):
            if len(fw_amp_dict[wave]) == 4:
                print istf(ename + ' Amplitude - Broad = '),  colored( snf(fw_amp_dict[wave][0]), 'green'), ' +/-', colored( snf(fw_amp_dict[wave][2]), 'green')
                print istf(ename + ' Amplitude - Narrow = '), colored( snf(fw_amp_dict[wave][1]), 'green'), ' +/-', colored( snf(fw_amp_dict[wave][3]), 'green')
                print

            else:
                print istf(ename + ' Amplitude - Single = '), colored( snf(fw_amp_dict[wave][0]), 'green'), ' +/-', colored( snf(fw_amp_dict[wave][1]), 'green')
                print
        
    print

    # with np.printoptions(formatter={'float': '{: .6f}'.format}):
    #     print popt
    # print
    # with np.printoptions(formatter={'float': '{: .6f}'.format}):
    #     print np.sqrt(np.diag(pcov))
    # print

    print '-> Plotting the initial guess model on top of the stacked spectrum'

    if fixed_widths == False:
        G2model = make_model(ewaves, amp_begin=amp_begin, two_gauss=True)
        G1model = make_model(ewaves, two_gauss=False)

        broad_comp  = G1model(wavelengths, m, b, bwidth, *bamps)
        narrow_comp = G1model(wavelengths, m, b, nwidth, *namps)
        total_model = G2model(wavelengths, *popt) * scale_fact

    else:
        G2model = make_model(ewaves, fixed_widths=True, width1=bwidth, width2=nwidth, width3=swidth, amp_begin=amp_begin, MC_lines=len(two_comp_lines), two_gauss=True)
        G1model_bro = make_model(two_comp_lines, fixed_widths=True, width1=bwidth, two_gauss=False)
        G1model_nar = make_model(two_comp_lines, fixed_widths=True, width1=nwidth, two_gauss=False)

        broad_comp  = G1model_bro(wavelengths, m, b, *bamps)
        narrow_comp = G1model_nar(wavelengths, m, b, *namps)
        total_model = G2model(wavelengths, *popt) * scale_fact

    model_kwargs = dict(bands=bands, stack_meth=stack_meth, norm_eline=norm_eline, run_description=run_descr)

    plot_model(wavelengths, luminosities, lum_errors, broad_comp, narrow_comp, total_model, rest_wave, which='Initial', pp=pp, **model_kwargs)

    print '-> Plot saved to PDF and binary file for later interactive use'
    print
    print


    if np.any(eline == '[OIII-]') and np.any(eline == '[OIII+]'):
        print '-> Removing [OIII]4959 from directly being fit'
        print '-> This line will still help constrain [OIII]5007 via their intrinsic flux ratio'

        midx = int(np.where(ewaves == 4958.910)[0])
        bamps, bamps_isig = np.delete(bamps, midx), np.delete(bamps_isig, midx)
        namps, namps_isig = np.delete(namps, midx), np.delete(namps_isig, midx)

        if fixed_widths == True:
            two_comp_lines = np.delete(two_comp_lines, midx)

        fit_lines = np.delete(fit_lines, int(np.where(fit_lines == 'OIII4959')[0]))

    if fixed_widths == True:
        if np.any(eline == '[NII-]') and np.any(eline == '[NII+]'):
            print '-> Removing [NII]6548 from directly being fit'
            print '-> This line will still help constrain [NII]6583 via their intrinsic flux ratio'

            midx = int(np.where(ewaves == 6548.048)[0])
            samps, samps_isig = np.delete(samps, midx - len(two_comp_lines)), np.delete(samps_isig, midx - len(two_comp_lines))
            one_comp_lines = np.delete(one_comp_lines, midx - len(two_comp_lines))

            fit_lines = np.delete(fit_lines, int(np.where(fit_lines == 'NII6548')[0]))
        
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
        
#sys.exit()
#for i in range(3):
        
    if fixed_widths == False:
        ndim = 2 * len(namps) + 4
        broad_width, nar_width, single_width = None, None, None
        MC_lines = len(ewaves)
    else:
        ndim = 2 * len(namps) + len(samps) + 2
        broad_width, nar_width, single_width = bwidth, nwidth, swidth
        MC_lines = len(bamps)
        
    while True:
        try:
            p0 = np.zeros((nwalkers, ndim))

            p0[:,0] = m + np.random.randn(nwalkers) * (m / 3.)
            p0[:,1] = b + np.random.randn(nwalkers) * (b / 3.)
            
            if fixed_widths == False:
                p0[:,2] = 75. + np.random.randn(nwalkers) * 2.0
                p0[:,3] = nwidth + np.random.randn(nwalkers) * 1.5
                
            p0[:,amp_begin:amp_begin+len(bamps)] = bamps + np.random.randn(nwalkers, len(bamps)) * np.divide(bamps, 3.)
            p0[:,amp_begin+len(bamps):amp_begin+2*len(bamps)] = namps + np.random.randn(nwalkers, len(namps)) * np.divide(namps, 3.)

            if fixed_widths == True:
                p0[:,amp_begin+2*len(bamps):] = samps + np.random.randn(nwalkers, len(samps)) * np.divide(samps, 3.)
            

            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
                                            args=(wavelengths, luminosities, lum_errors, eline, ewaves, amp_begin, fixed_widths, broad_width, nar_width, single_width, MC_lines), \
                                            live_dangerously=False \
                                           )

            print '- '+colored('FITTING SPECTRA:', 'cyan')+' -'
            print
            print '\nRUNNING A BURN-IN OF ' + str(burn) + ' STEPS:\n'

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

            print '\nRUNNING FULL MCMC RUN OF ' + str(iterations) + ' STEPS:\n'

            for k, result in enumerate(sampler.sample(position, iterations=iterations, rstate0=state)):
                n = int((width+1) * float(k) / iterations)
                sys.stdout.write("\r<{0}{1}]".format('#' * n, ' ' * (width - n)))
                if (k+1) % 100 == 0:
                    print("{0:5.1%}".format(float(k) / iterations)),
            sys.stdout.write("\n")

            end_time = time.time()
            tot_time = end_time - start_time

            print '\nTotal sampler run-time (not including burn-in)\n','--- %.5s seconds ---' % (tot_time),'===> --- %.5s minutes ---' % (tot_time / 60.),'\n'


            acceptance = sampler.acceptance_fraction
            print 'Acceptance fraction for ' + str(nwalkers) + ' walkers:\n',acceptance,'\n'

            flatchain = sampler.chain[:,:,:].reshape((-1, ndim))

            # if fixed_widths == False:
            #     param_names = ['Slope', 'y-Int', r'$\sigma_{\rm B}$', r'$\sigma_{\rm N}$']+ \
            #                   [line+' BA' for line in fit_lines]+ \
            #                   [line+' NA' for line in fit_lines]
            # else:
            #     param_names = ['Slope', 'y-Int']+ \
            #                   [str(int(line))+' BA' for line in two_comp_lines]+ \
            #                   [str(int(line))+' NA' for line in two_comp_lines]+ \
            #                   [str(int(line))+' SA' for line in one_comp_lines]
            
            # fig = corner(flatchain, labels=param_names, plot_contours=True, use_math_text=True, show_titles=True, title_fmt='.2e', quantiles=[0.16,0.5,0.84])
            # fig.savefig('corner_plot_'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_descr+'.pdf')
            # plt.close(fig)

            actime, window_size = sampler.get_autocorr_time(low=10, high=5000, step=1, c=10)

            break

        except emcee.autocorr.AutocorrError as err:
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


    pp2 = PdfPages(cwd+'/param_hists/two_gaussian_fits/'+bands+'-bands_'+stack_meth+'_'+norm_eline+'_parameter_hists_no_offset_'+run_descr+'.pdf')

    kwargs = dict(title='Filters: '+bands+'   Stacked via: '+stack_meth+'   Norm by: '+norm_eline, ylabel='Number of Instances', std_div=20., pp=pp2)


    fit_dict['Filters'] = np.append(fit_dict['Filters'], bands)
    fit_dict['Stacking_Method'] = np.append(fit_dict['Stacking_Method'], stack_meth)
    fit_dict['Norm_Eline'] = np.append(fit_dict['Norm_Eline'], norm_eline)

    added_BN_key_endings = ['_Broad_Amp', '_Narrow_Amp', '_Broad_Lum', '_Broad_Lum_sig', '_Narrow_Lum', '_Narrow_Lum_sig', '_Total_Lum', '_Total_Lum_sig']
    added_S_key_endings  = ['_Single_Amp', '_Total_Lum', '_Total_Lum_sig']

    fit_broad_amps, fit_narrow_amps, fit_single_amps = np.array([]), np.array([]), np.array([])

    fstf = '{: <53}'.format

    if fixed_widths == True:
        amp_waves = np.append(np.append(two_comp_lines, two_comp_lines), one_comp_lines)

        amp_comps = np.chararray(len(amp_waves), itemsize=1)
        amp_comps[0:len(two_comp_lines)] = 'B'
        amp_comps[len(two_comp_lines):2*len(two_comp_lines)] = 'N'
        amp_comps[2*len(two_comp_lines):] = 'S'

        sorted_ind = np.argsort(amp_waves)

        amp_comps  = amp_comps[sorted_ind]

        flatchain  = flatchain[:, [0,1]+list(sorted_ind+2)]

        print 'Ascending-order sort of emission-line wavelengths: ', amp_waves[sorted_ind]
        print 'Gaussian component to plot posterior of: ', amp_comps
        print

        ampcount = 0

        bsig = bwidth
        nsig = nwidth
        ssig = swidth

    else:
        bsig = flatchain[:,2]
        nsig = flatchain[:,3]
        

    for j, line in enumerate(fit_lines):
        
        if fixed_widths == False or (fixed_widths == True and amp_comps[ampcount] == 'B'):

            if fixed_widths == False:
                bamp_chain = flatchain[:, amp_begin+j]
                namp_chain = flatchain[:, amp_begin+len(bamps)+j]

            else:
                bamp_chain = flatchain[:, amp_begin+ampcount]
                namp_chain = flatchain[:, amp_begin+ampcount+1]
                ampcount += 2

            if eline[-1] == '[OIII-]' and line == 'OIII4959':
                print '-> Multiplying the broad and narrow amplitude chains of '+colored(line, 'magenta')+' by '+colored(o3_ratio, 'magenta')
                print
                bamp_chain = np.multiply(bamp_chain, o3_ratio)
                namp_chain = np.multiply(namp_chain, o3_ratio)
                line = 'OIII5007'

            bamp_max, _, _, bamp_sig, pp2 = pf.posterior_gen(bamp_chain, xlabel=line+' Broad Amplitude', axis_sn='x', color='xkcd:tomato red', **kwargs)
            namp_max, _, _, namp_sig, pp2 = pf.posterior_gen(namp_chain, xlabel=line+' Narrow Amplitude', axis_sn='x', color='xkcd:tangerine', **kwargs)

            fit_broad_amps  = np.append(fit_broad_amps, bamp_max)
            fit_narrow_amps = np.append(fit_narrow_amps, namp_max)

            print '-> Converting the width chains from velocity (km/s) to Angstroms using '+colored(line, 'magenta'),
            print ' at '+colored(elines_restframe.loc[line, 'Rest_Lambda'], 'magenta')+' A'
            print
        
            kms_to_a = elines_restframe.loc[line, 'Rest_Lambda'] / c
        
            broad_lum  = np.multiply(np.multiply(bamp_chain, np.multiply(bsig, kms_to_a)), np.sqrt(2. * np.pi))
            narrow_lum = np.multiply(np.multiply(namp_chain, np.multiply(nsig, kms_to_a)), np.sqrt(2. * np.pi))

            tot_lum = np.add(broad_lum, narrow_lum)

            blum_max, _, _, blum_sig, pp2 = pf.posterior_gen(broad_lum, xlabel=line+' Broad Luminosity', axis_sn='x', color='xkcd:dark sky blue', **kwargs)
            nlum_max, _, _, nlum_sig, pp2 = pf.posterior_gen(narrow_lum, xlabel=line+' Narrow Luminosity', axis_sn='x', color='xkcd:steel', **kwargs)

            tot_lum_max, _, _, tot_lum_sig, pp2 = pf.posterior_gen(tot_lum, xlabel=line+' Total Luminosity', axis_sn='x', color='xkcd:light plum', **kwargs)

            added_fit_dict_values = [bamp_max, namp_max, blum_max, blum_sig, nlum_max, nlum_sig, tot_lum_max, tot_lum_sig]

            for idx, key_ending in enumerate(added_BN_key_endings):
                fit_dict[line+key_ending] = np.full(len(comp_bands), np.nan)
                fit_dict[line+key_ending][count] = added_fit_dict_values[idx]
                
            print fstf('The most probable broad amplitude for '+line+' is: '),   colored( snf(bamp_max),'green'), ' +/-', colored( snf(bamp_sig),'green')
            print fstf('The most probable narrow amplitude for '+line+' is: '),  colored( snf(namp_max),'green'), ' +/-', colored( snf(namp_sig),'green')
            print fstf('The most probable broad luminosity for '+line+' is: '),  colored( snf(blum_max),'green'), ' +/-', colored( snf(blum_sig),'green')
            print fstf('The most probable narrow luminosity for '+line+' is: '), colored( snf(nlum_max),'green'), ' +/-', colored( snf(nlum_sig),'green')
            print
            print fstf('The most probable TOTAL luminosity for '+line+' is: '), colored( snf(tot_lum_max),'green'), ' +/-', colored( snf(tot_lum_sig),'green')
            print

        elif fixed_widths == True and amp_comps[ampcount] == 'S':
            samp_chain = flatchain[:, amp_begin+ampcount]
            ampcount += 1

            samp_max, _, _, samp_sig, pp2 = pf.posterior_gen(samp_chain, xlabel=line+' Amplitude', axis_sn='x', color='xkcd:sage', **kwargs)

            fit_single_amps = np.append(fit_single_amps, samp_max)

            print '-> Converting the width chains from velocity (km/s) to Angstroms using '+colored(line, 'magenta'),
            print ' at '+colored(elines_restframe.loc[line, 'Rest_Lambda'], 'magenta')+' A'
            print
        
            kms_to_a = elines_restframe.loc[line, 'Rest_Lambda'] / c

            tot_lum = np.multiply(np.multiply(samp_chain, np.multiply(ssig, kms_to_a)), np.sqrt(2. * np.pi))

            tot_lum_max, _, _, tot_lum_sig, pp2 = pf.posterior_gen(tot_lum, xlabel=line+' Luminosity', axis_sn='x', color='xkcd:light plum', **kwargs)

            added_fit_dict_values = [samp_max, tot_lum_max, tot_lum_sig]

            for idx, key_ending in enumerate(added_S_key_endings):
                fit_dict[line+key_ending] = np.full(len(comp_bands), np.nan)
                fit_dict[line+key_ending][count] = added_fit_dict_values[idx]

            print fstf('The most probable single amplitude for '+line+' is: '), colored( snf(samp_max),'green'), ' +/-', colored( snf(samp_sig),'green')
            print
            print fstf('The most probable TOTAL luminosity for '+line+' is: '), colored( snf(tot_lum_max),'green'), ' +/-', colored( snf(tot_lum_sig),'green')
            print


    if fixed_widths == False:
        broad_bsize  = np.std(flatchain[:,2]) / 20.
        narrow_bsize = np.std(flatchain[:,3]) / 20.
        
        bfig, bax = plt.subplots()
        bax.hist(flatchain[:,2], bins=np.arange(min(flatchain[:,2]), max(flatchain[:,2])+broad_bsize, broad_bsize), color='xkcd:wheat', alpha=0.5, edgecolor='grey')
        bax.minorticks_on()
        bax.tick_params(which='both', left=True, right=True, bottom=True, top=True)
        bax.set_xlabel(r'$\sigma_{\rm B}$ ($\AA$)')
        bax.set_title(norm_eline+'  ---  '+stack_meth+'  ---  '+bands)
        plt.tight_layout()
        plt.savefig('broad_sigma_histogram_'+bands+'-bands_'+norm_eline+'_'+stack_meth+'.pdf')
        plt.close(bfig)

        nfig, nax = plt.subplots()
        nax.hist(flatchain[:,3], bins=np.arange(min(flatchain[:,3]), max(flatchain[:,3])+narrow_bsize, narrow_bsize), color='xkcd:cement', alpha=0.5, edgecolor='grey')
        nax.minorticks_on()
        nax.tick_params(which='both', left=True, right=True, bottom=True, top=True)
        nax.set_xlabel(r'$\sigma_{\rm N}$ ($\AA$)')
        nax.set_title(norm_eline+'  ---  '+stack_meth+'  ---  '+bands)
        plt.tight_layout()
        plt.savefig('narrow_sigma_histogram_'+bands+'-bands_'+norm_eline+'_'+stack_meth+'.pdf')
        plt.close(nfig)
        
        fit_dict['Sigma_Broad'][count], _, _, broad_sigma_sig, pp2   = pf.posterior_gen(flatchain[:,2], xlabel=r'$\sigma_{\rm B}$ ($\AA$)', color='xkcd:wheat', **kwargs)
        fit_dict['Sigma_Narrow'][count], _, _, narrow_sigma_sig, pp2 = pf.posterior_gen(flatchain[:,3], xlabel=r'$\sigma_{\rm N}$ ($\AA$)', color='xkcd:cement', **kwargs)

        fit_dict['Sigma_Single'][count] = np.nan

        print fstf('The most probable broad width for the filter is: '),  colored( ff(fit_dict['Sigma_Broad'][count]),'green'),  ' +/-', colored( ff(broad_sigma_sig),'green')
        print fstf('The most probable narrow width for the filter is: '), colored( ff(fit_dict['Sigma_Narrow'][count]),'green'), ' +/-', colored( ff(narrow_sigma_sig),'green')

    else:
        fit_dict['Sigma_Broad'][count]  = bwidth
        fit_dict['Sigma_Narrow'][count] = nwidth

        print fstf('The fixed broad width for the filter is: '),  colored( ff(bwidth), 'green')
        print fstf('The fixed narrow width for the filter is: '), colored( ff(nwidth), 'green')

        if swidth != None:
            fit_dict['Sigma_Single'][count] = swidth

            print fstf('The fixed single width for the filter is: '), colored( ff(swidth), 'green')

        else:
            fit_dict['Sigma_Single'][count] = np.nan

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
        fit_broad_amps  = np.insert(fit_broad_amps, midx, fit_broad_amps[-1]/o3_ratio)
        fit_narrow_amps = np.insert(fit_narrow_amps, midx, fit_narrow_amps[-1]/o3_ratio)
        
    elif np.any(eline == '[NII-]') and np.any(eline == '[NII+]'):
        fit_single_amps = np.insert(fit_single_amps, midx - len(two_comp_lines), fit_single_amps[-1]/n2_ratio)

    elif eline[-1] == '[OIII-]':
        fit_broad_amps[-1]  /= o3_ratio
        fit_narrow_amps[-1] /= o3_ratio

    if fixed_widths == False:
        all_fit_amps = list(fit_broad_amps) + list(fit_narrow_amps)
        
        broad_comp   = G1model(wavelengths, fit_dict['Slope'][count], fit_dict['y-Int'][count], fit_dict['Sigma_Broad'][count], *fit_broad_amps)
        narrow_comp  = G1model(wavelengths, fit_dict['Slope'][count], fit_dict['y-Int'][count], fit_dict['Sigma_Narrow'][count], *fit_narrow_amps)
        total_model  = G2model(wavelengths, fit_dict['Slope'][count], fit_dict['y-Int'][count], fit_dict['Sigma_Broad'][count], fit_dict['Sigma_Narrow'][count], *all_fit_amps)

    else:
        all_fit_amps = list(fit_broad_amps) + list(fit_narrow_amps) + list(fit_single_amps)
        
        broad_comp   = G1model_bro(wavelengths, fit_dict['Slope'][count], fit_dict['y-Int'][count], *fit_broad_amps)
        narrow_comp  = G1model_nar(wavelengths, fit_dict['Slope'][count], fit_dict['y-Int'][count], *fit_narrow_amps)
        total_model  = G2model(wavelengths, fit_dict['Slope'][count], fit_dict['y-Int'][count], *all_fit_amps)

    plot_model(wavelengths, luminosities, lum_errors, broad_comp, narrow_comp, total_model, rest_wave, which='Final', pp=pp3, **model_kwargs)
    

pp3.close()



cols = [fits.Column(name='Filters', format='3A', array=fit_dict['Filters']), \
        fits.Column(name='Stacking_Method', format='15A', array=fit_dict['Stacking_Method']), \
        fits.Column(name='Norm_Eline', format='10A', array=fit_dict['Norm_Eline']) \
       ]

for key in fit_dict.keys()[3:]:
    cols = cols + [fits.Column(name=key, format='D', array=fit_dict[key])]

hdu = fits.BinTableHDU.from_columns(cols)
hdu.writeto('two_gaussian_fit_parameters_stacked_spectra_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_descr+'.fits', overwrite=True)

print '-> '+colored('two_gaussian_fit_parameters_stacked_spectra_'+stack_meth+'_'+norm_eline+'_no_offset_'+run_descr+'.fits', 'green')+' written!'
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print
