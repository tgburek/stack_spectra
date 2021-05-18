#! /usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sns_setstyle
import fits_readin as fr
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import simps
from scipy.interpolate import interp1d
from spectres import spectres
from termcolor import colored
from matplotlib.backends.backend_pdf import PdfPages
from plotting_functions import Confidence_Interval, conf_int


cB_br={}
cB_br['Ha_Hb']={}
cB_br['Hg_Hb']={}

cB_br['Ha_Hb']['10000']=2.86
cB_br['Ha_Hb']['12500']=2.82
cB_br['Ha_Hb']['15000']=2.79
cB_br['Ha_Hb']['20000']=2.75

cB_br['Hg_Hb']['10000']=0.468
cB_br['Hg_Hb']['12500']=0.471
cB_br['Hg_Hb']['15000']=0.473
cB_br['Hg_Hb']['20000']=0.475

## HELPER FUNCTIONS

def sig_x_over_y(x, y, xerr, yerr):
    sigx2 = np.square(xerr)
    sigy2 = np.square(yerr)

    first_pdiv2  = np.square(np.divide(1.,y))
    second_pdiv2 = np.square(np.divide(x,np.square(y)))

    first_tot_term  = np.multiply(first_pdiv2, sigx2)
    second_tot_term = np.multiply(second_pdiv2, sigy2)

    sigma_result = np.sqrt(np.add(first_tot_term, second_tot_term))

    return sigma_result



def sig_x_times_y(x, y, xerr, yerr):
    x2, sigx2 = np.square(x), np.square(xerr)
    y2, sigy2 = np.square(y), np.square(yerr)

    first_tot_term  = np.multiply(y2, sigx2)
    second_tot_term = np.multiply(x2, sigy2)

    sigma_result = np.sqrt(np.add(first_tot_term, second_tot_term))

    return sigma_result



def add_in_quad(x1, x2):
    x1_sq = np.square(x1)
    x2_sq = np.square(x2)

    sigma_result = np.sqrt(np.add(x1_sq, x2_sq))

    return sigma_result



def gaussian(x, mu, std):   #Creating a normalized Gaussian distribution
    norm_term    = 1. / (std * np.sqrt(2.*np.pi))
    exp_sq_term  = np.square(np.divide(np.subtract(x, mu), std))
    exp_term_tot = np.exp(np.multiply(exp_sq_term, -0.5))

    gauss = np.multiply(exp_term_tot, norm_term)
    
    return gauss


def normalizePD(pd, ind_var):
    area_under_curve = simps(pd, ind_var)

    pdf = np.divide(pd, area_under_curve)

    return pdf


def AV(obs_int_ratio, al_av_bl, al_av):
    numerator   = np.multiply(np.log10(obs_int_ratio), 2.5)
    denominator = al_av_bl - al_av

    A_V = np.divide(numerator, denominator)

    return A_V


##Cardelli (1989) extinction curve definition with preset R_V = 3.1
def cardelli(wavelengths, av=-1., rv=3.1, verbose=False):
    wave_in_inv_microns = np.divide(1.,np.multiply(wavelengths,10.**-4))
    y = np.subtract(wave_in_inv_microns, 1.82)

    a = lambda y: 1. + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b = lambda y: 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    
    ax = a(y)
    bx = b(y)

    al_av = np.add(ax, np.divide(bx,rv))

    al = np.multiply(al_av, av)
    
    if verbose == True:
        print 'Wavelengths: ', colored(wavelengths, 'green')
        print 'A(V): ', colored('%.4f' % av, 'green')
        print 'R(V): ', colored(rv, 'green')
        print 'A_lambda: ', colored(al, 'green', attrs=['bold'])

    return al


## STACKING FUNCTIONS

def shift_to_obs_frame(rest_wavelengths, redshift=0.):
    print colored('-> ','magenta')+'Shifting from rest wavelengths to observed wavelengths...'
    print 'Object lies at z = '+colored(redshift,'green')
    print

    obs_wavelengths = np.multiply(rest_wavelengths, 1.+redshift)

    print 'Observed-wavelengths calculated.'
    print

    return obs_wavelengths



def shift_to_rest_frame(obs_wavelengths, redshift=0.):
    print colored('-> ','magenta')+'Shifting from observed wavelengths to rest wavelengths...'
    print 'Object lies at z = '+colored(redshift,'green')
    print

    rest_wavelengths = np.divide(obs_wavelengths, 1.+redshift)

    print 'Rest-wavelengths calculated.'
    print

    return rest_wavelengths



def Flux_to_Lum(measurements, measurement_errs, redshift=0., H0=70., Om0=0.3, Ob0=0.05, Tcmb0=2.725, Lum_to_Flux=False, densities=False, verbose=False):

    ##CODE SHOULD BE UPDATED TO HANDLE FREQUENCY DENSITIES
    
    if Lum_to_Flux == False:
        print colored('-> ','magenta')+'Converting from flux to luminosity...'
    else:
        print colored('-> ','magenta')+'Converting from luminosity to flux...'

    if densities == False:
        print 'Luminosities and fluxes here are '+colored('bolometric','magenta')
    else:
        print 'Luminosities and fluxes here are '+colored('densities per unit WAVELENGTH','magenta')
        
    print

    cosmology = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, Tcmb0=Tcmb0)

    if verbose == True:
        print 'Object lies at z = '+colored(redshift,'green')
        print
        print 'Cosmological parameters (z=0) to be used in luminosity distance calculations:'
        print 'H = '+colored(cosmology.H0,'green')
        print 'O(matter) = '+colored(cosmology.Om0,'green')
        print 'O(darkenergy) = '+colored(cosmology.Ode0,'green')
        print 'O(baryon) = '+colored(cosmology.Ob0,'green')
        print 'T(CMB) = '+colored(cosmology.Tcmb0,'green')
        print

    
    flux_to_lum = lambda flux, lum_dist: np.multiply(np.multiply(flux, np.square(lum_dist)), 4.*np.pi)  ## bolometric fluxes and luminosities
    lum_to_flux = lambda lum, lum_dist:  np.divide(np.divide(lum, np.square(lum_dist)), 4.*np.pi)
        
    cm_in_Mpc = 3.086e24

    obj_lum_dist_Mpc = cosmology.luminosity_distance(redshift).value
    obj_lum_dist_cm  = obj_lum_dist_Mpc * cm_in_Mpc

    print 'The luminosity distance to the galaxy (in Mpc): '+colored(obj_lum_dist_Mpc,'green')
    print 'The luminosity distance to the galaxy (in cm): '+colored(obj_lum_dist_cm,'green')
    print

    ## The (1+redshift) factor accounts for the fact that flux and luminosity are densities per unit WAVELENGTH
    ## The (1+redshift) factor would not be applied to bolometric luminosities and fluxes
    ## The (1+redshift) factor would be in the denominator / numerator respectively if densities were per unit frequency
    ## See Hogg+2002 K-correction paper

    if Lum_to_Flux == False:
        converted_meas = flux_to_lum(measurements, obj_lum_dist_cm)
        conv_meas_errs = flux_to_lum(measurement_errs, obj_lum_dist_cm)

        if densities == True:
            converted_meas = np.multiply(converted_meas, 1.+redshift)
            conv_meas_errs = np.multiply(conv_meas_errs, 1.+redshift)

    else:
        converted_meas = lum_to_flux(measurements, obj_lum_dist_cm)
        conv_meas_errs = lum_to_flux(measurement_errs , obj_lum_dist_cm)

        if densities == True:
            converted_meas = np.divide(converted_meas, 1.+redshift)
            conv_meas_errs = np.divide(conv_meas_errs, 1.+redshift)
    
    print 'Conversion complete.'    
    print

    return converted_meas, conv_meas_errs



def cardelli_av_calc(hg=(np.nan,np.nan), hb=(np.nan,np.nan), ha=(np.nan,np.nan), filters=['nan','nan','nan'], sys_err=0., rv=3.1, \
                     Te=10000, ne=100, N=5000, verbose=False, plot=False, id_num='', mask='', stack_meth='', norm_eline='' \
                    ):
    
    ##Cardelli decrement dependent
    
    print colored('-> ','magenta')+'Calculating the A(V) value and its uncertainty for extinction correction...'
    print

    hg_flux, hg_ferr = hg[0], hg[1]
    hb_flux, hb_ferr = hb[0], hb[1]
    ha_flux, ha_ferr = ha[0], ha[1]

    hg_filt, hb_filt, ha_filt = filters[0], filters[1], filters[2]

    Te, ne = str(Te), str(ne)

    hb_ha = 1./cB_br['Ha_Hb'][Te]
    hg_hb = cB_br['Hg_Hb'][Te]
    hg_ha = hg_hb * hb_ha

    obs_int_sample = np.linspace(1./N, 1.5, N) #Essentially a sample (N long) from ~0 to 1 representing range of obs/int ratio
    
    if verbose == True:
        print 'Assumed extinction curve: '+colored('Cardelli+89','green')
        print 'Assumed R(V) value: '+colored(rv,'green')
        print 'Assumed Te value (K): '+colored(Te,'green')
        print 'Assumed ne value (cm^-3): '+colored(ne,'green')
        print 'Assumed interfilter systematic error (if needed): '+colored(sys_err,'green')
        print
        print 'The assumed Te and ne correspond to an intrinsic Ha/Hb = '+colored(cB_br['Ha_Hb'][Te],'green')
        print 'The assumed Te and ne correspond to an intrinsic Hg/Hb = '+colored(cB_br['Hg_Hb'][Te],'green')
        print
        print 'The assumed Te and ne therefore correspond to an intrinsic Hb/Ha = '+colored('%.3f' % hb_ha,'green')
        print 'The assumed Te and ne therefore correspond to an intrinsic Hg/Ha = '+colored('%.3f' % hg_ha,'green')
        print
        print

    if np.all(np.isnan(hg) == False) and np.all(np.isnan(hb) == False) and np.all(np.isnan(ha) == True):  ## Hg/Hb
        print 'The Balmer decrement to be used for this galaxy: '+colored('H-gamma/H-beta','green')
        print

        labels, colors = ['Hg/Hb'], ['black']

        y    = 1./0.4340459 - 1.82
        y_bl = 1./0.4861321 - 1.82

        obs_int = (hg_flux / hb_flux) / hg_hb

        obs_ratio_stat = sig_x_over_y(hg_flux, hb_flux, hg_ferr, hb_ferr)

        if hg_filt != hb_filt:
            obs_ratio_sys    = sys_err * (hg_flux / hb_flux)
            obs_ratio_toterr = add_in_quad(obs_ratio_stat, obs_ratio_sys)

        else:
            obs_ratio_toterr = obs_ratio_stat
        
        obs_int_sig = obs_ratio_toterr / hg_hb

        obs_int_pdf = gaussian(obs_int_sample, obs_int, obs_int_sig)

        obs_int_arr, obs_int_sig_arr, obs_int_pdf_arr = np.array([obs_int]), np.array([obs_int_sig]), np.array([obs_int_pdf])

    elif np.all(np.isnan(hg) == False) and np.all(np.isnan(hb) == True) and np.all(np.isnan(ha) == False): ## Hg/Ha
        print 'The Balmer decrement to be used for this galaxy: '+colored('H-gamma/H-alpha','green')
        print

        labels, colors = ['Hg/Ha'], ['black']

        y    = 1./0.4340459 - 1.82
        y_bl = 1./0.6562794 - 1.82

        obs_int = (hg_flux / ha_flux) / hg_ha

        obs_ratio_stat = sig_x_over_y(hg_flux, ha_flux, hg_ferr, ha_ferr)

        if (hg_filt == 'Y' and ha_filt == 'H') or (hg_filt == 'J' and ha_filt == 'K'):
            sys_err = add_in_quad(sys_err, sys_err)

        obs_ratio_sys    = sys_err * (hg_flux / ha_flux)
        obs_ratio_toterr = add_in_quad(obs_ratio_stat, obs_ratio_sys)

        obs_int_sig = obs_ratio_toterr / hg_ha

        obs_int_pdf = gaussian(obs_int_sample, obs_int, obs_int_sig)

        obs_int_arr, obs_int_sig_arr, obs_int_pdf_arr = np.array([obs_int]), np.array([obs_int_sig]), np.array([obs_int_pdf])

    elif np.all(np.isnan(hg) == True) and np.all(np.isnan(hb) == False) and np.all(np.isnan(ha) == False): ## Hb/Ha
        print 'The Balmer decrement to be used for this galaxy: '+colored('H-beta/H-alpha','green')
        print

        labels, colors = ['Hb/Ha'], ['black']

        y    = 1./0.4861321 - 1.82
        y_bl = 1./0.6562794 - 1.82

        obs_int = (hb_flux / ha_flux) / hb_ha

        obs_ratio_stat   = sig_x_over_y(hb_flux, ha_flux, hb_ferr, ha_ferr)
        obs_ratio_sys    = sys_err * (hb_flux / ha_flux)
        obs_ratio_toterr = add_in_quad(obs_ratio_stat, obs_ratio_sys)
        
        obs_int_sig = obs_ratio_toterr / hb_ha

        obs_int_pdf = gaussian(obs_int_sample, obs_int, obs_int_sig)

        obs_int_arr, obs_int_sig_arr, obs_int_pdf_arr = np.array([obs_int]), np.array([obs_int_sig]), np.array([obs_int_pdf])
        
    elif np.all(np.isnan(hg) == False) and np.all(np.isnan(hb) == False) and np.all(np.isnan(ha) == False): ## Hg/Ha and Hb/Ha
        print 'The Balmer decrements to be used for this galaxy: '+colored('H-gamma/H-alpha','green')+' and '+colored('H-beta/H-alpha','green')
        print

        labels, colors = ['Hg/Ha', 'Hb/Ha'], ['black', 'red']

        y    = np.array([1./0.4340459-1.82, 1./0.4861321-1.82])
        y_bl = 1./0.6562794 - 1.82
        
        hg_obs_int = (hg_flux / ha_flux) / hg_ha
        hb_obs_int = (hb_flux / ha_flux) / hb_ha

        hg_obs_stat = sig_x_over_y(hg_flux, ha_flux, hg_ferr, ha_ferr)
        hb_obs_stat = sig_x_over_y(hb_flux, ha_flux, hb_ferr, ha_ferr)
        hb_obs_sys  = sys_err * (hb_flux / ha_flux)
        
        if (hg_filt == 'Y' and ha_filt == 'H') or (hg_filt == 'J' and ha_filt == 'K'):
            sys_err = add_in_quad(sys_err, sys_err)

        hg_obs_sys  = sys_err * (hg_flux / ha_flux)

        hg_obs_toterr = add_in_quad(hg_obs_stat, hg_obs_sys)
        hb_obs_toterr = add_in_quad(hb_obs_stat, hb_obs_sys)

        hg_obs_int_sig = hg_obs_toterr / hg_ha #Dividing the error in (Hg/Ha)_obs by the constant (Hg/Ha)_int
        hb_obs_int_sig = hb_obs_toterr / hb_ha

        hg_obs_int_pdf = gaussian(obs_int_sample, hg_obs_int, hg_obs_int_sig) #Creating Gaussian PDF of (Hg/Ha)_obs/(Hg/Ha)_int
        hb_obs_int_pdf = gaussian(obs_int_sample, hb_obs_int, hb_obs_int_sig)

        obs_int_arr     = np.array([hg_obs_int, hb_obs_int])
        obs_int_sig_arr = np.array([hg_obs_int_sig, hb_obs_int_sig])
        obs_int_pdf_arr = np.array([hg_obs_int_pdf, hb_obs_int_pdf])
        
    else:
        raise KeyError('Either only 1 line was provided or no lines were provided to the function call')


    print 'Values for the Balmer decrement(s) used - if multiple decrements are used, the first values displayed correspond to Hg/Ha...'
    print 'Observed Balmer ratio / Instrinsic Balmer ratio: ', colored(np.array([obs_int_arr,obs_int_sig_arr]).T, 'green')
    print
    print 'The gaussian(s) describing this (these) values and uncertainties have been calculated and normalized...'
    print 'Area under PDF(s): ', colored(simps(obs_int_pdf_arr, obs_int_sample), 'green') #Area under the curve is 1
    print

    
    if plot == True:

        fig, axes = plt.subplots()
        for i in range(obs_int_pdf_arr.shape[0]):
            axes.plot(obs_int_sample, obs_int_pdf_arr[i], color=colors[i], label=labels[i])
            axes.axvline(x = obs_int_sample[obs_int_pdf_arr[i] == np.amax(obs_int_pdf_arr[i])][0], color='black', linestyle='--',linewidth=0.7, \
                         label = '%.4f' % obs_int_sample[obs_int_pdf_arr[i] == np.amax(obs_int_pdf_arr[i])][0] \
                        )
            
        axes.minorticks_on()
        axes.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        axes.legend(loc='upper left', fontsize='x-small', fancybox=True, frameon=True, framealpha=0.8, edgecolor='black')
        axes.set_xlabel('Observed Balmer Ratio / Intrinsic Balmer Ratio')
        axes.set_ylabel('Normalized P(ratio)')
        axes.set_title(id_num+' in '+mask)
        #pp1.savefig()
        fig.savefig('obs_over_int_Bratio_'+stack_meth+'_'+norm_eline+'_no_offset_fw_full_spectrum_'+Te+'K.pdf')
        plt.close(fig)
        

    print 'Calculating the probability distribution of A(V) for this object...'
    print
        
    a = lambda y: 1. + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b = lambda y: 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

    a_bl    = a(y_bl)         ##Calculating a with baseline y-value
    b_bl    = b(y_bl)         ##Caculating b with baseline y-value
    a_av_bl = a_bl + b_bl/rv  ##Calculating A_lambda/A_V with baseline a and b values and R_V=3.1
    
    if np.all(np.isnan(hg+hb+ha) == False):

        if y[0] != 1./0.4340459-1.82 or y[1] != 1./0.4861321-1.82:
            raise ValueError('All calculations for objects using H-gamma, H-beta, and H-alpha must first use H-gamma, then H-beta')

        if plot == True:
            fig, axes = plt.subplots()
        
        av_prob   = np.array([ np.zeros((3,N)), np.zeros((3,N)) ])
        av_common = np.linspace(-0.5, 2.5, N) #Range of possible A_V values given measurement uncertainties
        
        for i in range(2):
            ax = a(y[i])
            bx = b(y[i])
            al_av = ax + bx/rv
            
            av_prob[i][0] = AV(obs_int_sample[::-1], a_av_bl, al_av)  #Calculating A_V over the range of possible ratio values
            av_prob[i][1] = obs_int_pdf_arr[i][::-1]                  #Probability distribution of ratio values
            av_prob[i][1] = normalizePD(av_prob[i][1], av_prob[i][0]) #Normalizing - Returns PDF

            interp_av = interp1d(av_prob[i][0], av_prob[i][1])

            av_prob[i][2] = interp_av(av_common)  #Interpolating over a shared range of A_V values (same A_V values for Hg and Hb)

            CI = Confidence_Interval(*conf_int(av_common, av_prob[i][2], 68))
            
            stf = '{: <54}'.format
            print stf('Area under '+labels[i]+' A(V) PDF - interpolated: '), colored('%.4f' % simps(av_prob[i][2], av_common), 'green')
            print stf('Most probable '+labels[i]+' A(V) value: '), colored('%.4f' % CI.most_probable_value,'green')
            print stf('Approximate 1-sigma uncertainty in '+labels[i]+' A(V) value: '),colored('%.4f' % CI.approximate_sigma,'green')

            if plot == True:

                axes.plot(av_prob[i][0], av_prob[i][1], color='cyan', linewidth=0.5, zorder=10)
                axes.plot(av_common, av_prob[i][2], color=colors[i], linewidth=2., zorder=0, label=labels[i])
                axes.axvline(x = CI.most_probable_value, color='xkcd:gunmetal', linestyle='--', linewidth=0.7, alpha=0.7, label='%.4f' % CI.most_probable_value)

        
        #Multiply the distributions to get a single posterior from which the A_V can be calculated
        final_av_prob = np.multiply(av_prob[0][2], av_prob[1][2])      
        final_av_prob = normalizePD(final_av_prob, av_common)
    
        CI = Confidence_Interval(*conf_int(av_common, final_av_prob, 68))
        
        if plot == True:

            axes.plot(av_common, final_av_prob, color='orange', zorder=20, label='Multiplied A(V) PDFs')
            axes.axvline(x = CI.most_probable_value, linewidth=0.7, color='black', label=str('%.4f' % CI.most_probable_value)+' +/- '+str('%.4f' % CI.approximate_sigma))

            axes.set_xlim([min(av_common), max(av_common)])
            axes.set_ylim([-0.00001, max(final_av_prob)*1.1])
            axes.minorticks_on()
            axes.tick_params(which='both', bottom=True, top=True, left=True, right=True)
            axes.legend(loc='upper right', fontsize='x-small', frameon=True, fancybox=True, framealpha=0.8, edgecolor='black')
            axes.set_xlabel(r'$\rm A_V$')
            axes.set_ylabel(r'Normalized $\rm P(A_V)$')
            axes.set_title(id_num+' in '+mask)
            #pp2.savefig()
            fig.savefig('AV_dist_ind_Bratios_and_combined_'+stack_meth+'_'+norm_eline+'_no_offset_fw_full_spectrum_'+Te+'K.pdf')
            plt.close(fig)

        print

    else:
        ax    = a(y)
        bx    = b(y)
        al_av = ax + bx/rv

        av_common = AV(obs_int_sample[::-1], a_av_bl, al_av)

        final_av_prob = obs_int_pdf[::-1]
        final_av_prob = normalizePD(final_av_prob, av_common)


    CI = Confidence_Interval(*conf_int(av_common, final_av_prob, 68))

    stf = '{: <54}'.format
    print stf('Area under resultant single A(V) PDF: '), colored('%.4f' % simps(final_av_prob, av_common), 'green')
    print stf('Most probable A(V) value overall: '), colored('%.4f' % CI.most_probable_value,'green',attrs=['bold'])
    print stf('Approximate 1-sigma uncertainty in A(V) value: '),colored('%.4f' % CI.approximate_sigma,'green',attrs=['bold'])
    print

    if plot == True:

        fig, axes = plt.subplots()
        axes.plot(av_common, final_av_prob, color='black')
        axes.text(0.79,0.75,'Most prob val: '+str('%.4f' % CI.most_probable_value)+'\nConf int.:   '+str('%.4f' % CI.percentage_of_total_area)+ \
                '%\nConf. Min:   '+str('%.4f' % CI.range_minimum)+'\nConf. Max:   '+str('%.4f' % CI.range_maximum)+'\nSigma:   '+str('%.4f' % CI.approximate_sigma), \
                transform=axes.transAxes,fontsize='x-small')
        axes.fill_between(av_common, 0, final_av_prob, alpha=0.3, facecolor='xkcd:gunmetal', linewidth=1)
        axes.fill_between(av_common[CI.range_minimum_idx : CI.range_maximum_idx+1], 0, final_av_prob[CI.range_minimum_idx : CI.range_maximum_idx+1], \
                        alpha=0.5, facecolor='lime', linewidth=1, edgecolor='green')
        axes.axvline(x = CI.most_probable_value, linewidth=0.7, linestyle='--', color='black')
        axes.set_xlim([min(av_common), max(av_common)])
        axes.minorticks_on()
        axes.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        axes.set_xlabel(r'$\rm A_V$')
        axes.set_ylabel(r'Normalized $\rm P(A_V)$')
        axes.set_title(id_num+' in '+mask)
        #pp3.savefig()
        fig.savefig('AV_dist_'+stack_meth+'_'+norm_eline+'_no_offset_fw_full_spectrum_'+Te+'K.pdf')
        plt.close(fig)


    # av=aprox_sig*np.random.randn(N)+av_max ###Generating sample of A_V values from new A_V distribution

    av     = CI.most_probable_value
    av_sig = CI.approximate_sigma

    return av, av_sig, av_common, final_av_prob



def dust_correct(rest_wavelengths, obs_lum, av, rv=3.1, id_num='', mask='', verbose=False, verbose_cardelli=False):
    
    #print colored('-> ','magenta')+'Correcting the observed luminosities for extinction due to dust...'
    #print

    if verbose == True:
        print colored(id_num,'magenta')+' in mask '+colored(mask,'magenta')
        print 'A(V) = '+colored('%.4f' % av,'green')
        print 'R(V) = '+colored(rv,'green')
        print 'Extinction curve: '+colored('Cardelli+89','green')
        print

    # if len(rest_wavelengths) != len(obs_lum):
    #     raise ValueError('The array of rest-frame wavelengths is not of the same length as the array of observed luminosities')

    a_lambdas = cardelli(rest_wavelengths, av=av, rv=rv, verbose=verbose_cardelli)

    intrinsic_lum = np.multiply(obs_lum, np.power(10., np.divide(a_lambdas, 2.5)))

    return intrinsic_lum



def normalize_spectra(int_lum_to_norm, eline, int_eline_lum, int_lum_errs=None, int_eline_lum_err=None):

    print colored('-> ','magenta')+"Normalizing the spectra's intrinsic luminosities by the intrinsic luminosity of the emission line "+colored(eline,'green')+' ...'
    print

    norm_spectra = np.divide(int_lum_to_norm, int_eline_lum)

    if np.all(int_lum_errs != None) and int_eline_lum_err != None:
        norm_spect_err = sig_x_over_y(int_lum_to_norm, int_eline_lum, int_lum_errs, int_eline_lum_err)

        print 'Spectrum normalized and error spectrum propogated'
        print

        return norm_spectra, norm_spect_err

    elif np.all(int_lum_errs == None) and int_eline_lum_err == None:

        print 'Spectrum normalized'
        print

        return norm_spectra

    else:
        raise Exception('Error spectrum or uncertainty of emission-line luminosity used for normalization provided with the other of these two set to None') 
        



def resample_spectra(new_wavelengths, rest_wavelengths, int_luminosities, lum_errors=None, fill=None, verbose=False):

    print colored('-> ','magenta')+'Resampling the spectra according to the aforementioned wavelength range and dispersion for the stack...'
    print
    print colored('-> ','magenta')+'Trimming the ends of the spectrum that fall outside the new wavelength range...'

    outside_range = np.where((rest_wavelengths < new_wavelengths[0]) | (rest_wavelengths > new_wavelengths[-1]))[0]

    rest_wavelengths, int_luminosities = np.delete(rest_wavelengths, outside_range), np.delete(int_luminosities, outside_range)

    if np.all(lum_errors != None):
        lum_errors = np.delete(lum_errors, outside_range)

    print colored('-> ','magenta')+'Resampling the spectra...'

    if np.all(lum_errors != None):
        new_luminosities, new_lum_errors = spectres(new_wavelengths, rest_wavelengths, int_luminosities, spec_errs=lum_errors, fill=fill, verbose=verbose)
        resampled_spectrum = np.array([new_wavelengths, new_luminosities, new_lum_errors]).T

    else:
        new_luminosities = spectres(new_wavelengths, rest_wavelengths, int_luminosities, spec_errs=lum_errors, fill=fill, verbose=verbose)
        resampled_spectrum = np.array([new_wavelengths, new_luminosities]).T

    print
    print colored('-> ','magenta')+'New wavelength, luminosity, and luminosity error (if resampled) arrays will be returned as a 2D array with columns in this respective order.'

    return resampled_spectrum



def combine_spectra(resampled_luminosities, method, resampled_lum_errs=None, axis=0):

    print colored('-> ','magenta')+'Combining all of the resampled spectra into one stack via a(n) '+colored(method,'green')+' ...'
    print

    if method == 'average' or method == 'mean':

        stacked_lums = np.mean(resampled_luminosities, axis=axis)

    elif method == 'median':

        stacked_lums = np.median(resampled_luminosities, axis=axis)

    elif method == 'weighted-average' and np.all(resampled_lum_errs != None):

        weights = np.divide(1., np.square(resampled_lum_errs))
        stacked_lums, sum_of_weights = np.average(resampled_luminosities, axis=axis, weights=weights, returned=True)
        stacked_errs = np.sqrt(np.divide(1., sum_of_weights))

        return stacked_lums, stacked_errs

    else:
        raise Exception('Method not recognized or no error spectrum supplied for a weighted average.')
    

    return stacked_lums



def multiply_stack_by_eline(stacked_luminosities, method, eline, int_eline_lum, comp_errs=None, eline_lum_error=None):

    print colored('-> ','magenta')+"Multiplying the stacked spectrum by the sample's "+colored(method,'green')+' '+colored(eline,'green')+' luminosity...'
    print 'This luminosity is (erg/s): '+colored(int_eline_lum,'green'),

    if method == 'weighted-average' and eline_lum_error != None:
        print ' +/- '+colored(eline_lum_error,'green')
        
    print '\n'

    luminosities = np.multiply(stacked_luminosities, int_eline_lum)

    if method == 'weighted-average' and np.all(comp_errs != None) and eline_lum_error != None:

        luminosity_errors = sig_x_times_y(stacked_luminosities, int_eline_lum, comp_errs, eline_lum_error)

        return luminosities, luminosity_errors

    elif method == 'weighted-average' and (np.all(comp_errs == None) or eline_lum_error == None):
        raise Exception('The method was set to "weighted-average" but either the spectrum error array or emission line uncertainty was not provided')

    return luminosities


