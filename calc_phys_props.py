#! /usr/bin/env python

import os
import re
import sys
import csv
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sns_setstyle
import fits_readin as fr
import stacking_functions as sf
import plotting_functions as pf
import iraf_phys_props_functions as ippf
from pyraf import iraf
from astropy.io import fits
from collections import OrderedDict
from astropy.stats import sigma_clip
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from termcolor import colored
from matplotlib.backends.backend_pdf import PdfPages
from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentDefaultsHelpFormatter

print

class HelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    pass

parser = ArgumentParser(formatter_class=HelpFormatter, description=(
    
'''Estimate stacked spectrum's dust extinction, strong-line ratios, 
electron temperature, election density, and direct metallicty'''

))

parser.add_argument('Luminosity_Table', \
                    help='The FITS file with the fit emission-line luminosities')

parser.add_argument('Norm_ELine', choices=['OIII5007', 'H-alpha'], \
                    help='The emission line used to normalize the spectra during stacking')

parser.add_argument('Stacking_Method', choices=['median', 'average', 'weighted-average'], \
                    help='The method with which the spectra in the sample were stacked')

parser.add_argument('Uncertainty', choices=['bootstrap', 'statistical'], \
                    help='How the uncertainty spectrum was calculated\n'
                         '(i.e. including cosmic variance or just statistically)')

args = parser.parse_args()

table      = args.Luminosity_Table
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
    

n2cal    = lambda n2: np.add(np.multiply(np.log10(n2), 0.57), 8.90)
o3n2cal  = lambda o3n2: np.add(np.multiply(np.log10(o3n2), -0.32), 8.73)
n2o2cal  = lambda n2o2: np.add(np.multiply(np.log10(n2o2), 0.73), 8.94)
o3po2cal = lambda o3po2: np.add(np.multiply(np.log10(o3po2), -0.4640), 8.3439)
        

# def lognormal(x, mu, sig):
#     fraction = np.divide(1., np.multiply(x, sig * np.sqrt(2.*np.pi)))
#     exponential = np.exp(np.divide(np.square(np.subtract(np.log(x), mu)), -2.*sig**2))

#     distribution = np.multiply(fraction, exponential)

#     return distribution

cwd = os.getcwd()

terminal_only = sys.stdout
logname_base  = cwd+'/logfiles/calculating_phys_props_'+stack_meth+'_'+norm_eline+'_'+uncert+'_no_offset_fw_full_spectrum'
sys.stdout    = Logger(logname=logname_base, mode='w')
# sys.stdout = Logger(logname=cwd+'/logfiles/calculating_phys_props_goods41547', mode='w')

print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print '- '+colored(('This program will take the fit emission-line luminosities and estimate:\n'
                    'Dust Extinction, Strong-Line Ratios, Electron Temperature, Electron Density, Direct Metallicity\n'
                    'THIS CODE IS IN DEVELOPMENT.'
                   ), 'cyan', attrs=['bold']),
print ' -'
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print

print 'Review of arguments given to this script:'
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print
print 'Arguments:'
print '-> Luminosity table: ', colored(table, 'cyan')
print '-> Spectra normalized by: ', colored(norm_eline, 'cyan')
print '-> Stacking method used: ', colored(stack_meth, 'cyan')
print '-> Uncertainty calculation method: ', colored(uncert, 'cyan')
print
print 

print 'The current working directory and path are: '+colored(cwd, 'cyan')
print

# output_path = cwd + '/mosdef_4363_lit_sources/goods41547/'
output_path = cwd + '/' + 'uncertainty_'+uncert+'_fitting_analysis/' + norm_eline + '_norm/'

if os.path.isdir(output_path+'all_temps_dust_plots/') == False:
    os.mkdir(output_path+'all_temps_dust_plots/')
    print 'Created directory: '+colored(output_path+'all_temps_dust_plots/', 'white')


elines_restframe = pd.read_csv('loi.txt', comment='#', delim_whitespace=True, names=['Eline','Eline_SH','Rest_Lambda'], index_col='Eline', \
                               dtype={'Eline': np.string_, 'Eline_SH': np.string_, 'Rest_Lambda': np.float64}, usecols=[0, 1, 2] \
                              )[['Eline_SH', 'Rest_Lambda']]  ## SH = shorthand


print 'The current working directory and path are: '+colored(cwd, 'cyan')
print
print '- '+colored('Table of rest-optical emission lines (not all are present in the stack)', 'magenta', attrs=['bold'])+' -'
print
print elines_restframe
print
print elines_restframe.dtypes
print
print
print


lum_table = fr.rc(output_path + table)


snf = '{: >11.4e}'.format
ff  = '{: >11.4f}'.format
stf = '{: <30}'.format

N = 100000
rv = 3.1
o3_doublet_ratio = 2.98
temperatures = [10000, 12500, 15000, 20000]


lines_in_stack = np.array(list(set([name for name in elines_restframe.index.tolist() for colname in lum_table.columns.names if name in colname])))
lines_in_stack[lines_in_stack == 'OIII5007'] = 'OIII4959'
elines = elines_restframe.loc[lines_in_stack, 'Rest_Lambda'].sort_values(ascending=True).to_frame()

elines['Obs_Luminosities'], elines['Obs_Lum_sigs'] = np.zeros(len(elines)), np.zeros(len(elines))

for line in elines.index.tolist():
    if line != 'OIII4959':
        elines.loc[line, 'Obs_Luminosities'] = lum_table[line+'_Total_Lum'][np.isnan(lum_table[line+'_Total_Lum']) == False][0]
        elines.loc[line, 'Obs_Lum_sigs'] = lum_table[line+'_Total_Lum_sig'][np.isnan(lum_table[line+'_Total_Lum_sig']) == False][0]
    else:
        elines.loc['OIII4959', 'Obs_Luminosities'] = lum_table['OIII5007_Total_Lum'][np.isnan(lum_table['OIII5007_Total_Lum']) == False][0] / o3_doublet_ratio
        elines.loc['OIII4959', 'Obs_Lum_sigs'] = lum_table['OIII5007_Total_Lum_sig'][np.isnan(lum_table['OIII5007_Total_Lum_sig']) == False][0] / o3_doublet_ratio
        

print '- '+colored('Table of rest-optical emission lines present in the stack', 'magenta', attrs=['bold'])+' -'
print
print elines
print
print
print

iraf.stsdas(); iraf.analysis(); iraf.nebular(); print  ##Semi-colons allow multiple operations to be written on one line

hg_tuple = (elines.loc['H-gamma', 'Obs_Luminosities'], elines.loc['H-gamma', 'Obs_Lum_sigs'])
hb_tuple = (elines.loc['H-beta', 'Obs_Luminosities'], elines.loc['H-beta', 'Obs_Lum_sigs'])
ha_tuple = (elines.loc['H-alpha', 'Obs_Luminosities'], elines.loc['H-alpha', 'Obs_Lum_sigs'])

## DETERMINING THE ELECTRON TEMPERATURE OF THE STACK BY ASSUMING DIFFERENT TEMPERATURES AND THEIR BALMER DECREMENTS

N_test = 10000
obs_lum_sample = OrderedDict.fromkeys(['OIII4363', 'OIII4959'])

for line in obs_lum_sample.keys():
    obs_lum_sample[line] = elines.loc[line, 'Obs_Lum_sigs'] * np.random.randn(N_test) + elines.loc[line, 'Obs_Luminosities']

estimated_temps = np.array([])

for temp in temperatures:
    print '-> Calculating the extinction in the visual band, A(V), assuming an electron temperature of T([OIII]) = ' + colored(temp, 'magenta') + ' K'
    print

    # av, av_sig, _, _ = sf.cardelli_av_calc(hg=hg_tuple, hb=hb_tuple, ha=ha_tuple, sys_err=0., rv=rv, Te=temp, ne=100, verbose=True, plot=True, \
    #                                        plot_title='GOODS-S 41547   '+r'$\rm T_e$ = '+str(temp)+' K', opath=output_path+'all_temps_dust_plots/' \
    #                                       )

    av, av_sig, _, _ = sf.cardelli_av_calc(hg=hg_tuple, hb=hb_tuple, ha=ha_tuple, filters=['JH','JH','HK'], sys_err=0., rv=rv, Te=temp, ne=100, verbose=True, plot=True, \
                                           plot_title=stack_meth.capitalize()+'-Stacked'+'   '+norm_eline+'-Normalized'+'   '+uncert.capitalize()+'-Uncertainty'+'   '+r'$\rm T_e$ = '+str(temp)+' K', \
                                           stack_meth=stack_meth, norm_eline=norm_eline, uncertainty=uncert, opath=output_path+'all_temps_dust_plots/' \
                                          )
    print
    print '-> Correcting the observed emission-line luminosities for dust extinction'
    print

    int_lum_colname = 'Int_Luminosities_'+str(temp)+'K'

    elines[int_lum_colname] = sf.dust_correct(elines['Rest_Lambda'].to_numpy(), elines['Obs_Luminosities'].to_numpy(), av, rv=rv, verbose=True, verbose_cardelli=True, \
                                              id_num='Plot', mask=stack_meth+'-Stacked'+'   '+norm_eline+'-Normalized'+'   '+'Te = '+str(temp)+' K' \
                                             )

    av_sample = av_sig * np.random.randn(N_test) + av
    neg_av    = np.where(av_sample < 0.)[0]
    av_sample = np.delete(av_sample, neg_av)

    lum_sample_test_dict = OrderedDict.fromkeys(['OIII4363', 'OIII4959'])

    for line in lum_sample_test_dict.keys():
        lum_sample_test_dict[line] = OrderedDict.fromkeys(['Obs_Lum_Sample', 'Int_Lum_Sample'])
        lum_sample_test_dict[line]['Obs_Lum_Sample'] = np.delete(obs_lum_sample[line], neg_av)
        lum_sample_test_dict[line]['Int_Lum_Sample'] = sf.dust_correct(elines.loc[line, 'Rest_Lambda'], lum_sample_test_dict[line]['Obs_Lum_Sample'], av_sample, rv=rv)

    neg_o3aur = np.where(lum_sample_test_dict['OIII4363']['Int_Lum_Sample'] < 0.)[0]

    for line in lum_sample_test_dict.keys():
        lum_sample_test_dict[line]['Int_Lum_Sample'] = np.delete(lum_sample_test_dict[line]['Int_Lum_Sample'], neg_o3aur)

    print
    print '-> Calculating the electron temperature with IRAF nebular.temden'
    print

    
    o3_ratio = np.divide(np.add(np.multiply(lum_sample_test_dict['OIII4959']['Int_Lum_Sample'], o3_doublet_ratio), lum_sample_test_dict['OIII4959']['Int_Lum_Sample']), \
                         lum_sample_test_dict['OIII4363']['Int_Lum_Sample'] \
                        )

    teO3_sample = np.array([])

    sys.stdout = terminal_only

    for i in range(len(o3_ratio)):
        print colored('Iteration = '+str(i+1), 'red', attrs=['bold'])
        iraf.temden('temperature', o3_ratio[i], atom='oxygen', spectrum=3, assume=150.)
        t_e3_guess = ippf.temden_result('temperature', o3_ratio=o3_ratio[i])
        teO3_sample = np.append(teO3_sample, t_e3_guess)
        print

    sys.stdout = Logger(logname=logname_base, mode='a')

    mp_temp, _, _, _, _ = pf.posterior_gen(teO3_sample, title=stack_meth+'_'+norm_eline+'_'+str(temp), xlabel='Te([OIII])', ylabel='Number of Instances', std_div=5.)

    print 'The most probable Te([OIII]) is: '+colored('%.4f' % mp_temp, 'green')
    print

    nearest_temp_idx = (np.abs(temperatures - mp_temp)).argmin()
    estimated_temps  = np.append(estimated_temps, temperatures[nearest_temp_idx])

    print
    print
    print colored('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', 'white')
    print
    print

if len(set(estimated_temps)) == 1:
    stack_temp = int(estimated_temps[0])
    print colored('The temperature (K) associated with this stack is: ', 'white'), colored(stack_temp, 'green', attrs=['bold'])
    print
else:
    raise ValueError('This stack could have more than one temperature associated with it')

print colored(('Calculating the intrinsic luminosities of the emission lines\n'
               'and the physical properties associated with the stack'), 'cyan')
print
print colored('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', 'white')



lum_sample_dict = OrderedDict.fromkeys(elines.index.tolist())

## TESTING MY (SLIGHTLY-ALTERED) ORIGINAL METHOD OF DUST-CORRECTING THE LINES

# av, av_sig, av_common, av_pdf = sf.cardelli_av_calc(hg=hg_tuple, hb=hb_tuple, ha=ha_tuple, sys_err=0., rv=rv, Te=stack_temp, ne=100, N=N/2, verbose=True, plot=True, \
#                                                     plot_title='GOODS-S 41547   '+r'$\rm T_e$ = '+str(stack_temp)+' K', opath=output_path \
#                                                    )

av, av_sig, av_common, av_pdf = sf.cardelli_av_calc(hg=hg_tuple, hb=hb_tuple, ha=ha_tuple, filters=['JH','JH','HK'], sys_err=0., rv=rv, Te=stack_temp, ne=100, N=N/2, verbose=True, plot=True, \
                                                    plot_title=stack_meth.capitalize()+'-Stacked'+'   '+norm_eline+'-Normalized'+'   '+uncert.capitalize()+'-Uncertainty'+'   '+r'$\rm T_e$ = '+str(stack_temp)+' K', \
                                                    stack_meth=stack_meth, norm_eline=norm_eline, uncertainty=uncert, opath=output_path \
                                                   )

av_sample = np.random.choice(av_common, size=N, replace=True, p=np.divide(av_pdf, np.sum(av_pdf)))
neg_av    = np.where(av_sample < 0.)[0]
av_sample = np.delete(av_sample, neg_av)

print
print 'Removing '+colored(len(neg_av), 'magenta')+' negative A(V) values from sample'
print 'This corresponds to '+colored('%.3f' % (len(neg_av) / float(N) * 100.), 'magenta')+' % of the sample'
print

pp = PdfPages(output_path + 'int_emiss_line_lum_dists_'+stack_meth+'_'+norm_eline+'_'+uncert+'.pdf')

cols = [fits.Column(name='Normalizing_Eline', format='10A', array=np.array([norm_eline])), \
        fits.Column(name='Stacking_Method',   format='16A', array=np.array([stack_meth])), \
        fits.Column(name='AV',                format='D',   array=np.array([av])), \
        fits.Column(name='AV_Sig',            format='D',   array=np.array([av_sig]))]

for line in lum_sample_dict.keys():
    lum_sample_dict[line] = OrderedDict.fromkeys(['Obs_Lum_Sample', 'Int_Lum_Sample', 'Int_Lum_Sample_RelHb'])
    lum_sample_dict[line]['Obs_Lum_Sample'] = elines.loc[line, 'Obs_Lum_sigs'] * np.random.randn(N-len(neg_av)) + elines.loc[line, 'Obs_Luminosities']
    lum_sample_dict[line]['Int_Lum_Sample'] = sf.dust_correct(elines.loc[line, 'Rest_Lambda'], lum_sample_dict[line]['Obs_Lum_Sample'], av_sample, rv=rv)
    
    int_lum_max, int_lum_perr, int_lum_merr, int_lum_sig, pp = pf.posterior_gen(lum_sample_dict[line]['Int_Lum_Sample'], title=stack_meth+'-Stacked   '+norm_eline+'-Normalized', \
                                                                                xlabel=line+' Intrinsic Luminosity', ylabel='Number of Instances', std_div=10., axis_sn='x', pp=pp \
                                                                               )

    cols += [fits.Column(name=line+'_Lum',      format='D', array=np.array([int_lum_max])), fits.Column(name=line+'_Lum_Perr', format='D', array=np.array([int_lum_perr])), \
             fits.Column(name=line+'_Lum_Merr', format='D', array=np.array([int_lum_merr])), fits.Column(name=line+'_Lum_Sig', format='D', array=np.array([int_lum_sig]))]

    print stf('Int. lum. of '+line+': '), colored( snf(int_lum_max), 'green'), ' +/-', colored( snf(int_lum_sig), 'green'),
    print colored( snf(elines.loc[line, 'Int_Luminosities_'+str(stack_temp)+'K']), 'white')

hdu = fits.BinTableHDU.from_columns(cols)
hdu.writeto(output_path + 'Intrinsic_Luminosities_'+stack_meth+'_'+norm_eline+'.fits', overwrite=True)

pp.close()

print
print colored('int_emiss_line_lum_dists_'+stack_meth+'_'+norm_eline+'_'+uncert+'.pdf', 'green')+' written'
print colored('Intrinsic_Luminosities_'+stack_meth+'_'+norm_eline+'.fits', 'green')+' written'
print

## CALCULATING THE STRONG-LINE RATIOS OF INTEREST

pp = PdfPages(output_path + 'strong_line_ratio_dists_'+stack_meth+'_'+norm_eline+'_'+uncert+'.pdf')
slr_kwargs = dict(title=stack_meth+'-stacked   '+norm_eline+'-normalized', ylabel='Number of Instances', pp=pp)

for line in lum_sample_dict.keys():
    lum_sample_dict[line]['Int_Lum_Sample_RelHb'] = np.divide(lum_sample_dict[line]['Int_Lum_Sample'], lum_sample_dict['H-beta']['Int_Lum_Sample'])

o2m = lum_sample_dict['OII3726']['Int_Lum_Sample_RelHb']
o2p = lum_sample_dict['OII3729']['Int_Lum_Sample_RelHb']
o3  = lum_sample_dict['OIII4363']['Int_Lum_Sample_RelHb']
o3m = lum_sample_dict['OIII4959']['Int_Lum_Sample_RelHb']
o3p = np.multiply(o3m, o3_doublet_ratio)

o2_sum   = np.add(o2m, o2p)
o2_ratio = np.divide(o2p, o2m)
o3_ratio = np.divide(np.add(o3m, o3p), o3)
o3po2    = np.divide(o3p, o2_sum)
o32      = np.divide(np.add(o3m, o3p), o2_sum)
r23      = np.add(np.add(o3m, o3p), o2_sum)

o2hb_max, o2hb_perr, o2hb_merr, o2hb_sig, pp     = pf.posterior_gen(np.log10(o2_sum), xlabel=r'log([OII]3726,3729 / H$\beta$)', **slr_kwargs)
o2rat_max, o2rat_perr, o2rat_merr, o2rat_sig, pp = pf.posterior_gen(o2_ratio, xlabel='[OII]3729 / [OII]3726', **slr_kwargs)
o3p_max, o3p_perr, o3p_merr, o3p_sig, pp         = pf.posterior_gen(np.log10(o3p), xlabel=r'log([OIII]5007 / H$\beta$)', **slr_kwargs)
o3po2_max, o3po2_perr, o3po2_merr, o3po2_sig, pp = pf.posterior_gen(np.log10(o3po2), xlabel='log([OIII]5007 / [OII]3726,3729)', **slr_kwargs)
o32_max, o32_perr, o32_merr, o32_sig, pp         = pf.posterior_gen(np.log10(o32), xlabel=r'$\rm log(O_{32})$', **slr_kwargs)
r23_max, r23_perr, r23_merr, r23_sig, pp         = pf.posterior_gen(np.log10(r23), xlabel=r'$\rm log(R_{23})$', **slr_kwargs)

neg_n2 = np.where(lum_sample_dict['NII6583']['Int_Lum_Sample'] < 0.)[0]

print
print 'Removing '+colored(len(neg_n2), 'magenta')+' negative [NII]6583 values from sample'
print 'This corresponds to '+colored('%.3f' % (len(neg_n2) / float(len(lum_sample_dict['NII6583']['Int_Lum_Sample'])) * 100.), 'magenta')+' % of the sample'
print

o2m_cut, o2p_cut = np.delete(o2m, neg_n2), np.delete(o2p, neg_n2)
o3_cut, o3m_cut, o3p_cut = np.delete(o3, neg_n2), np.delete(o3m, neg_n2), np.delete(o3p, neg_n2)
ha_cut = np.delete(lum_sample_dict['H-alpha']['Int_Lum_Sample_RelHb'], neg_n2)
n2_cut = np.delete(lum_sample_dict['NII6583']['Int_Lum_Sample_RelHb'], neg_n2)
o2_sum_cut = np.delete(o2_sum, neg_n2)

o3m_obs, hb_obs = np.delete(lum_sample_dict['OIII4959']['Obs_Lum_Sample'], neg_n2), np.delete(lum_sample_dict['H-beta']['Obs_Lum_Sample'], neg_n2)
n2_obs, ha_obs  = np.delete(lum_sample_dict['NII6583']['Obs_Lum_Sample'], neg_n2), np.delete(lum_sample_dict['H-alpha']['Obs_Lum_Sample'], neg_n2)

n2ha = np.divide(n2_cut, ha_cut)
o3n2 = np.divide(o3p_cut, n2ha)
n2o2 = np.divide(n2_cut, o2_sum_cut)

o3phb_obs = np.divide(np.multiply(o3m_obs, o3_doublet_ratio), hb_obs)
n2ha_obs  = np.divide(n2_obs, ha_obs)

n2ha_max, n2ha_perr, n2ha_merr, n2ha_sig, pp = pf.posterior_gen(np.log10(n2ha), xlabel=r'log([NII]6583 / H$\alpha$)', **slr_kwargs)
n2ha_obs_max, n2ha_obs_perr, n2ha_obs_merr, n2ha_obs_sig, pp     = pf.posterior_gen(np.log10(n2ha_obs), xlabel=r'$\rm log([NII]6583 / H\alpha)_{obs}$', **slr_kwargs)
o3phb_obs_max, o3phb_obs_perr, o3phb_obs_merr, o3phb_obs_sig, pp = pf.posterior_gen(np.log10(o3phb_obs), xlabel=r'$\rm log([OIII]5007 / H\beta)_{obs}$', **slr_kwargs)

slr_dict = OrderedDict.fromkeys(['SLR', 'MP_Val', 'Sigma', 'PErr', 'MErr'])

# slr_dict['SLR']    = np.array(['[OII]3729/[OII]3726','[OII]/Hb','[OIII]5007/Hb','[OIII]5007/[OII]','O32','R23'])
# slr_dict['MP_Val'] = np.array([o2rat_max, o2hb_max, o3p_max, o3po2_max, o32_max, r23_max])
# slr_dict['Sigma']  = np.array([o2rat_sig, o2hb_sig, o3p_sig, o3po2_sig, o32_sig, r23_sig])
# slr_dict['PErr']   = np.array([o2rat_perr, o2hb_perr, o3p_perr, o3po2_perr, o32_perr, r23_perr])
# slr_dict['MErr']   = np.array([o2rat_merr, o2hb_merr, o3p_merr, o3po2_merr, o32_merr, r23_merr])

slr_dict['SLR']    = np.array(['[OII]3729/[OII]3726','[OII]/Hb','[OIII]5007/Hb','[OIII]5007/[OII]','O32','R23','[NII]6583/Ha', '[NII]6583/Ha_(obs)', '[OIII]5007/Hb_(obs)'])
slr_dict['MP_Val'] = np.array([o2rat_max, o2hb_max, o3p_max, o3po2_max, o32_max, r23_max, n2ha_max, n2ha_obs_max, o3phb_obs_max])
slr_dict['Sigma']  = np.array([o2rat_sig, o2hb_sig, o3p_sig, o3po2_sig, o32_sig, r23_sig, n2ha_sig, n2ha_obs_sig, o3phb_obs_sig])
slr_dict['PErr']   = np.array([o2rat_perr, o2hb_perr, o3p_perr, o3po2_perr, o32_perr, r23_perr, n2ha_perr, n2ha_obs_perr, o3phb_obs_perr])
slr_dict['MErr']   = np.array([o2rat_merr, o2hb_merr, o3p_merr, o3po2_merr, o32_merr, r23_merr, n2ha_merr, n2ha_obs_merr, o3phb_obs_merr])

slrDF = pd.DataFrame.from_dict(slr_dict, orient='columns').set_index('SLR', drop=True)
slrDF.to_csv(output_path + 'strong_line_ratio_values_'+stack_meth+'_'+norm_eline+'.txt', sep='\t', float_format='%.4f', \
             header=True, index=True, index_label='#Ratio', line_terminator='\n')
    
print colored('strong_line_ratio_dists_'+stack_meth+'_'+norm_eline+'_'+uncert+'.pdf', 'green')+' written'
print colored('strong_line_ratio_values_'+stack_meth+'_'+norm_eline+'.txt', 'green')+' written'
print

pp.close()

## CALCULATING THE ELECTRON TEMPERATURE, ELECTRON DENSITY, DIRECT METALLICITY, AND STRONG-LINE METALLICITIES

neg_o3aur = np.where(lum_sample_dict['OIII4363']['Int_Lum_Sample'] < 0.)[0]

print 'Removing '+colored(len(neg_o3aur), 'magenta')+' negative [OIII]4363 samples'
print 'This corresponds to '+colored('%.3f' % (len(neg_o3aur) / float(len(lum_sample_dict['OIII4363']['Int_Lum_Sample'])) * 100.), 'magenta')+' % of the sample'
print

o2_ratio_cut = np.delete(o2_ratio, neg_o3aur)
o2_sum_cut   = np.delete(o2_sum, neg_o3aur)
o3_ratio_cut = np.delete(o3_ratio, neg_o3aur)
o3m_cut = np.delete(o3m, neg_o3aur)
o3p_cut = np.delete(o3p, neg_o3aur)

print '-> Calculating the electron temperature, electron density, and direct metallicity with IRAF nebular.temden'
print 

sys.stdout = terminal_only

physical_properties = ippf.phys_props(stack_meth+'_'+norm_eline+'_stack', o2_ratio_cut, o3_ratio_cut, o2_sum_cut, o3m_cut, o3p_cut, n_e_guess=150., verbose=False)

sys.stdout = Logger(logname=logname_base, mode='a')

pp = PdfPages(output_path + 'physical_property_dists_'+stack_meth+'_'+norm_eline+'_'+uncert+'.pdf')

phys_prop_kwargs = dict(title=stack_meth+'-stacked   '+norm_eline+'-normalized', ylabel='Number of Instances', pp=pp)

mpto2, _, _, to2_sig, pp = pf.posterior_gen(physical_properties['T_OII'], xlabel=r'$\rm T_e$([OII]) (K)', **phys_prop_kwargs)
mpto3, _, _, to3_sig, pp = pf.posterior_gen(physical_properties['T_OIII'], xlabel=r'$\rm T_e$([OIII]) (K)', **phys_prop_kwargs)
#mpne, _, _, ne_sig, pp  = pf.posterior_gen(physical_properties['n_e'], xlabel=r'$\rm n_e\ (cm^{-3})$', **phys_prop_kwargs)

mp_metal, metal_perr, metal_merr, metal_sig, pp = pf.posterior_gen(physical_properties['Total_Metallicity'], xlabel=r'$\rm 12+log(O/H)_{direct}$', **phys_prop_kwargs)
#mp_metal_sol, _, _, metal_sol_sig, pp = pf.posterior_gen(physical_properties['Metallicity_Solar'], xlabel=r'Z/Z_\odot', **phys_prop_kwargs)

n2_metal_max, n2_metal_perr, n2_metal_merr, n2_metal_sig, pp             = pf.posterior_gen(n2cal(n2ha), xlabel='12+log(O/H) [N2]', **phys_prop_kwargs)
o3n2_metal_max, o3n2_metal_perr, o3n2_metal_merr, o3n2_metal_sig, pp     = pf.posterior_gen(o3n2cal(o3n2), xlabel='12+log(O/H) [O3N2]', **phys_prop_kwargs)
n2o2_metal_max, n2o2_metal_perr, n2o2_metal_merr, n2o2_metal_sig, pp     = pf.posterior_gen(n2o2cal(n2o2), xlabel='12+log(O/H) [N2O2]', **phys_prop_kwargs)
o3po2_metal_max, o3po2_metal_perr, o3po2_metal_merr, o3po2_metal_sig, pp = pf.posterior_gen(o3po2cal(o3po2), xlabel='12+log(O/H) [O32]', **phys_prop_kwargs)


pp.close()

c1  = fits.Column(name='Normalizing_Eline',  format='10A', array=np.array([norm_eline]))
c2  = fits.Column(name='Stacking_Method',    format='16A', array=np.array([stack_meth]))
c3  = fits.Column(name='TeOII',              format='D',   array=np.array([mpto2]))
c4  = fits.Column(name='TeOII_Sig',          format='D',   array=np.array([to2_sig]))
c5  = fits.Column(name='TeOIII',             format='D',   array=np.array([mpto3]))
c6  = fits.Column(name='TeOIII_Sig',         format='D',   array=np.array([to3_sig]))
c7  = fits.Column(name='Direct_Metallicity', format='D',   array=np.array([mp_metal]))
c8  = fits.Column(name='Metallicity_Perr',   format='D',   array=np.array([metal_perr]))
c9  = fits.Column(name='Metallicity_Merr',   format='D',   array=np.array([metal_merr]))
c10 = fits.Column(name='Metallicity_Sig',    format='D',   array=np.array([metal_sig]))

hdu = fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10])
hdu.writeto(output_path + 'physical_properties_'+stack_meth+'_'+norm_eline+'.fits', overwrite=True)

slr_metal_dict = OrderedDict.fromkeys(['Calibration', 'Metallicity', 'Metallicity_Sig', 'Metallicity_PErr', 'Metallicity_MErr'])

slr_metal_dict['Calibration']      = np.array(['N2', 'O3N2', 'N2O2', 'O32'])
slr_metal_dict['Metallicity']      = np.array([n2_metal_max, o3n2_metal_max, n2o2_metal_max, o3po2_metal_max])
slr_metal_dict['Metallicity_Sig']  = np.array([n2_metal_sig, o3n2_metal_sig, n2o2_metal_sig, o3po2_metal_sig])
slr_metal_dict['Metallicity_PErr'] = np.array([n2_metal_perr, o3n2_metal_perr, n2o2_metal_perr, o3po2_metal_perr])
slr_metal_dict['Metallicity_MErr'] = np.array([n2_metal_merr, o3n2_metal_merr, n2o2_metal_merr, o3po2_metal_merr])

slr_metalDF = pd.DataFrame.from_dict(slr_metal_dict, orient='columns').set_index('Calibration', drop=True)
slr_metalDF.to_csv(output_path + 'strong_line_metallicities_'+stack_meth+'_'+norm_eline+'.txt', sep='\t', float_format='%.4f', \
                   header=True, index=True, index_label='#Ratio', line_terminator='\n')

print colored('physical_properties_'+stack_meth+'_'+norm_eline+'_'+uncert+'.pdf', 'green')+' written'
print colored('physical_properties_'+stack_meth+'_'+norm_eline+'.fits', 'green')+' written'
print colored('strong_line_metallicities_'+stack_meth+'_'+norm_eline+'.txt', 'green')+' written'
print
print
print
print

print elines
    
print
print
print
    




