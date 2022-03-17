#! /usr/bin/env python

import csv
import numpy as np
from pyraf import iraf
from termcolor import colored
from collections import OrderedDict


solar_metal = 8.69 #Asplund et al. 2009
solar_err   = 0.05

def temden_result(which_one,o3_ratio=-99.):
    with open('/home/tim/uparm/nertemden.par') as csvfile:
        text=csv.reader(csvfile,delimiter=',')
        text=list(text)
        if text[6][3] != 'INDEF':
            result=float(text[6][3])
        else:
            if which_one == 'density':
                result=10.
                print colored('Density of '+str(result)+' cm^-3 will be used','green')
            elif which_one == 'temperature':
                result=32900./np.log(o3_ratio/7.9)
                print colored('Temperature of '+str(result)+' K will be used (n_e=1)','green')
    return result




def phys_props(ID, density_ratio, o3_ratio, o2, o3m, o3p, n_e_guess=150., verbose=True):
    keys=['ID', 'Density_Ratio', 'T_OIII', 'T_OII', 'n_e', 'SI_Metallicity', 'DI_Metallicity', 'Total_Metallicity', 'Metallicity_Solar']

    int_props = OrderedDict.fromkeys(keys)
    int_props['ID'] = ID
    int_props['Density_Ratio'] = density_ratio

    for key in keys[2:len(keys)]:
        int_props[key] = np.array([])

    print colored('IDENTIFICATION NUMBER: ', 'cyan') + colored(ID, 'white')
    print
    print 'Te([OIII]) temperature estimate(s)'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print

    for i in range(len(o3_ratio)):
        print colored('Iteration = '+str(i+1), 'red', attrs=['bold'])
        iraf.temden('temperature', o3_ratio[i], atom='oxygen', spectrum=3, assume=n_e_guess)
        t_e3_guess = temden_result('temperature', o3_ratio=o3_ratio[i])
        int_props['T_OIII'] = np.append(int_props['T_OIII'], t_e3_guess)
        print

    t = np.multiply(int_props['T_OIII'], 10**-4)

    int_props['T_OII'] = np.multiply(np.subtract(t, np.multiply(np.subtract(t, 1.), 0.3)), 10**4)  ##Campbell et al. (1986)

    print 'ne density estimate(s)'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    for i in range(len(int_props['T_OII'])):
        print colored('Iteration = '+str(i+1), 'red', attrs=['bold'])
        print 'Density Ratio: ', density_ratio[i]
        print 'Te([OII]): ', int_props['T_OII'][i]
        iraf.temden('density', density_ratio[i], atom='oxygen', spectrum=2, transition='J(2,1)/J(3,1)', assume=int_props['T_OII'][i])
        n_e2_guess = temden_result('density')
        int_props['n_e'] = np.append(int_props['n_e'], n_e2_guess)
        print

    t2 = np.multiply(int_props['T_OII'], 10**-4)

    x = np.multiply(np.multiply(int_props['n_e'],np.divide(1.,np.sqrt(t))),10**-4)

    x2 = np.multiply(np.multiply(int_props['n_e'],np.divide(1.,np.sqrt(t2))),10**-4)

    op_hp = np.power(10,np.add(np.add(np.add(np.add(np.add(np.add(np.log10(o2),5.961),np.divide(1.676,t2)),np.multiply(np.log10(t2),-0.4)),np.multiply(t2,-0.034)),np.log10(np.add(np.multiply(x2,1.35),1.))),-12.))

    opp_hp = np.power(10,np.add(np.add(np.add(np.add(np.add(np.log10(np.add(o3m,o3p)),6.2),np.divide(1.251,t)),np.multiply(np.log10(t),-0.55)),np.multiply(t,-0.014)),-12.))

    o_h = np.add(op_hp,opp_hp)

    int_props['SI_Metallicity'] = np.add(np.log10(op_hp), 12.)
    int_props['DI_Metallicity'] = np.add(np.log10(opp_hp), 12.)

    int_props['Total_Metallicity'] = np.add(np.log10(o_h), 12.)
    int_props['Metallicity_Solar'] = np.power(10., np.subtract(int_props['Total_Metallicity'], solar_metal))

    # gt_95=np.where(int_props['Metallicity'] > 9.5)[0]
    # print 'gt_95: ',int_props['Metallicity'][gt_95]
    # print 'len gt_95: ',float(len(gt_95))/float(len(int_props['Metallicity']))*100.
    # for key in int_props.keys():
    #     int_props[key]=np.delete(int_props[key],gt_95)



    if verbose == True:
        stf = '{: <55}'.format
        np.set_printoptions(formatter={'float_kind': '{: .4f}'.format})

        print stf('Te([OIII]) (K) estimates using IRAF.TEMDEN: '),           colored(int_props['T_OIII'], 'green')
        print stf('Te([OII]) (K) estimates using Campbell et al. (1986): '), colored(int_props['T_OII'], 'green')
        print stf('[OII] or [SII] intensity ratios: '),                      colored(int_props['Density_Ratio'], 'green')
        print stf('n_e (cm^-3) estimates using IRAF.TEMDEN: '),              colored(int_props['n_e'], 'green')
        print stf('12+log(O^+/H^+): '),                                      colored(int_props['SI_Metallicity'], 'green')
        print stf('12+log(O^++/H^+): '),                                     colored(int_props['DI_Metallicity'], 'green')
        print stf('Metallicity (12+log(O/H)): '),                            colored(int_props['Total_Metallicity'], 'green')
        print stf('Metallicity in solar units: '),                           colored(int_props['Metallicity_Solar'], 'green')

        np.set_printoptions(formatter={'float_kind': '{: f}'.format})

    return int_props