#! /usr/bin/env python

#import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import sns_setstyle
from scipy.interpolate import interp1d
from scipy.integrate import simps
from matplotlib.backends.backend_pdf import PdfPages


class Confidence_Interval:

    def __init__(self, most_prob_val, ci_min, ci_min_prob, ci_max, ci_max_prob, ci_min_idx, ci_max_idx, parea):

        self.most_probable_value = most_prob_val
        self.range_minimum = ci_min
        self.range_maximum = ci_max
        self.range_minimum_prob = ci_min_prob
        self.range_maximum_prob = ci_max_prob
        self.range_minimum_idx  = ci_min_idx
        self.range_maximum_idx  = ci_max_idx
        self.percentage_of_total_area = parea

        self.positive_error = self.range_maximum - self.most_probable_value
        self.negative_error = self.range_minimum - self.most_probable_value

        self.approximate_sigma = (self.positive_error - self.negative_error) / 2.


        
def label_adj_exp(value,precision):
    if value == 0 or value == 0.:
        return '0'
    else:
        e=np.log10(np.abs(value))
        m=np.sign(value)*10**(1+e-int(e))
        #return r'${:.{prec}f}E{{{:d}}}$'.format(m, int(e-1),prec=precision)
        return r'${:.{prec}f} \cdot 10^{{{:d}}}$'.format(m, int(e-1),prec=precision)
        


def conf_int(param_array, pdf, percent):
    np_pdf   = np.array(pdf)
    np_param = np.array(param_array)
    percent  = float(percent) / 100.
    
    pmax     = np.amax(np_pdf)
    pmax_idx = int(np.where(np_pdf == pmax)[0])
    tarea    = simps(np_pdf, np_param)
    
    for i in range(1, len(np_pdf)):
        rval_idx = pmax_idx + i
        rval     = np_pdf[rval_idx]
        lval_idx = (np.abs(np_pdf[0:pmax_idx] - rval)).argmin()
        lval     = np_pdf[lval_idx]
        area     = simps(np_pdf[lval_idx:rval_idx+1], np_param[lval_idx:rval_idx+1])

        if area >= percent*tarea:
            conf_int_min = np_param[lval_idx]
            conf_int_max = np_param[rval_idx]
            carea        = (area/tarea) * 100. 
            break
        
    return np_param[pmax_idx], conf_int_min, lval, conf_int_max, rval, lval_idx, rval_idx, carea



def posterior_gen(aoi, title='', xlabel='', ylabel='', std_div=10., color='xkcd:gunmetal', axis_sn=None, plot_cenfunc=False, pp=None):  ##axis_sn 'sn' = scientific notation
    #print aoi
    aoi_std=np.std(aoi)
    bsize=aoi_std/std_div
    
    aoi_val,aoi_bin=np.histogram(aoi,bins=np.arange(min(aoi),max(aoi)+bsize,bsize),density=False)

    aoi_bin_centers=np.add(aoi_bin[0:-1],bsize/2.)
    #mp_aoi=aoi_bin_centers[np.where(aoi_val == max(aoi_val))[0]]
    nbins=len(aoi_val)

    interp=interp1d(aoi_bin_centers,aoi_val)
    xnew=np.linspace(aoi_bin_centers[0],aoi_bin_centers[-1],num=nbins*5+1)
    ynew=interp(xnew)

    aoi_conf=conf_int(xnew,ynew,68)
    aoi_max,aoi_bin_lo,aoi_val_lo,aoi_bin_hi,aoi_val_hi,aoi_lo_idx,aoi_hi_idx,parea = \
        aoi_conf[0],aoi_conf[1],aoi_conf[2],aoi_conf[3],aoi_conf[4],aoi_conf[5],aoi_conf[6],aoi_conf[7]
    perr=aoi_bin_hi-aoi_max
    merr=aoi_bin_lo-aoi_max
    aprox_sig=(perr-merr)/2.
        
    fig,ax=plt.subplots()

    ax.hist(aoi,bins=np.arange(min(aoi),max(aoi)+bsize,bsize),color=color,alpha=0.5,edgecolor='grey')
    ax.plot(xnew,ynew,color='black',linewidth=0.5,label='Linear Interpolation')
    if plot_cenfunc == True:
        ax.axvline(x = np.mean(aoi), color='blue', linewidth=0.5, label='Mean of Data')
        ax.axvline(x = np.median(aoi), color='magenta', linewidth=0.5, label='Median of Data')
    #print '\naoi_max: ',aoi_max,'\nparea: ',parea,'\naoi_bin_lo: ',aoi_bin_lo,'\naoi_bin_hi: ',aoi_bin_hi,'\nbsize: ', \
    #      bsize,'\naprox_sig: ',aprox_sig,'\n'
    ax.text(0.79,0.67,'Most prob val: '+'%.4e' % float(aoi_max)+'\nConf int.:   '+'%.4f' % float(parea)+ \
            '\nConf. Min:   '+'%.4e' % float(aoi_bin_lo)+'\nConf. Max:   '+'%.4e' % float(aoi_bin_hi)+ \
            '\nBinsize:   '+'%.4e' % float(bsize)+'\nBinsize:   STD / '+str(int(std_div))+ \
            '\nSigma:   '+'%.4e' % float(aprox_sig),transform=ax.transAxes,fontsize='xx-small')
    ax.fill_between(xnew[aoi_lo_idx:aoi_hi_idx+1],0,ynew[aoi_lo_idx:aoi_hi_idx+1],alpha=0.3,facecolor='lime', \
                    linewidth=1,label='Confidence Range')

    if axis_sn is not None:
        formatter=ticker.FuncFormatter(lambda axis, p: label_adj_exp(axis,3))
        if axis_sn == 'x':
            plt.gca().xaxis.set_major_formatter(formatter)
        elif axis_sn == 'y':
            plt.gca().yaxis.set_major_formatter(formatter)
        elif axis_sn == 'both':
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.gca().yaxis.set_major_formatter(formatter)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', left=True, right=True, bottom=True, top=True, labelsize=6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right', fontsize='xx-small', frameon=True, fancybox=True, framealpha=0.8, edgecolor='black')

    if pp is not None:
        pp.savefig()
        #plt.show()
        #time.sleep(30)
    else:
        plt.show()
    
    plt.close(fig)
    
    return aoi_max, perr, merr, aprox_sig, pp
