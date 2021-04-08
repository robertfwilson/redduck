import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.ndimage as ndimage
from scipy import signal 
from matplotlib.colors import LogNorm

import pandas as pd

from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from astropy.io import fits
import glob

import warnings
warnings.filterwarnings('ignore')


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *


#from .duckwave import WaveCalSol

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)



class Belly(object):

    def __init__(self, WavelengthSolution, specimg, flatimg, orders, meta=None):

        self.WaveCal = WavelengthSolution
        self.flat = flatimg
        self.img = specimg
        self.orders=orders
        self.meta = meta

        self.spict = {}


    def _extract_spectrum(self, order):

         xtrace, ytrace = get_order_trace(self.flat, order_num=order, dx=2, dy=4)
         flux, _ = extract_order(self.img, xtrace, ytrace, width=4.,
                                 do_weighted_extraction=False,plot=False)

         return flux


    def _get_trace(self,order):
         
         xtrace, ytrace = get_order_trace(self.flat, order_num=order, dx=2, dy=4)
         return xtrace,ytrace

     
    def _get_flatflux(self,order):

         xtrace, ytrace = get_order_trace(self.flat, order_num=order, dx=2, dy=4)

         flat_flux, _ = extract_order(self.flat, xtrace, ytrace, width=4.,
                                      do_weighted_extraction=False,plot=False)
         return flat_flux


    def get_spectrum(self, fname):


        return 1.
         

     
    def save_spectrum(self, fname):

         for o in self.orders:
             flux = self._extract_spectrum(o)
             wave = self.WaveCal.get_wavelengths(o,flux)

             flat_flux = self._get_flatflux(o)

             save_order(fname, wave=wave, flux=flux, flat=flat_flux, order_num=o)

             print('{}/{} orders done'.format(o+1, max(self.orders)), end='\r' )






        
    
        


def save_order(object_name, wave, flux, order_num, flat=None):

    save_df = pd.DataFrame({'wave':wave, 'flux':flux, 'flat':flat})
    
    try:
        save_df.to_csv(object_name+'_Spec/'+object_name+'_order{:02d}'.format(order_num)+'.txt', index=False)
    except FileNotFoundError:
        os.mkdir(object_name+'_Spec')
        save_df.to_csv(object_name+'_Spec/'+object_name+'_order{:02d}'.format(order_num)+'.txt',index=False)
    




def get_order_ypix(order_num, instrument='arces'):

    orders = pd.read_csv(dir_path+'/'+instrument+'/orders.txt')

    center_pix = orders['center_ypix'].loc[orders['order']==order_num]
    return float(center_pix)
    




def get_order_trace(masterflat, order_num, dx=2, dy=4):

    order_ypix = get_order_ypix(order_num)
    xtrace, ytrace  = trace_order(masterflat, 1000, order_ypix, dx=2, dy=4 )

    ytrace_fit = fit_traced_order(xtrace,ytrace)

    
    return xtrace, ytrace_fit
    





def get_center_orders(data, top_row = 250, bottom_row = 1600, dx=10, sigma=3, peak_dist=5):
    
    dSlice = np.sum(data[top_row:bottom_row, 1000-dx:1000+dx], axis=1)

    filtered = ndimage.gaussian_filter(dSlice, sigma)
    peak_rows , peak_row_info = signal.find_peaks(filtered, distance=peak_dist)

    return peak_rows+top_row, dSlice


def trace_order(data,xmid,ymid, dx=2, dy=4, edge_cut=100.):
    
    x_new = xmid
    y_add = ymid
    
    x_trace = []
    y_trace = []
    
    smooth_data = ndimage.gaussian_filter(data, (dx,dy))
    while x_new<len(data[0,:])-edge_cut:
        x_new+=dx
        
        y_add += np.argmax(np.sum(smooth_data[int(y_add-dy):int(y_add+dy+1),x_new:x_new+2*dx],axis=1) ) - dy
        
        y_trace.append(y_add)
        x_trace.append(x_new)
    
    x_new = xmid
    y_add = ymid
    while x_new>edge_cut:
        x_new-=dx
        
        y_add += np.argmax(np.sum(smooth_data[int(y_add-dy):int(y_add+dy+1),x_new:x_new+2*dx],axis=1) ) - dy
        
        y_trace.append(y_add)
        x_trace.append(x_new)

    x_trace_sorted = np.sort(x_trace)
    y_trace_sorted = np.array(y_trace)[np.argsort(x_trace)]
    
    return x_trace_sorted, y_trace_sorted






def fit_traced_order(xtrace, ytrace, niter=3, poly_order=4):
        
    # initialize a linear fitter
    fit = fitting.LinearLSQFitter()

    # initialize the outlier removal fitter
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=3.0)
    
    # initialize model
    init_mod = models.Polynomial1D(degree=poly_order)
    
    fit_mod, mask = or_fit(init_mod, xtrace, ytrace,)

    
    return fit_mod(np.arange(min(xtrace), max(xtrace)))



def gauss1d(x, loc, sigma):
    return np.exp( - 0.5 * (x-loc)**2. / (sigma)**2.)



def weighted_extraction(spec2d, ytrace):
    
    width = spec2d.shape[1]
    v_distance = np.arange(width) - width/2.

    ytrace_remainder = ytrace%1.
    
    spec = []
    for i in range(len(ytrace)):
        weights = gauss1d(v_distance, loc=ytrace_remainder[i], sigma=width/4.)
        
        spec.append( np.sum(spec2d[i,:] * weights)  / np.sum(weights) )
    
    return np.array(spec)
    




def extract_order(spec_img, xtrace, ytrace, width=5, do_weighted_extraction=False, plot=False):
    
    spec_2d = []
    xtrace = np.arange(min(xtrace), max(xtrace))
    
    for i,x in enumerate(xtrace):
        
        yt_upper = int(np.ceil(ytrace[i]+width))
        yt_lower = int(np.floor(ytrace[i]-width))
        
        spec_2d.append(spec_img[yt_lower:yt_upper,x] ) 
    
    spec2d = np.array(spec_2d)
    
    weighted_spec = weighted_extraction(spec2d, ytrace)
    unweighted_spec = np.sum(spec2d, axis=1)
        
    if plot:
        
        f, (ax1,ax2) = plt.subplots(2,1,figsize=(10,4))
        
        m = np.median(spec_img[int(min(ytrace)-10):int(max(ytrace)+10),:])
        s = np.std(spec_img[int(min(ytrace)-10):int(max(ytrace)+10),:])
        
        ax1.pcolormesh(spec_img, vmin=m-s, vmax=m+3*s, cmap='Greys_r')
        ax1.plot(xtrace, ytrace+width, 'r--')
        ax1.plot(xtrace, ytrace-width, 'r--')

        ax1.set_ylim(min(ytrace)-10, max(ytrace)+10)
        ax2.plot(xtrace, unweighted_spec/np.median(unweighted_spec), label='unweighted')
        ax2.plot(xtrace, weighted_spec/np.median(weighted_spec), '--',label='weighted')
        ax2.legend()
        
        plt.show()
    
    return np.array(unweighted_spec), np.array(weighted_spec)
        
        


        
def extract_all_orders(data, peak_rows, flat, specwidth=4, edge_cut=100, do_weighted_extraction=False, plot=False):
    
    all_spectra = []
    
    x_traces = []
    y_traces = []
    
    for i,row in enumerate(peak_rows):
    
        xt, yt = trace_order(flat, xmid=1000, ymid=row, dx=1, dy=specwidth)
        yt_fit = fit_traced_order(xt,yt)
        
        x_traces.append(xt)
        y_traces.append(yt_fit)

        flat_order = extract_order(flat, xt, yt_fit, width=specwidth, 
                                   do_weighted_extraction=do_weighted_extraction)
        
        spec_order = extract_order(data, xt, yt_fit, width=specwidth, 
                                   do_weighted_extraction=do_weighted_extraction,
                                  plot=False)
    
        deblazed_order = spec_order/flat_order
        
        if plot:
            
            f, (ax1,ax2) = plt.subplots(2,1,figsize=(10,4))
        
            m = np.median(data[int(min(yt_fit)-10):int(max(yt_fit)+10),:]), 
            s = np.std(data[int(min(yt_fit)-10):int(max(yt_fit)+10),:])

            
            ax1.set_title('order {}'.format(i))
            
            ax1.pcolormesh(data, vmin=m-s, vmax=m+3*s, cmap='Greys_r')
            ax1.plot(xt, yt_fit+specwidth, 'r--')
            ax1.plot(xt, yt_fit-specwidth, 'r--')

            ax1.set_ylim(min(yt_fit)-10, max(yt_fit)+10)
            ax2.plot(xt, deblazed_order/np.median(deblazed_order),)
            ax2.set_ylim(0,1.3)
            plt.show()
            
        
        all_spectra.append(deblazed_order)
    
    return all_spectra, x_traces, y_traces




def median_norm(x, window=355):
    return x/signal.medfilt(x, window)

    

def replace_xpix_w_refined(x,x_ref):
    
    min_diffs = []
    for i in range(len(x)):
        
        x_diffs = x[i]-x_ref
        min_diffs.append( x_diffs[np.argmin( np.abs(x_diffs) )] )
            
    return x - np.median(min_diffs)





        

        






