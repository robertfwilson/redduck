import numpy as np
import pandas as pd


from astropy import units as u
from astropy import modeling
from astropy.convolution import convolve, Gaussian2DKernel, convolve_fft
import matplotlib.pyplot as plt
from glob import glob

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

from .eshelduck import extract_order, get_order_trace, save_order

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)





def get_thar_lines_in_order(order, instrument='arces'):

    tharlines = pd.read_csv(dir_path+'/'+instrument+'/thar_solution.txt')

    ordertharlines = tharlines.loc[tharlines['order']==order+1] # add one becasue it's easier than changing the whole file. 
    wave, xpix = ordertharlines['wave'].to_numpy(), ordertharlines['xpix'].to_numpy()

    return xpix, wave




def refine_peaks(xpix, flux, xcens, window_width=3., mode='gauss'):
    
    fitter = modeling.fitting.LevMarLSQFitter()
    
    refined_xcen_signoise = []
    refined_xcens = []
    bad_indices = []
    for i,x0 in enumerate(xcens):
        
        x = xpix[np.abs(xpix-x0) <= window_width]
        y = flux[np.abs(xpix-x0) <= window_width]
    
        if mode=='gauss':
            model = modeling.models.Gaussian1D(mean=x0, amplitude=np.max(y),stddev=1.5,
                                               bounds={'mean': (min(x), max(x)), 'stddev':(0.25,1.5)},
                                              )
            model.cov_matrix=True
            
            fitted_peak = fitter(model,x,y)
            x0_refined = 0.+fitted_peak.mean 
            
                        
            signoise = fitted_peak.amplitude/np.sqrt(np.median(flux[np.abs(xpix-x0) <= 3.*window_width])+fitted_peak.amplitude )
            
            if np.abs(x0_refined-x0)>2.:
                print('BAD PEAK AT {}'.format(int(x0)))
                plt.plot(x,y,'o')
                xp = np.linspace(min(x), max(x), 50)
                plt.plot(xp, fitted_peak(xp))
                plt.axvline(x0_refined)
                plt.show()
                
                bad_indices.append(i)
                
        
        refined_xcens.append(x0_refined)
        refined_xcen_signoise.append(signoise)
        
    bad_peak_mask = np.ones_like(refined_xcens, dtype=bool)
    bad_peak_mask[bad_indices] = False
                    
    return np.array(refined_xcens), np.array(refined_xcen_signoise), bad_peak_mask



def calculate_wavesol(thar_spec, knownx, waves,order_num, plot=True,):


    refined_x, _, _ = refine_peaks(np.arange(len(thar_spec) ), thar_spec, knownx)


    if len(refined_x)<7:
        polydegree=3
    else:
        polydegree=4
    
    wave_sol = np.poly1d(np.polyfit(refined_x, waves, polydegree))    

    if plot:
        
        f, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True, figsize=(9,6))

        ax1.plot(np.arange(len(thar_spec)), thar_spec/np.max(thar_spec))
        ax1.set_ylim(-0.05,1.05)


        for i,lin in enumerate(refined_x):
            for ax in [ax1,ax2,ax3]:
                ax.axvline(lin, ls='--', color='r', lw=1)
    
    
        ax1.set_title('ThAr Line Calibration: Order {}, $\\lambda_0$ {}'.format(order_num, int(wave_sol(1000.)) ), )
    
        ax2.plot(knownx,waves, 'ro')
        ax2.plot(np.arange(len(thar_spec)), wave_sol(np.arange(len(thar_spec))), 'k-')
    
        resids = waves-wave_sol(refined_x)
    
        ax3.plot(refined_x, resids, 'ro', label='rms: {:.4f}'.format(np.std(resids)))
        ax3.axhline(0, ls='--', color='k')
    
        ax3.set_xlabel('pixel number')
        ax1.set_ylabel('Intensity')
        ax2.set_ylabel('Wavelength ($\mathregular{\AA}$)')
        ax3.set_ylabel('Resid. ($\mathregular{\AA}$)')
    
        ax3.legend()
        
        plt.subplots_adjust(hspace=0)
        plt.tight_layout()

        try:
            plt.savefig('wavecals/wavecal_order{}_'.format(order_num+1)+'.pdf')
        except FileNotFoundError:
            os.mkdir('wavecals')
            plt.savefig('wavecals/wavecal_order{}_'.format(order_num+1)+'.pdf')        

        plt.show()

    if  len(wave_sol.coeffs)==5:
        return wave_sol.coeffs

    elif len(wave_sol.coeffs)==4:
        return np.append([0],wave_sol.coeffs)






class WaveCalSol:

    def __init__(self, orders=None, coeffs=None):

        self.orders = orders
        self.coeffs =  coeffs
        
    def get_wavelengths(self, order, flux): 

        good_index = np.argmin(np.abs(self.orders - order) )

        if self.orders[good_index]!=order:
            print('ORDER NOT SAVED IN SOLUTION, {}, {}'.format(self.orders[good_index],order) )
            return np.nan

        wave_poly = np.poly1d(self.coeffs[good_index])
        return wave_poly(np.arange(0,len(flux)) )



    def from_file(self, fname):

        df = pd.read_csv(fname)
        self.orders = df['order'].to_numpy()
        self.coeffs = df[['a0','a1','a2','a3','a4']].to_numpy()

        return self


    def from_lamps(self, master_thar, master_flat, orders, save=True, plot_extract=False,
                   plot_calibration=True, save_spec=False):

        self.orders = orders
        coeffs = []
        for i in orders:

            xtrace, ytrace = get_order_trace(master_flat, order_num=i, dx=2, dy=3.5)
            thar_spec, _ = extract_order(master_thar, xtrace, ytrace, 
                                         width=4., do_weighted_extraction=False,
                                         plot=plot_extract)
        
            knownx, knownwave = get_thar_lines_in_order(i)
            sol = calculate_wavesol(thar_spec, knownx, knownwave, i, plot=plot_calibration)

            coeffs.append(sol)

            if save_spec:

                flat_spec, _ = extract_order(master_flat, xtrace, ytrace, 
                                         width=4., do_weighted_extraction=False,
                                             plot=False)
                
                wavesave = np.poly1d(sol)
                save_order('ThAr', wave=wavesave(np.arange(len(thar_spec))),
                           flux=thar_spec, flat=flat_spec, order_num=i)
                
        coeffs = np.array(coeffs)
        self.coeffs = coeffs
        if save:
            
            caldict={'order':orders, 'a0':self.coeffs[:,0], 'a1':self.coeffs[:,1],
                      'a2':self.coeffs[:,2], 'a3':self.coeffs[:,3], 'a4':self.coeffs[:,4]} 

            caldf = pd.DataFrame(caldict).to_csv('wavesol.txt',index=False)


        return self



