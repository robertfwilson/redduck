import numpy as np
from astropy import units as u
from astropy.nddata import CCDData
from astropy import modeling
from astropy.convolution import convolve, Gaussian2DKernel, convolve_fft
import matplotlib.pyplot as plt
from glob import glob

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *










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







