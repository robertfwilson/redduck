import numpy as np
import barycorrpy 



class Billocity(object):

    def __init__(self, TargetBelly, StandardBelly, good_orders=range(7,99)):

        self.template=StandardBelly
        self.target=TargetBelly

    def _calc_crosscorr(self, ordernum):        
        

        return dv, cross


    def _get_barrycorr(self, ):

        return 0.



    
def get_rv_cross_correlation(star_wave, star_flux, temp_wave, temp_flux, os_factor=2):
    
    max_wave = max(max(star_wave), max(temp_wave))
    min_wave = min(min(star_wave), min(temp_wave))
    
    wave_scale = np.logspace(np.log10(min_wave),np.log10(max_wave),os_factor*len(temp_wave))
    
    starflux_interp = np.interp(wave_scale, star_wave, star_flux, left=1., right=1.)
    tempflux_interp = np.interp(wave_scale, temp_wave, temp_flux, left=1., right=1.)

    cross_corr = correlate(1.-starflux_interp, 1.-tempflux_interp, mode='same')
    dlambda = wave_scale - np.median(wave_scale)
    
    dv = (dlambda/np.median(wave_scale) ) * 299792.
    
    return dv, cross_corr
