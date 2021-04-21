import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from scipy import signal


class Bill(object):

    def __init__(self, TargetBelly, StandardBelly, bad_orders=[]):

        self.template=StandardBelly
        self.target=TargetBelly

        self.order_waves=None
        self.rvs = None
        self.crosscorrs = {}

    def _calc_crosscorr(self, ordernum,trim_edge=250,window=301):

        targetflux = self.target._get_norm_flux(ordernum)
        targetwave = self.target.WaveCal.get_wavelengths(ordernum, targetflux)


        templateflux = self.template._get_norm_flux(ordernum)
        templatewave = self.target.WaveCal.get_wavelengths(ordernum, templateflux)
                
        dv, crosscorr = get_rv_crosscorr(targetwave,targetflux, templatewave,
                                         templateflux,os_factor=2,
                                         trim_edge=trim_edge)
        
        return dv, crosscorr


    def _get_barycorr(self, ra, dec,  obstime, earthlocation='APO'):

        earth = EarthLocation.of_site(earthlocation)
        sc = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))

        midtime = Time(obstime, format='isot',scale='tai')
        barycorr = sc.radial_velocity_correction(obstime=midtime, location=earth)  

        return  barycorr.to(u.km/u.s) / (u.km/u.s)



    def get_all_crosscorrs(self, ):

        return 1.



def median_norm(x, window=255):
    return x/signal.medfilt(x, window)

def get_centroid(x, y, i_x, dx=5, deg=1):
    
    leftdrop=y[i_x]-y[i_x-1]
    rightdrop=y[i_x]-y[i_x+1]
    
    if leftdrop>rightdrop:
        i_x+=1
    return np.sum(x[i_x-dx:i_x+dx]*y[i_x-dx:i_x+dx]**deg)/np.sum(y[i_x-dx:i_x+dx]**deg)


    
def get_rv_crosscorr(star_wave, star_flux, temp_wave, temp_flux, os_factor=2,trim_edge=200):


    trim_edge=int(trim_edge)

    if any(star_wave[1:]-star_wave[:-1]<1):
        star_flux = star_flux[::-1]
        star_wave = star_wave[::-1]
    if any(temp_wave[1:]-temp_wave[:-1]<1):
        temp_flux = temp_flux[::-1]
        temp_wave = temp_wave[::-1]
         
    max_wave = max(max(star_wave), max(temp_wave))
    min_wave = min(min(star_wave), min(temp_wave))
    
    wave_scale = np.logspace(np.log10(min_wave),np.log10(max_wave),os_factor*len(temp_wave))
    
    starflux_interp = np.interp(wave_scale, star_wave[trim_edge:-trim_edge],
                                star_flux[trim_edge:-trim_edge], left=1., right=1.)
    tempflux_interp = np.interp(wave_scale, temp_wave[trim_edge:-trim_edge],
                                temp_flux[trim_edge:-trim_edge], left=1., right=1.)


    cross_corr = np.correlate(1.-starflux_interp, 1.-tempflux_interp, mode='same')
    dlambda = wave_scale - np.median(wave_scale)
    
    dv = (dlambda/np.median(wave_scale) ) * 299792.
    
    return dv, cross_corr


