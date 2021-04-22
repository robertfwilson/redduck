import numpy as np
from astropy import units as u
from astropy.nddata import CCDData
from astropy.io import fits
from astropy import modeling
from astropy.convolution import convolve, Gaussian2DKernel, convolve_fft
import matplotlib.pyplot as plt
from glob import glob

from .utils import *

import photutils
from scipy.interpolate import griddata



def course_coord_update(template, img, return_convolved=False, smooth=True, nsig=10):

    '''
        This function takes in one image (template), and gives the offset in x and y needed to align another 
        image (img) to the first image. This is performed by cross-correlating the 2 images against eachother 
        and selecting the max point. Because I'm simply taking the maximum point, it is a little susceptible to 
        outliers and things like cosmic rays, but the smoothing function helps to mitigate that quite a bit. 
        
        Important Note: Because this is simply selecting the highest point, it does not give subpixel-centroids. 
        Hence, why I call it a course coordinate update. I'd recommend running this function on an image first, 
        and then running the fine_coord_update() functions once you get an estimate where the star should be. 

        template: 2D array. The image in which you know the centroid of your target star
        img: 2D array. The image for which you want to know the coordinate offset from the "template" image
        return_convolved: bool. whether or not you want to return the 2d correlation image
        smooth: bool. if you want to smooth the 2d correlated image before calculating the maximum. 
        nsig: float. number of pixels to use as a standard deviation on the gaussian smoothing
    '''
    
    img = np.copy(img)
    template = np.copy(template[::-1, ::-1])

    img -= np.median(img)
    img[img>nsig*np.std(img)] = nsig*np.std(img)
    img[img<-nsig*np.std(img)] = -nsig*np.std(img)

    template-=np.median(template)
    template[template>nsig*np.std(template)] = nsig*np.std(template)
    template[template<-nsig*np.std(template)] = -nsig*np.std(template)


    try:
        convolved = convolve_fft(img, template, boundary='wrap', normalize_kernel=True)
    except Exception:
        convolved = convolve_fft(img, template, boundary='wrap', normalize_kernel=False)

    if smooth:
        gauss_kernel = Gaussian2DKernel(2)
        convolved = convolve(convolved, gauss_kernel, boundary='wrap')

    convolved_masked = np.copy(convolved)
    m,s = np.median(convolved), np.std(convolved)
    high_mask = convolved > 3.*s+m
    convolved_masked[~high_mask] = m
    
    x_ind, y_ind = np.argmax(np.sum(convolved_masked, axis=0)), np.argmax(np.sum(convolved_masked, axis=1))
    
    x_offset = np.arange(-len(img[0, :])/2, len(img[0, :])/2)
    y_offset = np.arange(-len(img[:, 0])/2, len(img[:, 0])/2)

    if return_convolved:
        return x_offset[x_ind], y_offset[y_ind], convolved_masked
    else:
        return x_offset[x_ind], y_offset[y_ind]



def fine_coord_update(img, x_guess, y_guess, mask_max_counts=65000, box_width=30, plot_fit=False, smooth=True, kernel_size=1.):
    
    '''
        img: 2D array. Should be the image you are analyzing
        x_guess: int, 1st guess for the x coordinate. Needs to be closer than box_width
        y_guess: int, 1st guess for the y coordinate. Needs to be closer than box_width
        mask_max_counts: Set all points with counts higher than this number equal to the median
        box_width: int,  The area to consider for the stars coordinates. Needs to be small enough to not include 
            extra stars, but big enough not to include errors on your x,y guess
        plot_fit: bool, show a plot to the gauss fit? 
        smooth: bool, convolve image with gaussian first? The advantage of this is that it will take out some 
            of the errors caused by the image being a donut instead of a gaussian. Especially useful for 
            non-uniform PSFs, such as ARCSAT's defocused image. For ARCTIC, this may note be necessary. 
            Try it anyway though! 
        kernel_size: float, standard deviation of gaussian kernel used to smooth data (pixels). Irrevelvant 
            if smooth is set to False
    '''
    
    box_size = int(box_width/2)
    x_guess = int(x_guess)
    y_guess = int(y_guess)
    
    # cutout the part of the image around the star of interest
    stamp = img[y_guess-box_size:y_guess+box_size,x_guess-box_size:x_guess+box_size ].astype(np.float64)
    cutout = np.copy(stamp)
    
    # change saturated pixels to 0, so it doesn't throw off fit
    cutout[cutout>mask_max_counts] = 0.
    
    if smooth:
        # Convolve image with gaussian kernel to limit the noise
        gauss_kernel = Gaussian2DKernel(kernel_size)
        cutout = convolve_fft(cutout, gauss_kernel, boundary='wrap',normalize_kernel=False)
    else:
        cutout_s = cutout
    
    # Sum pixels in x,y directions 
    x_sum = np.sum(cutout, axis=0)
    y_sum = np.sum(cutout, axis=1)

    x_sum -= np.min(x_sum)
    y_sum -= np.min(y_sum)
        

    # Fit a gaussian to the x and y summed columns
    l = np.arange(box_width)-box_size
    fitter = modeling.fitting.LevMarLSQFitter()
    model = modeling.models.Gaussian1D(amplitude=np.max(x_sum), mean=0.)   # depending on the data you need to give some initial values
    fitted_x = fitter(model, l, x_sum)
    fitted_y = fitter(model, l, y_sum)
    
    # Add the offset from the fitted gaussian to the original guess
    x_cen = x_guess + fitted_x.mean 
    y_cen = y_guess + fitted_y.mean 
    
    if plot_fit:
                
        f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4))
        
        ax1.plot(l, x_sum, 'o', color='C0', label='x offset')
        ax1.plot(l, y_sum, 'o', color='C1', label='y offset')

        ax1.plot(l, fitted_x(l), 'C0')
        ax1.plot(l, fitted_y(l), 'C1')
        
        ax1.legend()
            
        m,s = np.median(stamp), np.std(stamp)
        ax2.imshow(img, origin='lower', cmap='Greys_r', interpolation='nearest',vmin=m-s, vmax=m+s)
        ax2.plot(x_cen, y_cen, 'r.', label='Refined Centroid')
        ax2.plot(x_guess, y_guess, 'b.', label='Initial Centroid')
        ax2.legend()
        
        ax2.set_xlim(x_cen-box_width, x_cen+box_width)
        ax2.set_ylim(y_cen-box_width, y_cen+box_width)

        f = np.copy(img)
        ygrid, xgrid = np.mgrid[0:len(f[:,0]), 0:len(f[0,:])]

        r = np.sqrt((xgrid-x_cen)**2. + (ygrid-y_cen)**2. )
        rcut = r<box_width/1.5

        ax3.plot(r[rcut].flatten(), f[rcut].flatten()-np.median(f), 'k.')
        
        plt.tight_layout()
        plt.show()
    
    
    return x_cen, y_cen




def update_coords(img, x_guess, y_guess, mask_max_counts=65000,box_width=30,plot_fit=False,
                  smooth=True, kernel_size=2.):
    
    '''
        img: 2D array. Should be the image you are analyzing
        x_guess: int, 1st guess for the x coordinate. Needs to be closer than box_width
        y_guess: int, 1st guess for the y coordinate. Needs to be closer than box_width
        mask_max_counts: Set all points with counts higher than this number equal to the median
        box_width: int,  The area to consider for the stars coordinates. Needs to be small enough to not include 
            extra stars, but big enough not to include errors on your x,y guess
        plot_fit: bool, show a plot to the gauss fit? 
        smooth: bool, convolve image with gaussian first? The advantage of this is that it will take out some 
            of the errors caused by the image being a donut instead of a gaussian. Especially useful for 
            non-uniform PSFs, such as ARCSAT's defocused image. For ARCTIC, this may not be necessary. 
            Try it anyway though! 
        kernel_size: float, standard deviation of gaussian kernel used to smooth data (pixels). Irrevelvant 
            if smooth is set to False
    '''
    
    box_size = int(box_width/2)
    
    x_guess = int(x_guess)
    y_guess=int(y_guess)
    # cutout the part of the image around the star of interest
    stamp = img[y_guess-box_size:y_guess+box_size,x_guess-box_size:x_guess+box_size ].astype(np.float64)
    cutout = np.copy(stamp)
    
    # change saturated pixels to 0, so it doesn't throw off fit
    cutout[cutout>mask_max_counts] = 0.
    
    if smooth:
        # Convolve image with gaussian kernel to limit the noise
        gauss_kernel = Gaussian2DKernel(kernel_size)
        cutout = convolve(cutout, gauss_kernel, boundary='extend')
    else:
        cutout_s = cutout
    # Subtract sky background
    cutout -= np.median(cutout)
    
    # Sum pixels in x,y directions 
    x_sum = np.sum(cutout, axis=0)
    y_sum = np.sum(cutout, axis=1)

    # Fit a gaussian to the x and y summed columns
    offset = np.arange(box_width)-box_size
    fitter = modeling.fitting.LevMarLSQFitter()
    model = modeling.models.Gaussian1D()   # depending on the data you need to give some initial values
    fitted_x = fitter(model, offset, x_sum)
    fitted_y = fitter(model, offset, y_sum)
    
    # Add the offset from the fitted gaussian to the original guess
    x_cen = x_guess + fitted_x.mean 
    y_cen = y_guess + fitted_y.mean 
    
    if plot_fit:
        
        f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))
        
        ax1.plot(offset, x_sum, 'o', color='C0', label='x offset')
        ax1.plot(offset, y_sum, 'o', color='C1', label='y offset')

        ax1.plot(offset, fitted_x(offset), 'C0')
        ax1.plot(offset, fitted_y(offset), 'C1')
        
        ax1.legend()
            
        m,s = np.median(stamp), np.std(stamp)
        ax2.imshow(stamp, vmin=m-s, vmax=m+s, origin='lower', cmap='Greys_r', interpolation='nearest',
                   extent=[-box_size,box_size,-box_size,box_size])
        ax2.plot(fitted_x.mean, fitted_y.mean, 'ro', label='updated')
        ax2.plot(0,0, 'bo', label='guess')
        ax2.legend()
        
        ax3.imshow(img, vmin=m-s, vmax=m+s, origin='lower', cmap='Greys_r', interpolation='nearest',)
        ax3.plot(x_cen, y_cen, 'ro', markersize=1)
        ax3.plot(x_guess, y_guess, 'bo', markersize=1)
        
        plt.tight_layout()
        plt.show()
    
    return x_cen, y_cen















def choose_best_aperture(img, x, y, radii, sig_thresh=3):

    data = np.copy(img)
    data = data.byteswap().newbyteorder()

    fluxes, _, _ = sep.sum_circle(data+sig_thresh*np.std(data), x, y, radii, subpix=0 )

    df = fluxes[1:]-fluxes[:-1]
    
    plt.plot(radii[1:], df, '-o')

    return radii[np.argmin(df) ]






def aperture_photometry(img, x, y, r, r_in=0, r_out=0, mask_max_counts=np.inf, subsky='median',return_err=False,gain=1.25,read_noise=10.):
    
    '''
        img: 2d numpy array, The data to perform phorometry on
        x: float, the x centroid of the star of interest (note: Could also be array if performing aperture photometry on multiple targets simultaneously)
        y: float, the y centroid of the star of interest (note: Could also be array if performing aperture photometry on multiple targets simultaneously)
        r: float, the DIAMETER of the aperture to be used in photometry (note: Could also be array if performing aperture photometry on multiple targets simultaneously)
        r_in, float, the DIAMETER of the inner annulus to be used in measuring the background flux (note: Could also be array if performing aperture photometry on multiple targets simultaneously)
        r_out, float, the DIAMETER of the outer annulus to be used in measuring the background flux (note: Could also be array if performing aperture photometry on multiple targets simultaneously)
    
    '''    
    
    # Make a copy of the data, so you don't obstruct the original in any way
    data = np.copy(img)
    data = data.byteswap().newbyteorder()
    
    # Mask out "bad" data points
    data[data>mask_max_counts] = np.median(data)

    positions = np.transpose([x,y])
    apertures = photutils.CircularAperture(positions, r=r)

    if subsky:
            #star_flux, star_flux_err, star_flux_flag = sep.sum_circle(data, x,y, r, bkgann=(r_in, r_out), subpix=5 )
        sky_annulus_aperture = photutils.CircularAnnulus(positions, r_in=r_in, r_out=r_out)
        annulus_mask = sky_annulus_aperture.to_mask(method='center')
        annulus_data = annulus_mask[0].multiply(data)

        annulus_data_1d = annulus_data[annulus_mask[0].data > 0]
        sky_bkg = np.median(annulus_data_1d) * apertures.area
        
    else:
        sky_bkg=0.
        


    phot_table = photutils.aperture_photometry(img, apertures, method='subpixel',
                                                   subpixels=5)
    star_flux = phot_table['aperture_sum'] - sky_bkg

    
    if return_err:
        n_bkg = np.pi * (r_out**2. - r_in**2.)
        n_pix = np.pi * r**2.
        nsky = sky_bkg
        
        sig =  np.sqrt(star_flux + n_pix * (1+n_pix/n_bkg) * (nsky + read_noise**2.+gain**2.*0.289**2. ) )

        return star_flux, sig
    
    return star_flux



def do_phot_nocoord(img_files, x, y, r, r_in, r_out, outfile, bw=30, airmass_key='AIRMASS', t_key='DATE-OBS', texp_key='EXPTIME', no_coord_update=False):


    phot_dicts = [{'x{}'.format(i):[], 'y{}'.format(i):[], 'flux{}'.format(i):[]} for i in range(len(x)) ]
    
    time = []
    airmass = []
        
    x_guess=x
    y_guess=y
    
    for j,f in enumerate(img_files):

        print('Measuring Image {0}/{1}:{2}'.format(j+1, len(img_files), f), end='\r' )

        img, header = fits.getdata(f, header=True)
            
        flux = aperture_photometry(img, x_guess, y_guess, r, r_in, r_out, mask_max_counts=65653, subsky=True)

        [dic['flux{}'.format(i)].append(flux[i]) for i,dic in enumerate(phot_dicts)]


    result = {k: v for d in phot_dicts for k, v in d.items()}
    result['time'] = time
    result['airmass'] = airmass
    result['frame'] = range(0,len(img_files) )

    result =  {k: np.array(v) for k, v in result.items()}
    
    return result






def replace_img_nans(array,method='cubic'):


    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    #mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    GD1 = griddata((x1, y1), newarr.ravel(),(xx, yy), method=method,fill_value=np.nanmedian(array))

    return GD1


def recursive_nan_replace(arr):
    i=0
    
    while any(np.isnan(arr).ravel() ) and i<5:
        i+=1
        if i>1:
            arr = replace_img_nans(arr, method='nearest')
        else:
            arr = replace_img_nans(arr, method='cubic')

    return arr

        

    
    
def mad(x):
    return np.median(np.abs(x- np.median(x)))*1.4826


    

def do_apphot(img_files, x, y, r, r_in, r_out, outfile, bw=30, airmass_key='AIRMASS', t_key='DATE-OBS', texp_key='EXPTIME', plot_coords=True, plot_all_coords=False, coord_kernel_size=3, update_centroids='fine', clean_crs=False, subsky=True, frame_int=(-9,-5),return_err=False,read_noise=13., gain=1.25 ):

    if return_err:
        phot_dicts = [{'x{}'.format(i):[], 'y{}'.format(i):[], 'flux{}'.format(i):[],'flux_err{}'.format(i):[]} for i in range(len(x)) ]
    else:
        phot_dicts = [{'x{}'.format(i):[], 'y{}'.format(i):[], 'flux{}'.format(i):[]} for i in range(len(x)) ]
    
    time = []
    airmass = []

    x_guess=x
    y_guess=y
    frame_list = []

    
    for j,f in enumerate(img_files):

        print('Measuring Image {0}/{1}:{2}'.format(j+1, len(img_files), f) )

        img, header = fits.getdata(f, header=True)
        frame_list.append(int( f[frame_int[0]:frame_int[1]] ))

        img = recursive_nan_replace(img)

        if clean_crs:
            #_, img = detect_cosmics(img, sigclip=4.5, cleantype='medmask',)
            img = img.byteswap().newbyteorder()

        time.append(header[t_key])
        airmass.append(header[airmass_key])
        

        if update_centroids!=False:
                
            if j>0 and update_centroids != 'fine':
                template = fits.getdata(img_files[j-1])
                dx, dy = course_coord_update(template = template, img = img, nsig=5 , )
            else:
                dx, dy = 0,0
                
            for i,dic in enumerate(phot_dicts):

                if i==0:
                    plot_fit=plot_coords
                else:
                    plot_fit=plot_all_coords
            
                x_update, y_update  = fine_coord_update(img, x_guess[i]+dx, y_guess[i]+dy, box_width=bw, plot_fit=plot_fit, smooth=True, kernel_size=coord_kernel_size)

                dic['x{}'.format(i)].append(x_update)
                dic['y{}'.format(i)].append(y_update)
            
                x_guess[i] = x_update
                y_guess[i] = y_update

            

        if return_err:

            flux, flux_err = aperture_photometry(img, x_guess, y_guess, r, r_in, r_out, mask_max_counts=65653, subsky=subsky, return_err=return_err, read_noise=read_noise,gain=gain)
            [dic['flux_err{}'.format(i)].append(flux_err[i]) for i,dic in enumerate(phot_dicts)]

        else:
            flux  = aperture_photometry(img, x_guess, y_guess, r, r_in, r_out, mask_max_counts=65653, subsky=subsky, return_err=return_err, read_noise=read_noise,gain=gain)


        [dic['flux{}'.format(i)].append(flux[i]) for i,dic in enumerate(phot_dicts)]


    result = {k: v for d in phot_dicts for k, v in d.items()}
    result['time'] = time
    result['airmass'] = airmass
    result['frame'] = frame_list

    result =  {k: np.array(v) for k, v in result.items()}
    
    return result



if __name__=='__main__':

    print('go go go')

