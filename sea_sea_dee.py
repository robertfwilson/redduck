import numpy as np
from astropy import units as u
from astropy.nddata import CCDData
from astropy.io import fits
from astropy import modeling
from astropy.convolution import convolve, Gaussian2DKernel, convolve_fft
import matplotlib.pyplot as plt
from glob import glob

from .utils import *

import ccdproc

try:
    from astroscrappy import detect_cosmics
except ImportError:
    print('AstroScrappy Not Installed. That\'s Okay, just don\'t try to remove cosmic rays')
    



# A quick function to get a list of files with the same prefix
def get_file_list(fnames, directory):
    f = directory + fnames
    return sorted(glob(f) )

# A quick function to look at an image
def display_image(image):
    plt.figure(figsize=(9,6))
    m, s = np.mean(image), np.std(image)
    plt.imshow(image, cmap='Greys_r', vmax=m+s, vmin=m-s, origin='lower')
    plt.colorbar(label='Counts')
    plt.show()




def mk_masterbias(bias_prefix, DATA_DIR, gain=1.25):

    bias_img_files = get_file_list(bias_prefix, DATA_DIR)

    print('Combining {} Bias Images'.format(len(bias_img_files) ) )

    # Combine list of bias images into 
    combined_bias = ccdproc.combine(bias_img_files, unit='adu', sigma_clip=True, method='median')

    # Correct for the gain
    master_bias = ccdproc.gain_correct(combined_bias, gain * u.photon/u.adu)
    
    return master_bias





def mk_masterdark(dark_prefix, DATA_DIR, master_bias, gain=1.25, texp_key='EXPTIME'):

    # Make list of Dark Images
    dark_img_files = get_file_list(dark_prefix, DATA_DIR)

    print('Combining {} Dark Images'.format(len(dark_img_files) ) ) 
    # Get the Exposure Time for the Darks
    dark_exptime = fits.getheader(dark_img_files[0])[texp_key]

    # Combine the dark frames into one 
    combined_dark = ccdproc.combine(dark_img_files, unit='adu', sigma_clip=True, method='median')
    
    # Gain correct the Dark Frame
    combined_dark_gaincorr = ccdproc.gain_correct(combined_dark, gain * u.photon/u.adu)

    # Subtract the Master Bias from the Dark Frame so that you are left only with Dark Current
    master_dark = ccdproc.subtract_bias(combined_dark_gaincorr, master_bias)

    return master_dark




def mk_masterflat(flat_prefix, DATA_DIR, master_dark, master_bias, dark_exptime, gain=1.25, texp_key='EXPTIME', combine_method='median'):

    # Make list of Flat-Field Images
    flat_img_files = get_file_list(flat_prefix, DATA_DIR)

    print('Combining {} Flat Field Images'.format(len(flat_img_files) ) ) 
    # Combine the Flat Field Images into a Single Image
    combined_flat = ccdproc.combine(flat_img_files, unit='adu', sigma_clip=True, method=combine_method)

    # Get the Exposure time for the Flat field, so that you can appropriately subtract the dark current
    flat_exptime = fits.getheader(flat_img_files[0])[texp_key]

    # Gain Correct the Flat Field
    combined_flat_gaincorr = ccdproc.gain_correct(combined_flat, gain * u.photon/u.adu)

    # Subtract the master bias Frame
    combined_flat_biassub = ccdproc.subtract_bias(combined_flat_gaincorr, master_bias)


    if not(master_dark is None):
        # Subtract off the Dark current to be left with only the relative scale of the images
        master_flat = ccdproc.subtract_dark(combined_flat_biassub, master_dark, dark_exposure=dark_exptime*u.second, data_exposure=flat_exptime*u.second,scale=True)
    else:
        return combined_flat_biassub

    return master_flat




def correct_image(img, master_bias, master_dark, master_flat, pixel_mask, dark_exptime, data_exptime, texp_key='EXPTIME', gain=1.25, fringe_frame=None, clean_cosmicrays=True,**cosmicray_kw):

    img_clean = img
    
    if clean_cosmicrays:
        data_mask, data_cleaned = detect_cosmics(img.data,inmask=pixel_mask,**cosmicray_kw)
        img_clean.data = data_cleaned
    
    img_gaincorr = ccdproc.gain_correct(img_clean, gain * u.photon/u.adu)
    
    img_bsub = ccdproc.subtract_bias(img_gaincorr, master_bias)

    if not(master_dark is None):
        img_dsub = ccdproc.subtract_dark(img_bsub, master_dark, dark_exposure=dark_exptime*u.second,data_exposure=data_exptime*u.second,scale=True)
    else:
        img_dsub=img_bsub

    if not(master_flat is None):
        img_corr = ccdproc.flat_correct(img_dsub, flat=master_flat)
    else:
        img_corr = img_dsub


    return img_corr





def correct_all_imgs(data_dir, img_fname, bias_fname, dark_fname, flat_fname, dark_exptime, sci_exptime, texp_key='EXPTIME', gain=1.25,pixel_mask=None, fringe_frame=None, clean_cosmicrays=False, combine_all=False, **cosmicray_kw):


    print('Creating Master Bias...')
    mbias = mk_masterbias(bias_fname, data_dir, gain=gain, )

    if dark_fname is None:
        print('Skipping Master Dark Creation...')
        mdark=None
    else:
        print('Creating Master Dark...')
        mdark = mk_masterdark(dark_fname, data_dir, mbias, gain=gain, texp_key=texp_key)

    if flat_fname is None:
        print('Skipping Master Flat Creation...')
        mflat=None
    else:
        print('Creating Master Flat...')
        mflat = mk_masterflat(flat_fname, data_dir, mdark, mbias, dark_exptime=dark_exptime, gain=gain, texp_key=texp_key)

    sci_files = sorted(get_file_list(img_fname, data_dir) )

    if pixel_mask==None:
        pixel_mask = np.zeros_like(mbias.data)


    corrected_img_files = []
    print('Correcting {} Science images'.format(len(sci_files)))
    for i,f in enumerate(sci_files):

        data, header = fits.getdata(f, header=True)
        
        img = CCDData(data, unit='adu')
        data_exptime = header[texp_key]
        
        corr_img = correct_image(img, master_bias=mbias, master_dark=mdark, master_flat=mflat, dark_exptime=dark_exptime, pixel_mask=pixel_mask, data_exptime = data_exptime, texp_key=texp_key, clean_cosmicrays=clean_cosmicrays, gain=gain, **cosmicray_kw) 

        if fringe_frame != None:
            fringe = fits.getdata(data_dir+fringe_frame)
            fringe_scaled = fringe * (np.median(corr_img)/np.median(fringe))
            corr_img = corr_img - fringe_scaled

        fits.writeto(data_dir+'corrected_'+img_fname[:-1]+'_{0:04d}.fits'.format(i), data=corr_img, header=header )

        if combine_all:
            corrected_img_files.append(data_dir+'corrected_'+img_fname[:-1]+'_{0:04d}.fits'.format(i))

        print('File {0}/{1}  Processed: {2}'.format(i+1, len(sci_files), f) )


    if combine_all:
        print('Combining all Proceesed Images ...  ')
        combined_science_img = ccdproc.combine(corrected_img_files, unit='adu', sigma_clip=True, method='median')
        fits.writeto(data_dir+'combined_'+img_fname[:-1]+'.fits', data=combined_science_img, header=header )

        
    return 1.




def process_ccd_imgs(data_dir, img_fname, bias_fname, dark_fname, flat_fname, pixel_mask, dark_exptime, sci_exptime, oscan=None, trim=None, texp_key='EXPTIME', gain=1.25, dark_scale=True , ):


    print('Creating Master Bias...')
    mbias = mk_masterbias(bias_fname, data_dir, gain=gain, )

    print('Creating Master Dark...')
    mdark = mk_masterdark(dark_fname, data_dir, mbias, gain=gain, texp_key=texp_key)

    print('Creating Master Flat...')
    mflat = mk_masterflat(flat_fname, data_dir, mdark, mbias, dark_exptime=dark_exptime, gain=gain, texp_key=texp_key)

    sci_files = sorted(get_file_list(img_fname, data_dir) )


    return 1.
