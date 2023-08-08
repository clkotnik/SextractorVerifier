'''
Created on Aug 7, 2023

@author: clkot
'''

if __name__ == '__main__':
    pass

#!/usr/bin/python3 
"""
Created on Jul 28, 2022


CLKotnik 
Call source extractor "sextractor" command line program.

Various post processing tests.

"""
import logging
import subprocess
import sys
from os import path, remove, rename, environ
from glob import glob
import gzip
import bz2
import shutil
import csv
import configargparse

# hack for testing without updating PYTHONPATH
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/mnt/e/dev/AAVSO/AAVSOnetOccam/scripts/pypline')

from observatory import Observatories

import math
from scipy.spatial.distance import cdist
import numpy as np
import tempfile
from textwrap import dedent
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.io import fits
from astropy import wcs
from astropy.table import QTable, Table, Column, join
from astropy.wcs import utils

from astroquery.gaia import Gaia

import warnings

_svn_id_ = "$Id: sextractor.py 1293 2022-11-09 12:31:33Z aavsonet $"

# suppress runtime and astropy warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings('ignore', category=wcs.FITSFixedWarning)
warnings.filterwarnings('ignore', category=fits.column.VerifyWarning)
warnings.filterwarnings('ignore', category=fits.card.VerifyWarning)




__all__ = ['call_sextractor']

logger = logging.getLogger(__name__)

def myround(x, base):
    return base * round(x/base)

def crossmatch_pixels(xy1,xy2):
    """
    Crossmatch the two arrays of image x-y coordinates
    Find the minimum distance from xy1 to an entry in xy2
    and index of entry in xy2.
    """
    xm = cdist(xy1,xy2)
    result = [ [np.amin(x), np.argmin(x)] for x in xm]
    return result

def get_image_coords(filename):
    """
    # If image is plate solved:
    #         get coordinate of center
    #         get radius of FOV     
    #        get exposure time
    #        get filter
    #        get object name
    #        get object sky coordinates
    #        get observatory
    #        get WCS
    #center,fov,exptime,filt,object,objcoord,xmax,ymax
    """
    try:
        hdul = fits.open(filename)
        hdr = hdul[0].header
    except Exception as e:
        logging.exception("Cannot open FITS header for file {}".format(filename))
        return None,None       
    
    exptime = hdr['EXPTIME']
    filt = hdr['FILTER']
    object = hdr['OBJECT']
    
    objctra = hdr['OBJCTRA'].replace(" ",":")
    objctdec = hdr['OBJCTDEC'].replace(" ",":")
    observat = hdr['OBSERVAT']
    
    objct_coord = SkyCoord(f"{objctra} {objctdec}",unit=(u.hour,u.deg))    
    w = WCS(hdr)
    if (w.has_celestial):
        frame=wcs_to_celestial_frame(w)
        logging.info(f"wcs_to_celestial_frame {frame}")
        # Find RA/DEC of center of image
        x = hdr['NAXIS1'] / 2
        y = hdr['NAXIS2'] / 2
        lon, lat = w.all_pix2world(x,y,0)
        center_coords = SkyCoord(lon,lat,unit=(u.deg,u.deg))
        lon, lat = w.all_pix2world(0,0,0)
        corner_coords = SkyCoord(lon,lat,unit=(u.deg,u.deg))

        fov_deg = center_coords.separation(corner_coords).deg
        return center_coords,fov_deg,exptime,filt,object,objct_coord,observat,w,hdr['NAXIS1'],hdr['NAXIS2']
    else:
        return None,None,None,None,None,None,None,None,None,None
    

def get_gaia(ra,dec,radius, magnitude,wcs,xmax,ymax):
    """
    Get a table of Gaia DR3 objects near the given coordinate that are
    brighter than the given magnitude in Gaia G band.  We use the wcs,
    xmax and ymax to remove objects that fall outside this specific
    image.
    
    ra - center point RA in degrees
    dec - center DEC in degrees
    radius - search radius around center in degrees
    magnitude - dimmer limit of object magnitude in Gaia G band
    wcs - image WCS
    xmax, ymax - image NAXIS1 and 2
    
    returns astropy Table of object found with:
        ra,dec,Gaia DR3 ID,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,ang_sep from center
    """
    
    # we round the basic parameters and use the  rounded values in the filename
    # where the results are cached so there is a good chance
    # the result set can apply to more than one image for a night
    #
    # Round the center point to 5 hundredths of a degree
    ra_rnd = myround(ra,0.05)
    dec_rnd = myround(dec,0.05)
    # field of view to one hundredth
    fov_rnd = myround(radius,0.01)
    # magnitude to one tenth
    mag_rnd = myround(magnitude,0.1)
    
    
    gaia_table_name = f"gaia_{ra_rnd:.2f}_{dec_rnd:.2f}_{fov_rnd:.2f}_{mag_rnd:.1f}.ecsv"
    if path.exists(gaia_table_name):
        result_table = Table.read(gaia_table_name)
        logging.info(f"read {len(result_table)} cached gaia objects from {gaia_table_name} ")
        return Table.read(gaia_table_name)
    
    stmt_filled = GaiaSelect.format(cen_ra=ra,cen_dec=dec,radius=radius,magnitude=magnitude)
    try:
        job = Gaia.launch_job_async(stmt_filled)
        result_table = job.get_results()
    except Exception as e:
        logger.exception('Gaia select failed with exception {}', e)
        raise e    
    
    logging.info(f"wrote {len(result_table)} cached gaia objects to {gaia_table_name} ")
    result_table.write(gaia_table_name)
    
    # find the x,y pixel coordinates of the Gaia objects in this image
    scoords = SkyCoord(result_table['ra'],result_table['dec'],unit=u.deg)
    pcoords = utils.skycoord_to_pixel(scoords,wcs)
    result_table['x']=pcoords[0]
    result_table['y']=pcoords[1]
    
    # Return just those Gaia objects that are within the image boundary
    gcat2 = Table(dtype=result_table.dtype)
    for i in range(len(result_table)):
        if (0 <= result_table['x'][i] <= xmax) and (0 <= result_table['y'][i] <= ymax):
            gcat2.add_row(result_table[i])
    
    return gcat2

GaiaSelect = """SELECT ra,dec,designation,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag, DISTANCE(
   POINT({cen_ra}, {cen_dec}),
   POINT(ra, dec)) AS ang_sep
FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS(
   POINT({cen_ra}, {cen_dec}),
   CIRCLE(ra, dec, {radius}))
AND phot_g_mean_mag < {magnitude}
ORDER BY ang_sep ASC
"""

def call_sextractor2(filename,output_location=None,do_clean=True,do_filter=True,thresh=1.5,minarea=8,remove_tmp=False,gain=1.0,saturation=50000.0,plate_scale=0):
    """
    Wrapper around sextractor command.
    
    Create the files necessary to run sextractor in a temporary directory
    with dynamic modifications to the config file.  Execute extractor on
    the FITS file provided.  Put the results into an astropy Table.
    
    filename - path name of FITS file to analyze
    do_filter - True if gaussian filter should be run on image before analysis
    
    returns: Astropy Table with a row per object identified by sextractor in he image

    """
    
    # Setup files required by sextractor
    (dir,fn) = path.split(filename)
    if len(dir) == 0:
        dir = "."
    tmp_location = tempfile.mkdtemp(dir=dir,prefix=f"sextemp_{fn}_")
    
    param_location = path.join(tmp_location, 'se_pypline.param')
    config_location = path.join(tmp_location, 'se_pypline.cfg')
    filter_location = path.join(tmp_location, 'se_pypline.conv')
    if output_location is None:
        output_location = path.join(tmp_location, 'se_pypline.cat')
    nnw_location = path.join(tmp_location, 'se_pypline.nnw')
    config_contents = SExtractor_config2.format(param_file=param_location,
                                              filter_file=filter_location,
                                              do_filter='Y' if do_filter else 'N',
                                              output_cat=output_location,
                                              detect_minarea=minarea,
                                              clean='Y' if do_clean else 'N',
                                              saturation=saturation,
                                              gain=gain,
                                              nnw_file=nnw_location,
                                              detect_thresh=thresh,
                                              plate_scale=plate_scale
                                              )
    with open(config_location, 'w') as f:
        f.write(config_contents)
    with open(filter_location, 'w') as f:
        f.write(SExtractor_filter2)
    with open(nnw_location, 'w') as f:
        f.write(SExtractor_nnw2)
    with open(param_location, 'w') as f:
        for param in SExtractor_param2:
            f.write(f"{param}\n")    
    #print(config_location)
    
    # Execute the sextractor command
    sex_cmd = ["source-extractor","-c",config_location,filename]
    try:
        completion = subprocess.run(sex_cmd,
                                     stdout = subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
        sex_output = str(completion.stdout)
        sex_error = str(completion.stderr)
        logging.debug("call_sextractor return code: {}".format(completion.returncode))
        logging.debug(">>>>>>>>>>>ERROR>>>>>>>>>>>>\n{}\n<<<<<<<<<<<<<<<<<<<<<<<<<\n".format(sex_error))
        logging.debug(">>>>>>>>>>>OUTPUT>>>>>>>>>>>>\n{}\n<<<<<<<<<<<<<<<<<<<<<<<<<\n".format(sex_output))
        return_status = completion.returncode
    except subprocess.CalledProcessError as e:
        return_status = e.returncode
        logger.exception('call_sextractor failed for %s with exception', filename)
        if tmp_location is not None:
            shutil.rmtree(tmp_location)
        raise e
        

    # Place resulting objects into a list
    with open(output_location, newline='') as sexfile:
        outreader = csv.DictReader(sexfile, fieldnames=SExtractor_param2,delimiter=' ',skipinitialspace=True)
        rows=list()
        for row in outreader:
            for k in row.keys():
                if "." in row[k] or "e" in row[k]:
                    row[k] = float(row[k])
                else:
                    row[k] = int(row[k])
            rows.append(row)

    # Create an astropy Table from the rows
    t = Table(rows=rows)

    # Remove the temporary files
    if tmp_location is not None:
        if remove_tmp:
            shutil.rmtree(tmp_location)
        else:
            print(f"tmp location: {tmp_location}")
    
    return return_status,t
SExtractor_param2 = [
    "X_IMAGE",
    "Y_IMAGE",
    "XWIN_IMAGE",
    "YWIN_IMAGE",
    "ALPHA_SKY",
    "DELTA_SKY",
    "ALPHAWIN_SKY",
    "DELTAWIN_SKY",
    "NUMBER",
    "EXT_NUMBER",
    "FWHM_IMAGE",
    "FWHM_WORLD",
    "A_IMAGE",
    "B_IMAGE",
    "AWIN_IMAGE",
    "BWIN_IMAGE",
    "KRON_RADIUS",
    "THRESHOLD",
    "ELONGATION",
    "ELLIPTICITY",
    "XMIN_IMAGE",
    "YMIN_IMAGE",
    "XMAX_IMAGE",
    "YMAX_IMAGE",
    "BACKGROUND",
    "ISOAREA_IMAGE",
    "FLAGS",
    "FLAGS_WIN",
    "CLASS_STAR",
    "FLUX_MAX",
    "FLUX_ISO",
    "FLUXERR_ISO",
    "MAG_ISO",
    "MAGERR_ISO",
    "FLUX_WIN",
    "FLUXERR_WIN",
    "MAG_WIN",
    "MAGERR_WIN",
    "FLUX_APER",
    "FLUXERR_APER",
    "MAG_APER",
    "MAGERR_APER",
    "FLUX_AUTO",
    "FLUXERR_AUTO",
    "MAG_AUTO",
    "MAGERR_AUTO",
    "SNR_WIN"]
SExtractor_filter2 = """CONV NORM
# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.
1 2 1
2 4 2
1 2 1
"""
SExtractor_filter_pyramid2 = """CONV NORM
# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.
1 2 1
2 4 2
1 2 1
"""
SExtractor_filter_mexhat_7X72 = """CONV NORM
# 7x7 convolution mask of a mexican-hat for images with FWHM~2.5 pixels.
-0.000284 -0.002194 -0.007273 -0.010722 -0.007273 -0.002194 -0.000284
-0.002194 -0.015640 -0.041259 -0.050277 -0.041259 -0.015640 -0.002194
-0.007273 -0.041259 -0.016356 0.095837 -0.016356 -0.041259 -0.007273
-0.010722 -0.050277 0.095837 0.402756 0.095837 -0.050277 -0.010722
-0.007273 -0.041259 -0.016356 0.095837 -0.016356 -0.041259 -0.007273
-0.002194 -0.015640 -0.041259 -0.050277 -0.041259 -0.015640 -0.002194
-0.000284 -0.002194 -0.007273 -0.010722 -0.007273 -0.002194 -0.000284
"""
SExtractor_filter_tophap_3X32 = """CONV NORM
# 3x3 convolution mask of a top-hat PSF with diameter = 3.0 pixels.
0.560000 0.980000 0.560000
0.980000 1.000000 0.980000
0.560000 0.980000 0.560000
"""

SExtractor_filter_default2 = """CONV NORM
# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.
1 2 1
2 4 2
1 2 1
"""
SExtractor_filter_gausian_3X32 = """CONV NORM
# 3x3 convolution mask of a gaussian PSF with FWHM = 2.0 pixels.
0.260856 0.483068 0.260856
0.483068 0.894573 0.483068
0.260856 0.483068 0.260856
"""
SExtractor_filter_gausian_9X92 = """CONV NORM
# 9x9 convolution mask of a gaussian PSF with FWHM = 5.0 pixels.
0.030531 0.065238 0.112208 0.155356 0.173152 0.155356 0.112208 0.065238 0.030531
0.065238 0.139399 0.239763 0.331961 0.369987 0.331961 0.239763 0.139399 0.065238
0.112208 0.239763 0.412386 0.570963 0.636368 0.570963 0.412386 0.239763 0.112208
0.155356 0.331961 0.570963 0.790520 0.881075 0.790520 0.570963 0.331961 0.155356
0.173152 0.369987 0.636368 0.881075 0.982004 0.881075 0.636368 0.369987 0.173152
0.155356 0.331961 0.570963 0.790520 0.881075 0.790520 0.570963 0.331961 0.155356
0.112208 0.239763 0.412386 0.570963 0.636368 0.570963 0.412386 0.239763 0.112208
0.065238 0.139399 0.239763 0.331961 0.369987 0.331961 0.239763 0.139399 0.065238
0.030531 0.065238 0.112208 0.155356 0.173152 0.155356 0.112208 0.065238 0.030531
"""
SExtractor_filter_gausian_7X72 = """CONV NORM
# 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
0.004963 0.021388 0.051328 0.068707 0.051328 0.021388 0.004963
0.021388 0.092163 0.221178 0.296069 0.221178 0.092163 0.021388
0.051328 0.221178 0.530797 0.710525 0.530797 0.221178 0.051328
0.068707 0.296069 0.710525 0.951108 0.710525 0.296069 0.068707
0.051328 0.221178 0.530797 0.710525 0.530797 0.221178 0.051328
0.021388 0.092163 0.221178 0.296069 0.221178 0.092163 0.021388
0.004963 0.021388 0.051328 0.068707 0.051328 0.021388 0.004963
"""
SExtractor_filter_gausian_5X52 = """CONV NORM
# 5x5 convolution mask of a gaussian PSF with FWHM = 2.5 pixels.
0.034673 0.119131 0.179633 0.119131 0.034673
0.119131 0.409323 0.617200 0.409323 0.119131
0.179633 0.617200 0.930649 0.617200 0.179633
0.119131 0.409323 0.617200 0.409323 0.119131
0.034673 0.119131 0.179633 0.119131 0.034673
"""
SExtractor_nnw2 = """NNW
# Neural Network Weights for the SExtractor star/galaxy classifier (V1.3)
# inputs:    9 for profile parameters + 1 for seeing.
# outputs:    ``Stellarity index'' (0.0 to 1.0)
# Seeing FWHM range: from 0.025 to 5.5'' (images must have 1.5 < FWHM < 5 pixels)
# Optimized for Moffat profiles with 2<= beta <= 4.

 3 10 10  1

-1.56604e+00 -2.48265e+00 -1.44564e+00 -1.24675e+00 -9.44913e-01 -5.22453e-01  4.61342e-02  8.31957e-01  2.15505e+00  2.64769e-01
 3.03477e+00  2.69561e+00  3.16188e+00  3.34497e+00  3.51885e+00  3.65570e+00  3.74856e+00  3.84541e+00  4.22811e+00  3.27734e+00

-3.22480e-01 -2.12804e+00  6.50750e-01 -1.11242e+00 -1.40683e+00 -1.55944e+00 -1.84558e+00 -1.18946e-01  5.52395e-01 -4.36564e-01 -5.30052e+00
 4.62594e-01 -3.29127e+00  1.10950e+00 -6.01857e-01  1.29492e-01  1.42290e+00  2.90741e+00  2.44058e+00 -9.19118e-01  8.42851e-01 -4.69824e+00
-2.57424e+00  8.96469e-01  8.34775e-01  2.18845e+00  2.46526e+00  8.60878e-02 -6.88080e-01 -1.33623e-02  9.30403e-02  1.64942e+00 -1.01231e+00
 4.81041e+00  1.53747e+00 -1.12216e+00 -3.16008e+00 -1.67404e+00 -1.75767e+00 -1.29310e+00  5.59549e-01  8.08468e-01 -1.01592e-02 -7.54052e+00
 1.01933e+01 -2.09484e+01 -1.07426e+00  9.87912e-01  6.05210e-01 -6.04535e-02 -5.87826e-01 -7.94117e-01 -4.89190e-01 -8.12710e-02 -2.07067e+01
-5.31793e+00  7.94240e+00 -4.64165e+00 -4.37436e+00 -1.55417e+00  7.54368e-01  1.09608e+00  1.45967e+00  1.62946e+00 -1.01301e+00  1.13514e-01
 2.20336e-01  1.70056e+00 -5.20105e-01 -4.28330e-01  1.57258e-03 -3.36502e-01 -8.18568e-02 -7.16163e+00  8.23195e+00 -1.71561e-02 -1.13749e+01
 3.75075e+00  7.25399e+00 -1.75325e+00 -2.68814e+00 -3.71128e+00 -4.62933e+00 -2.13747e+00 -1.89186e-01  1.29122e+00 -7.49380e-01  6.71712e-01
-8.41923e-01  4.64997e+00  5.65808e-01 -3.08277e-01 -1.01687e+00  1.73127e-01 -8.92130e-01  1.89044e+00 -2.75543e-01 -7.72828e-01  5.36745e-01
-3.65598e+00  7.56997e+00 -3.76373e+00 -1.74542e+00 -1.37540e-01 -5.55400e-01 -1.59195e-01  1.27910e-01  1.91906e+00  1.42119e+00 -4.35502e+00

-1.70059e+00 -3.65695e+00  1.22367e+00 -5.74367e-01 -3.29571e+00  2.46316e+00  5.22353e+00  2.42038e+00  1.22919e+00 -9.22250e-01 -2.32028e+00


 0.00000e+00 
 1.00000e+00 
"""
SExtractor_config2 = """
# Taken from astrometry.py from Matt Craig and default.sex ver 2.25.3
# obtained with "sextractor -d"
#

#-------------------------------- Catalog ------------------------------------
CATALOG_NAME     {output_cat}       # name of the output catalog
CATALOG_TYPE     ASCII     # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,
                                # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC
PARAMETERS_NAME  {param_file}  # name of the file containing catalog contents

#------------------------------- Extraction ----------------------------------

DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)
DETECT_MINAREA   {detect_minarea}              # min. # of pixels above threshold
DETECT_THRESH    {detect_thresh}             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
ANALYSIS_THRESH  {detect_minarea}            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2

FILTER           {do_filter}              # apply filter for detection (Y or N)?
#FILTER_NAME      /mnt/e/dev/AAVSO/AAVSOnetScripts/pypline/config/default.conv   # name of the file containing the filter
FILTER_NAME      {filter_file}  # name of the file containing the filter

DEBLEND_NTHRESH  32             # Number of deblending sub-thresholds
DEBLEND_MINCONT  0.005          # Minimum contrast parameter for deblending

CLEAN            {clean}              # Clean spurious detections? (Y or N)?
CLEAN_PARAM      1.0            # Cleaning efficiency

MASK_TYPE        CORRECT        # type of detection MASKing: can be one of
                                # NONE, BLANK or CORRECT

#------------------------------ Photometry -----------------------------------

PHOT_APERTURES   7              # MAG_APER aperture diameter(s) in pixels
PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius>
PHOT_PETROPARAMS 2.0, 3.5       # MAG_PETRO parameters: <Petrosian_fact>,
                                # <min_radius>

SATUR_LEVEL      {saturation}        # level (in ADUs) at which arises saturation
SATUR_KEY        SATURATE       # keyword for saturation level (in ADUs)

MAG_ZEROPOINT    0.0            # magnitude zero-point
MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)
GAIN             {gain}         # detector gain in e-/ADU
#                GAIN is obtained from the TELESCOP table and put in GAIN, above
#                so GAIN_KEY is removed
#GAIN_KEY         EGAIN          # keyword for detector gain in e-/ADU
PIXEL_SCALE      {plate_scale}  # size of pixel in arcsec (0=use FITS WCS info)

#------------------------- Star/Galaxy Separation ----------------------------

SEEING_FWHM      7.1           # stellar FWHM in arcsec
STARNNW_NAME     {nnw_file}    # Neural-Network_Weight table filename

#------------------------------ Background -----------------------------------

BACK_SIZE        64             # Background mesh: <size> or <width>,<height>
BACK_FILTERSIZE  3              # Background filter: <size> or <width>,<height>

BACKPHOTO_TYPE   GLOBAL         # can be GLOBAL or LOCAL
#BACKPHOTO_TYPE   LOCAL          # can be GLOBAL or LOCAL

#------------------------------ Check Image ----------------------------------

CHECKIMAGE_TYPE  NONE           # can be NONE, BACKGROUND, BACKGROUND_RMS,
                                # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,
                                # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,
                                # or APERTURES
CHECKIMAGE_NAME  back_gfilter.fits,obj_gfilter.fits,aper_gfilter.fits     # Filename for the check-image

#--------------------- Memory (change with caution!) -------------------------

MEMORY_OBJSTACK  3000           # number of objects in stack
MEMORY_PIXSTACK  300000         # number of pixels in stack
MEMORY_BUFSIZE   1024           # number of lines in buffer

#----------------------------- Miscellaneous ---------------------------------

VERBOSE_TYPE     NORMAL         # can be QUIET, NORMAL or FULL
HEADER_SUFFIX    .head          # Filename extension for additional headers
WRITE_XML        N              # Write XML file (Y/N)?
XML_NAME         sex.xml        # Filename for XML output
XSL_URL          file:///e:/documents/Astronomy/AAVSO/Development/AAVSOnet/srcext/test02_bsm_nh2_220723/sextractor.xsl

"""

def call_sextractor(filename,output_location=None,do_clean=True,do_filter=True,thresh=1.5,minarea=8):
    """
    Wrapper around sextractor command.
    
    Create the files necessary to run sextractor in a temporary directory
    with dynamic modifications to the config file.  Execute extractor on
    the FITS file provided.  Put the results into an astropy Table.
    
    filename - path name of FITS file to analyze
    do_filter - True if gaussian filter should be run on image before analysis
    
    returns: Astropy Table with a row per object identified by sextractor in he image

    """
    
    # Setup files required by sextractor
    tmp_location = tempfile.mkdtemp()
    param_location = path.join(tmp_location, 'se_pypline.param')
    config_location = path.join(tmp_location, 'se_pypline.cfg')
    filter_location = path.join(tmp_location, 'se_pypline.conv')
    if output_location is None:
        output_location = path.join(tmp_location, 'se_pypline.cat')
    nnw_location = path.join(tmp_location, 'se_pypline.nnw')
    config_contents = SExtractor_config.format(param_file=param_location,
                                              filter_file=filter_location,
                                              do_filter='Y' if do_filter else 'N',
                                              output_cat=output_location,
                                              detect_minarea=minarea,
                                              clean='Y' if do_clean else 'N',
                                              saturation='50000.0',
                                              gain='2.0',
                                              nnw_file=nnw_location,
                                              detect_thresh=thresh
                                              )
    with open(config_location, 'w') as f:
        f.write(config_contents)
    with open(filter_location, 'w') as f:
        f.write(SExtractor_filter)
    with open(nnw_location, 'w') as f:
        f.write(SExtractor_nnw)
    with open(param_location, 'w') as f:
        for param in SExtractor_param:
            f.write(f"{param}\n")    
    #print(config_location)
    
    # Execute the sextractor command
    sex_cmd = ["source-extractor","-c",config_location,filename]
    try:
        completion = subprocess.run(sex_cmd,
                                     stdout = subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
        sex_output = str(completion.stdout)
        sex_error = str(completion.stderr)
        logging.debug("call_sextractor return code: {}".format(completion.returncode))
        logging.debug(">>>>>>>>>>>ERROR>>>>>>>>>>>>\n{}\n<<<<<<<<<<<<<<<<<<<<<<<<<\n".format(sex_error))
        logging.debug(">>>>>>>>>>>OUTPUT>>>>>>>>>>>>\n{}\n<<<<<<<<<<<<<<<<<<<<<<<<<\n".format(sex_output))
        return_status = completion.returncode
    except subprocess.CalledProcessError as e:
        return_status = e.returncode
        logger.exception('call_sextractor failed for %s with exception', filename)
        if tmp_location is not None:
            shutil.rmtree(tmp_location)
        raise e
        

    # Place resulting objects into a list
    with open(output_location, newline='') as sexfile:
        outreader = csv.DictReader(sexfile, fieldnames=SExtractor_param,delimiter=' ',skipinitialspace=True)
        rows=list()
        for row in outreader:
            for k in row.keys():
                if "." in row[k] or "e" in row[k]:
                    row[k] = float(row[k])
                else:
                    row[k] = int(row[k])
            rows.append(row)

    # Create an astropy Table from the rows
    t = Table(rows=rows)

    # Remove the temporary files
    if tmp_location is not None:
        shutil.rmtree(tmp_location)
    
    return return_status,t

SExtractor_param = [
    "X_IMAGE",
    "Y_IMAGE",
    "XWIN_IMAGE",
    "YWIN_IMAGE",
    "ALPHA_SKY",
    "DELTA_SKY",
    "ALPHAWIN_SKY",
    "DELTAWIN_SKY",
    "NUMBER",
    "EXT_NUMBER",
    "FWHM_IMAGE",
    "FWHM_WORLD",
    "A_IMAGE",
    "B_IMAGE",
    "AWIN_IMAGE",
    "BWIN_IMAGE",
    "KRON_RADIUS",
    "THRESHOLD",
    "ELONGATION",
    "ELLIPTICITY",
    "XMIN_IMAGE",
    "YMIN_IMAGE",
    "XMAX_IMAGE",
    "YMAX_IMAGE",
    "BACKGROUND",
    "ISOAREA_IMAGE",
    "FLAGS",
    "FLAGS_WIN",
    "CLASS_STAR",
    "FLUX_MAX",
    "FLUX_ISO",
    "FLUXERR_ISO",
    "MAG_ISO",
    "MAGERR_ISO",
    "FLUX_WIN",
    "FLUXERR_WIN",
    "MAG_WIN",
    "MAGERR_WIN",
    "FLUX_APER",
    "FLUXERR_APER",
    "MAG_APER",
    "MAGERR_APER",
    "FLUX_AUTO",
    "FLUXERR_AUTO",
    "MAG_AUTO",
    "MAGERR_AUTO",
    "SNR_WIN"]
SExtractor_filter = """CONV NORM
# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.
1 2 1
2 4 2
1 2 1
"""
SExtractor_filter_pyramid = """CONV NORM
# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.
1 2 1
2 4 2
1 2 1
"""
SExtractor_filter_mexhat_7X7 = """CONV NORM
# 7x7 convolution mask of a mexican-hat for images with FWHM~2.5 pixels.
-0.000284 -0.002194 -0.007273 -0.010722 -0.007273 -0.002194 -0.000284
-0.002194 -0.015640 -0.041259 -0.050277 -0.041259 -0.015640 -0.002194
-0.007273 -0.041259 -0.016356 0.095837 -0.016356 -0.041259 -0.007273
-0.010722 -0.050277 0.095837 0.402756 0.095837 -0.050277 -0.010722
-0.007273 -0.041259 -0.016356 0.095837 -0.016356 -0.041259 -0.007273
-0.002194 -0.015640 -0.041259 -0.050277 -0.041259 -0.015640 -0.002194
-0.000284 -0.002194 -0.007273 -0.010722 -0.007273 -0.002194 -0.000284
"""
SExtractor_filter_tophap_3X3 = """CONV NORM
# 3x3 convolution mask of a top-hat PSF with diameter = 3.0 pixels.
0.560000 0.980000 0.560000
0.980000 1.000000 0.980000
0.560000 0.980000 0.560000
"""

SExtractor_filter_default = """CONV NORM
# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.
1 2 1
2 4 2
1 2 1
"""
SExtractor_filter_gausian_3X3 = """CONV NORM
# 3x3 convolution mask of a gaussian PSF with FWHM = 2.0 pixels.
0.260856 0.483068 0.260856
0.483068 0.894573 0.483068
0.260856 0.483068 0.260856
"""
SExtractor_filter_gausian_9X9 = """CONV NORM
# 9x9 convolution mask of a gaussian PSF with FWHM = 5.0 pixels.
0.030531 0.065238 0.112208 0.155356 0.173152 0.155356 0.112208 0.065238 0.030531
0.065238 0.139399 0.239763 0.331961 0.369987 0.331961 0.239763 0.139399 0.065238
0.112208 0.239763 0.412386 0.570963 0.636368 0.570963 0.412386 0.239763 0.112208
0.155356 0.331961 0.570963 0.790520 0.881075 0.790520 0.570963 0.331961 0.155356
0.173152 0.369987 0.636368 0.881075 0.982004 0.881075 0.636368 0.369987 0.173152
0.155356 0.331961 0.570963 0.790520 0.881075 0.790520 0.570963 0.331961 0.155356
0.112208 0.239763 0.412386 0.570963 0.636368 0.570963 0.412386 0.239763 0.112208
0.065238 0.139399 0.239763 0.331961 0.369987 0.331961 0.239763 0.139399 0.065238
0.030531 0.065238 0.112208 0.155356 0.173152 0.155356 0.112208 0.065238 0.030531
"""
SExtractor_filter_gausian_7X7 = """CONV NORM
# 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
0.004963 0.021388 0.051328 0.068707 0.051328 0.021388 0.004963
0.021388 0.092163 0.221178 0.296069 0.221178 0.092163 0.021388
0.051328 0.221178 0.530797 0.710525 0.530797 0.221178 0.051328
0.068707 0.296069 0.710525 0.951108 0.710525 0.296069 0.068707
0.051328 0.221178 0.530797 0.710525 0.530797 0.221178 0.051328
0.021388 0.092163 0.221178 0.296069 0.221178 0.092163 0.021388
0.004963 0.021388 0.051328 0.068707 0.051328 0.021388 0.004963
"""
SExtractor_filter_gausian_5X5 = """CONV NORM
# 5x5 convolution mask of a gaussian PSF with FWHM = 2.5 pixels.
0.034673 0.119131 0.179633 0.119131 0.034673
0.119131 0.409323 0.617200 0.409323 0.119131
0.179633 0.617200 0.930649 0.617200 0.179633
0.119131 0.409323 0.617200 0.409323 0.119131
0.034673 0.119131 0.179633 0.119131 0.034673
"""
SExtractor_nnw = """NNW
# Neural Network Weights for the SExtractor star/galaxy classifier (V1.3)
# inputs:    9 for profile parameters + 1 for seeing.
# outputs:    ``Stellarity index'' (0.0 to 1.0)
# Seeing FWHM range: from 0.025 to 5.5'' (images must have 1.5 < FWHM < 5 pixels)
# Optimized for Moffat profiles with 2<= beta <= 4.

 3 10 10  1

-1.56604e+00 -2.48265e+00 -1.44564e+00 -1.24675e+00 -9.44913e-01 -5.22453e-01  4.61342e-02  8.31957e-01  2.15505e+00  2.64769e-01
 3.03477e+00  2.69561e+00  3.16188e+00  3.34497e+00  3.51885e+00  3.65570e+00  3.74856e+00  3.84541e+00  4.22811e+00  3.27734e+00

-3.22480e-01 -2.12804e+00  6.50750e-01 -1.11242e+00 -1.40683e+00 -1.55944e+00 -1.84558e+00 -1.18946e-01  5.52395e-01 -4.36564e-01 -5.30052e+00
 4.62594e-01 -3.29127e+00  1.10950e+00 -6.01857e-01  1.29492e-01  1.42290e+00  2.90741e+00  2.44058e+00 -9.19118e-01  8.42851e-01 -4.69824e+00
-2.57424e+00  8.96469e-01  8.34775e-01  2.18845e+00  2.46526e+00  8.60878e-02 -6.88080e-01 -1.33623e-02  9.30403e-02  1.64942e+00 -1.01231e+00
 4.81041e+00  1.53747e+00 -1.12216e+00 -3.16008e+00 -1.67404e+00 -1.75767e+00 -1.29310e+00  5.59549e-01  8.08468e-01 -1.01592e-02 -7.54052e+00
 1.01933e+01 -2.09484e+01 -1.07426e+00  9.87912e-01  6.05210e-01 -6.04535e-02 -5.87826e-01 -7.94117e-01 -4.89190e-01 -8.12710e-02 -2.07067e+01
-5.31793e+00  7.94240e+00 -4.64165e+00 -4.37436e+00 -1.55417e+00  7.54368e-01  1.09608e+00  1.45967e+00  1.62946e+00 -1.01301e+00  1.13514e-01
 2.20336e-01  1.70056e+00 -5.20105e-01 -4.28330e-01  1.57258e-03 -3.36502e-01 -8.18568e-02 -7.16163e+00  8.23195e+00 -1.71561e-02 -1.13749e+01
 3.75075e+00  7.25399e+00 -1.75325e+00 -2.68814e+00 -3.71128e+00 -4.62933e+00 -2.13747e+00 -1.89186e-01  1.29122e+00 -7.49380e-01  6.71712e-01
-8.41923e-01  4.64997e+00  5.65808e-01 -3.08277e-01 -1.01687e+00  1.73127e-01 -8.92130e-01  1.89044e+00 -2.75543e-01 -7.72828e-01  5.36745e-01
-3.65598e+00  7.56997e+00 -3.76373e+00 -1.74542e+00 -1.37540e-01 -5.55400e-01 -1.59195e-01  1.27910e-01  1.91906e+00  1.42119e+00 -4.35502e+00

-1.70059e+00 -3.65695e+00  1.22367e+00 -5.74367e-01 -3.29571e+00  2.46316e+00  5.22353e+00  2.42038e+00  1.22919e+00 -9.22250e-01 -2.32028e+00


 0.00000e+00 
 1.00000e+00 
"""
SExtractor_config = """
# Taken from astrometry.py from Matt Craig and default.sex ver 2.25.3
# obtained with "sextractor -d"
#

#-------------------------------- Catalog ------------------------------------
CATALOG_NAME     {output_cat}       # name of the output catalog
CATALOG_TYPE     ASCII     # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,
                                # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC
PARAMETERS_NAME  {param_file}  # name of the file containing catalog contents

#------------------------------- Extraction ----------------------------------

DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)
DETECT_MINAREA   {detect_minarea}              # min. # of pixels above threshold
DETECT_THRESH    {detect_thresh}             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
ANALYSIS_THRESH  {detect_minarea}            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2

FILTER           {do_filter}              # apply filter for detection (Y or N)?
#FILTER_NAME      /mnt/e/dev/AAVSO/AAVSOnetScripts/pypline/config/default.conv   # name of the file containing the filter
FILTER_NAME      {filter_file}  # name of the file containing the filter

DEBLEND_NTHRESH  32             # Number of deblending sub-thresholds
DEBLEND_MINCONT  0.005          # Minimum contrast parameter for deblending

CLEAN            {clean}              # Clean spurious detections? (Y or N)?
CLEAN_PARAM      1.0            # Cleaning efficiency

MASK_TYPE        CORRECT        # type of detection MASKing: can be one of
                                # NONE, BLANK or CORRECT

#------------------------------ Photometry -----------------------------------

PHOT_APERTURES   7              # MAG_APER aperture diameter(s) in pixels
PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius>
PHOT_PETROPARAMS 2.0, 3.5       # MAG_PETRO parameters: <Petrosian_fact>,
                                # <min_radius>

SATUR_LEVEL      {saturation}   # level (in ADUs) at which arises saturation
SATUR_KEY        SATURATE       # keyword for saturation level (in ADUs)

MAG_ZEROPOINT    0.0            # magnitude zero-point
MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)
GAIN             {gain}         # detector gain in e-/ADU
#                OC61 images do not contain EGAIN and its gain is 2.0
GAIN_KEY         EGAIN          # keyword for detector gain in e-/ADU
PIXEL_SCALE      0              # size of pixel in arcsec (0=use FITS WCS info)

#------------------------- Star/Galaxy Separation ----------------------------

SEEING_FWHM      1.2            # stellar FWHM in arcsec
STARNNW_NAME     {nnw_file}    # Neural-Network_Weight table filename

#------------------------------ Background -----------------------------------

BACK_SIZE        64             # Background mesh: <size> or <width>,<height>
BACK_FILTERSIZE  3              # Background filter: <size> or <width>,<height>

BACKPHOTO_TYPE   GLOBAL         # can be GLOBAL or LOCAL
#BACKPHOTO_TYPE   LOCAL          # can be GLOBAL or LOCAL

#------------------------------ Check Image ----------------------------------

CHECKIMAGE_TYPE  NONE           # can be NONE, BACKGROUND, BACKGROUND_RMS,
                                # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,
                                # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,
                                # or APERTURES
CHECKIMAGE_NAME  back_gfilter.fits,obj_gfilter.fits,aper_gfilter.fits     # Filename for the check-image

#--------------------- Memory (change with caution!) -------------------------

MEMORY_OBJSTACK  3000           # number of objects in stack
MEMORY_PIXSTACK  300000         # number of pixels in stack
MEMORY_BUFSIZE   1024           # number of lines in buffer

#----------------------------- Miscellaneous ---------------------------------

VERBOSE_TYPE     NORMAL         # can be QUIET, NORMAL or FULL
HEADER_SUFFIX    .head          # Filename extension for additional headers
WRITE_XML        N              # Write XML file (Y/N)?
XML_NAME         sex.xml        # Filename for XML output
XSL_URL          file:///e:/documents/Astronomy/AAVSO/Development/AAVSOnet/srcext/test02_bsm_nh2_220723/sextractor.xsl

"""

def isBlank (myString):
    return not (myString and myString.strip())

def get_photometry(filepath):
    """
    Parse the photometry file, locate the target star's coordinates in
    the header.  Locate all the objects and place them in an astropy Table.
    With:
        ra
        dec
        peak ADU
        sky ADU
        instrumental magnitude
        magnitude err
        target ra (same for each object)
        target dec 
    
    Return None if a problem parsing the file.
    """
    target_ra = None
    target_dec = None


    with open(filepath) as fp:
        line = fp.readline()
        linenbr = 1
        # Loop through the header looking for object RA and DEC
        while line:
            if line.startswith('#'):
                line=line.strip()
                if (line.startswith("#RA= ")):
                    target_ra = line[5:]
                elif (line.startswith("#DEC= ")):
                    target_dec = line[6:]
            else:
                # Passed header
                break
            line = fp.readline()
            linenbr += 1

        if target_ra is not None and target_dec is not None:
            target_coords = SkyCoord(target_ra,target_dec,unit=(u.hour,u.deg))
            tra = target_coords.ra.deg
            tdec = target_coords.dec.deg
        else:
            logging.warning("Failed to find object coordinates in {}".format(filepath))
            return None

        # The data portion of the file contains three lines per object measured
        # A blank line or EOF terminates the data portion
        #
        # We are looking for the star closest to the target coordinates and will save
        # its values here for returning to caller
        minsep = 999999999999.0
        minvals = (None,None,None,None)

        rows=list()                 # list of dictionaries one per object in file

        nbrStars = 0
        line1 = line
        
        while line:
            if (not line1 or isBlank(line1)):
                break
            linenbr += 1
            line2 = fp.readline()
            if (not line2 or isBlank(line2)):
                break       
            linenbr += 1
            line3 = fp.readline()
            if (not line3 or isBlank(line3)):
                break
            linenbr += 1    

            sd = dict()
            try:
                sd['rax'] = float(line1[:12])
                sd['decx'] = float(line1[12:24])
                sd['xx'] = float(line1[24:32])
                sd['yx'] = float(line1[32:40])
                sd['peakx'] = float(line1[58:67])
                sd['skyx'] = float(line1[67:76])
                sd['xmag3'] = float(line2[14:21])
                sd['xerr3'] = float(line3[14:21])
                sd['tra'] = tra
                sd['tdec'] = tdec
                nbrStars += 1
            except ValueError:
                logging.exception("Floating point conversion exception:\n{}{}{}".format(line1,line2,line3))
                return None
            line1 = fp.readline()

            # Add the dictionary for th e object to the list
            rows.append(sd)

    # Create an astropy Table from the rows
    return Table(rows=rows)

def get_hastrom(filepath):
    """
    Parse the ouput of hastrom.  Return all the objects found as
    an astropy table. 
    
    Return None if a problem parsing the file.
    
    
    This from the dophot subroutine in the file dophot.f part of hphot.
            write (2,901) nn,xc,yc,fwhmx(n),fwhmy(n),sharp(n),apmax,skymod
    901     format (i5,2f8.2,3f6.2,2f9.2)
            write (2,900) (apmag(i),i=1,nap)
            write (2,900) (aperr(i),i=1,nap)
    900     format (9f7.3)
    
    This output is read by hastrom executable and the RA and DEC
    are determined.  Then the 901 line is rewritten with RA/DEC
    in columns 1-24, "nn" removed the rest of the line unchanged.
    Then the two 900 lines are rewritten unchanged.
    """

    if not path.exists(filepath):
        print(f"hastrom file {filepath} does not exist")
        return None

    with open(filepath) as fp:
        line = fp.readline()
        linenbr = 1
        # Loop through the header looking for object RA and DEC
        while line:
            if line.startswith('#'):
                # skip header
                line = fp.readline()
                continue
            else:
                # Passed header
                break
            line = fp.readline()
            linenbr += 1

        # The data portion of the file contains three lines per object measured
        # A blank line or EOF terminates the data portion

        rows=list()                 # list of dictionaries one per object in file

        nbrStars = 0
        line1 = line
        
        while line:
            if (not line1 or isBlank(line1)):
                break
            linenbr += 1
            line2 = fp.readline()
            if (not line2 or isBlank(line2)):
                break       
            linenbr += 1
            line3 = fp.readline()
            if (not line3 or isBlank(line3)):
                break
            linenbr += 1    

            sd = dict()
            try:
                sd['rax'] = float(line1[:12])
                sd['decx'] = float(line1[12:24])
                sd['xx'] = float(line1[24:32])
                sd['yx'] = float(line1[32:40])
                sd['peakx'] = float(line1[58:67])
                sd['skyx'] = float(line1[67:76])
                sd['xmag1'] = float(line2[0:7])
                sd['xmag2'] = float(line2[7:14])
                sd['xmag3'] = float(line2[14:21])
                sd['xmag4'] = float(line2[21:28])
                sd['xmag5'] = float(line2[28:35])
                sd['xmag6'] = float(line2[35:42])
                sd['xmag7'] = float(line2[42:49])
                sd['xerr1'] = float(line3[0:7])
                sd['xerr2'] = float(line3[7:14])
                sd['xerr3'] = float(line3[14:21])
                sd['xerr4'] = float(line3[21:28])
                sd['xerr5'] = float(line3[28:35])
                sd['xerr6'] = float(line3[35:42])
                sd['xerr7'] = float(line3[42:49])                
                nbrStars += 1
            except ValueError:
                print("Floating point conversion exception:\n{}{}{}".format(line1,line2,line3))
                return None
            line1 = fp.readline()

            # Add the dictionary for th e object to the list
            rows.append(sd)


    # Create an astropy Table from the rows
    return Table(rows=rows)

def haltwarn(filepath,isHalt,desc):
    # Write a HALT or WARN message
    logging.error(desc)
    with open(filepath, 'a') as hfile:
        hfile.write(desc)
    if isHalt:
        print("HALTing {}",desc)    
    else:
        print("WARNing {}",desc)   
        
    return     
    
def get_pgm_parms():
    """
    Get the programs config file and command line arguments.
    
    Return a configargparse arguments object
    """
    parser = configargparse.ArgumentParser(description='Collect image data for the pipeline HTML report')
    parser.add_argument('-c', '--my-config', required=True, is_config_file=True, help='config file path', env_var="MY_CONFIG")
    parser.add_argument('-o', '--observatory', required=True, help='observatory name', env_var ='OBSERVATORY')
    parser.add_argument('-d', '--yyyymmdd',    required=True, help='datenite process date yyyymmdd', env_var = 'YYYYMMDD')
    parser.add_argument('-D', '--procdir',     required=True, help='base directory for processed images')
    parser.add_argument('-l','--logfile',      required=True, help='Filename to log messages to')
    parser.add_argument('-v', '--verbose',                    help='Turn on verbose output', 
                        action='store_true',default=False,env_var='VERBOSE')
    parser.add_argument('-E', '--haltfile',    required=True, help='filename to signal HALT completion')
    parser.add_argument('-W', '--warnfile',    required=True, help='filename to signal WARNING completion')
    parser.add_argument('-F', '--filelist',    required=True, help='Filename for raw file list')
    parser.add_argument('-K', '--area_sextractor',    required=True, help='minimum object area (pixels)', default=7.5)
    parser.add_argument('-L', '--thresh_sextractor',    required=True, help='threshold as factor of background', default=1.5)

    args,unknown = parser.parse_known_args()

    setup_logging(args.verbose,filename=args.logfile)
    
    for arg in vars(args):
        attr = getattr(args, arg)
        if isinstance(attr, str) and arg.startswith("db_"):
            t = attr.split(";")
            if len(t) == 4:
                attr = "{};{};;{}".format(t[0],t[1],t[3])
        logging.debug ("        {}  {}".format(arg, attr))
                
    return args    

def setup_logging(verbose,filename):
    """
    Setup the logging subsystem configuration
    """
    if (verbose):
        lev = logging.DEBUG
    else:
        lev = logging.INFO
    mode = 'a'
    form = '%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s'
    # Following to log to stdout
    #logging.basicConfig(level=lev,stream=sys.stdout, filemode = mode,format=form)
    logging.basicConfig(level=lev,filename = filename, filemode = mode,format=form)
    
def get_directory(telescope,yymmdd,img_base_dir,doYyyy=False):
    """
    Construct the directory name that corresponds to the telescope and date. 
    Check that the directory exists.  
       
    If doYyyy is true, look within a yyyy directory for the yymmdd directory
    
    Input:
    telescope: name of telescope
    yymmdd: date
    img_base_dir : base path name of images
    doYyyy : boolean, true to look for directory <telescope>/yyyy/yymmdd
    
    Return the directory name 
    or None if it does not exist
    """
    yyyy = '20' + yymmdd[:2]
    if telescope.lower() == 'sro50':
        telescope = 'sro'
    else:
        telescope = telescope.lower()
    
    if (len(telescope) > 0) and (len(yymmdd) > 0):
        if not doYyyy:
            imgdir = img_base_dir + '/' + telescope + '/' + yymmdd 
            if path.exists(imgdir):
                return imgdir
            else:
                # Sometimes the yymmdd directory is within a yyyy directory
                return get_directory(telescope,yymmdd,img_base_dir,True)
        else:
            imgdir = img_base_dir + '/' + telescope + '/' + yyyy + '/' + yymmdd 
            if path.exists(imgdir):
                return imgdir
            else:
                return None
    else:
        return None    

def extract_processing(filepath,thresh,minarea):
    """
    Extract the sources from the image specified by the filepath using the specified parameters
    using sextractor.  Have sextractor write the images to the same directory as the images
    with an "se" inserted in the filename like so: YYMMDDse.9999.
    
    Read the sources found by the FORTRAN programs in the "a" file and crossmatch them with
    the source extractor sources.
    
    Return the number of sources found by sextractor, FORTRAN and the number of sextractor sources
    the crossmatch the FORTRAN sources.
    """
    was_compressed = False
    if not path.exists(filepath):
        filepath_comp = filepath + ".bz2"
        if path.exists(filepath_comp):
            logging.info(f"Uncompressing image {filepath_comp}")
            # We uncompress the file since sextractor does not work on compressed files
            with bz2.open(filepath_comp, 'rb') as f_in:
                with open(filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)   
            was_compressed = True
            
        else:
            logging.error(f"Image file {filepath} does not exist")
            return None,None,None
    output_location = filepath[:-5] + "se" + filepath[-5:] + ".csv"
    
    center,fov,exptime,filt,object,objcoord,observat,iwcs,xmax,ymax = get_image_coords(filepath)
    if center is None:
        logging.error(f"FITS file {filepath} has no WCS")
        # file was originally compressed, remove the uncompressed version we created
        if was_compressed:
            logging.info(f"Removed uncompressed image {filepath}")
            remove(filepath)
        return None,None,None    
    
    logging.info(f"Image center RA {center.ra.deg:.3f} deg, DEC {center.dec.deg:.3f} deg radius FOV {fov:.2f} deg")
    
    status,setab = call_sextractor(filepath,thresh=thresh,minarea=minarea)
    logging.info(f"Sextractor status {status} Found {len(setab)} sources in {filepath}")
    
    # Recalculate object RA/DEC - sextractor seems off
    lon, lat = iwcs.all_pix2world(setab['XWIN_IMAGE'],setab['YWIN_IMAGE'],0)
    pix2sky = SkyCoord(lon,lat,unit=(u.deg,u.deg))
    setab['ALPHAWIN_SKY_REV'] = pix2sky.ra.deg
    setab['DELTAWIN_SKY_REV'] = pix2sky.dec.deg

    # write sextractor objects to CSV file
    setab.write(output_location,overwrite=True)

    # Get the original photometry file as an astropy table
    photfile_orig = filepath[:-5] + "a" + filepath[-5:]
    if not path.exists(photfile_orig):
        logging.error(f"Original photometry file not found {photfile_orig}")
        origtab = None
    else:
        origtab = get_photometry(photfile_orig)
        if origtab is None:
            logging.error("Error parsing original photometry file {}".format(photfile_orig))
        else:
            logging.info(f"parsed original file {len(origtab)} objects")
    
    
    # write original sources to CSV file
    #origtab.write(photfile_orig + ".csv",overwrite=True)

    #xmatch = crossmatch_pixels(xy1,xy2)
    if origtab is not None:
        lenorig = len(origtab)
        if len(setab) > 10000 or len(origtab) > 10000:
            logging.error(f"Unreasonably large source counts - image suspect: SE {len(setab)} FORT {len(origtab)}")
            nbr_matched = 0
        else:
            xy1 = [ [setab['XWIN_IMAGE'][i],setab['YWIN_IMAGE'][i]] for i in range(len(setab))]
            xy2 = [ [origtab['xx'][i],origtab['yx'][i]] for i in range(len(origtab))]
            xm = crossmatch_pixels(xy1,xy2)
            maxdiff = 1.5
            nbr_matched = sum(1 for e in xm if e[0] <= maxdiff)
    else:
        logging.error(f"Original FORTRAN results not found")
        lenorig = 0
        nbr_matched = 0
        
    
    # file was originally compressed, remove the uncompressed version we created
    if was_compressed:
        logging.info(f"Removed uncompressed image {filepath}")
        remove(filepath)
         
    return len(setab),lenorig,nbr_matched


def get_plate_scale(wcs):
    """
    Return the image plate scale in arc seconds
    """
    """
    lon, lat = wcs.wcs_pix2world(0, 0, 1)
    p1 = SkyCoord(lon,lat,unit=(u.deg,u.deg))
    lon, lat = wcs.wcs_pix2world(0, 1, 1)
    p2 = SkyCoord(lon,lat,unit=(u.deg,u.deg))    
    return p1.separation(p2).arcsec
    """
    scale = utils.proj_plane_pixel_scales(wcs)
    return scale[0] * 3600
"""

Main processing - testing
Normal processing calls extract_processing directly
"""

if __name__ == "__main__":
    
    """
    THIS SECTION PROCESSES AN ENTIRE DATENITE OF IMAGES
    # Handle command line arguments
    # setup logging    
    args = get_pgm_parms()    
    
    logging.info("Begin execution, observatory {}, date {}".format(args.observatory,args.yyyymmdd))
    logging.info(_svn_id_)        
    
    # Get full path to rawimages.txt file and make sure it exists
    imagedir = get_directory(args.observatory,args.yyyymmdd[2:],args.procdir)
    if imagedir is not None:
        filepath = path.join(imagedir,args.filelist)
        if not path.exists(filepath):
            filepath = None
    if filepath is None:
        msg = "Cannot locate {} file list for {} on {}".format(args.filelist,args.observatory,args.yyyymmdd[2:])
        logging.critical(msg)
        haltwarn(args.haltfile,True,msg)
        sys.exit(-1)    
        
    rowlist = []
    with open(filepath,newline='') as csvfile:
        reader = csv.DictReader(csvfile,delimiter='\t')
        # 
        # In the following rowdict contains image attributes passed from the calibration step and come
        # primarily from the FITS header.  rowdict will be updated with sextractor results and the file
        # overwritten.
        for img in reader:
            rowlist.append(img)
            if img['IMAGETYP'] != 'light':
                # Skip everything except science images
                logging.info("=== skipping  {}, {}, {}".format(args.observatory,args.yyyymmdd[2:],img['FILENAME']))        
                continue
            logging.info("=== processing  {}, {}, {} aka {}".format(args.observatory,args.yyyymmdd[2:],img['FILENAME'],img['NEW-FILENAME']))
            
            # extract and match sources in image         
            fitspath = path.join(imagedir,img['NEW-FILENAME'])
            srcs_sex,srcs_fort,xmatch_sex = extract_processing(fitspath,args.area_sextractor,args.thresh_sextractor)
            logging.info(f"Source extracted image {fitspath} {srcs_sex},{srcs_fort},{xmatch_sex}")
    exit()
    """
    
    """
    Process a single image through source extractor
    """
    lev = logging.DEBUG
    mode = 'a'
    form = '%(asctime)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s'    
    logging.basicConfig(level=lev,stream=sys.stdout, filemode = mode,format=form)
    if (len(sys.argv) != 4) and (len(sys.argv) != 5):
        logggin.info("format: {} <observatory> <YYYYMMDD> <science image filename> [<hastrom filename>]".format(sys.argv[0]))
        sys.exit()
        
    ob = Observatories(sys.argv[1]).get_telescope_datenite(sys.argv[2])
    if ob is None:
        logging.error(f"Exiting, invalid observatory: {sys.argv[1]} {sys.argv[2]}")
        sys.exit()
    else:
        gain = ob['gain']
        saturation = ob['linearitylimit']
    
    img = sys.argv[3]    
        
    # get image WCS
    center,fov,exptime,filt,object,objcoord,observat,iwcs,xmax,ymax = get_image_coords(img)    
    
    #use value from web service call
    plate_scale = get_plate_scale(iwcs)
    logging.info(f"Found observatory: for {sys.argv[1]} {sys.argv[2]} with gain {gain}, linearitylimit {saturation}, plate scale: {plate_scale:.2f}")
    
    # call source extractor
    status,setab = call_sextractor2(img, gain=gain, saturation=saturation,plate_scale=plate_scale)
    logging.info(f"sextractor status {status}")


    # write sextractor objects to CSV file
    output_location = img[:6] + "se" + img[6:] + ".csv"    
    setab.write(output_location,overwrite=True)
    logging.info(f"se len {len(setab)}")

    
    """
    Now we evaluate the sextractor results by comparing to other catalogs for this
    region.  
    
    First, we have the FORTRAN programs hphot/hastrom that product a catalog
    of objects with both pixel and sky coordinates.  We match of pixel coordinates.
     
    Read the hastrom output file if provided
    """
    maxdist = 1.0
    sejoin = None
    if (len(sys.argv) == 5):
        hastrom = get_hastrom(sys.argv[4])
        if hastrom is not None:
            # write hastrom out as CSV file
            output_location = sys.argv[4] + ".csv"    
            hastrom.write(output_location,overwrite=True) 
            logging.info(f"hastrom len {len(hastrom)}")           

            xy1 = [ [setab['X_IMAGE'][i],setab['Y_IMAGE'][i]] for i in range(len(setab))]
            xy2 = [ [hastrom['xx'][i],hastrom['yx'][i]] for i in range(len(hastrom))]
            xm = crossmatch_pixels(xy1,xy2)
            logging.info(f"hastrom crossmath len {len(xm)}")
            
            # Join the source extractor and hastrom tables on the crossmatch
            dlist = [xm[i][0] for i in range(len(xm))]
            klist = [xm[i][1] for i in range(len(xm))]
            setab['cdist'] = dlist
            setab['hkey'] = klist
            hastrom['hkey'] = [i for i in range(0,len(hastrom))]
            setab.add_column("Y", name='hmatch', index=0) 
            for i in range(len(setab)):
                if setab['cdist'][i] > maxdist:
                    setab['hmatch'][i] = 'N'  
                    # remove join key to indicate not matched when we do outer join  
                    setab['hkey'][i] = -9999        
            
            sejoin = join(setab,hastrom,join_type='outer')
            sejoin['hmatch'].fill_value=' '
            sejoin = sejoin.filled()
            se_matched = sejoin[sejoin['hmatch'] == 'Y']
            se_not_matched  = sejoin[sejoin['hmatch'] == 'N']
            hastrom_not_matched = sejoin[sejoin['hmatch'] == ' ']
            #sep_contraint = setab['cdist'] < maxdist
            #c_matches = setab[sep_contraint]
            logging.info(f"matches setab and hastrom within {maxdist} pixels:")
            logging.info(f"     se <= dist     {len(se_matched)}")
            logging.info(f"     se >  dist     {len(se_not_matched)}")
            logging.info(f"     hastrom > dist {len(hastrom_not_matched)}")
            
            # Save the joined table
            output_location = img[:6] + "sj" + img[6:] + ".csv"    
            sejoin.write(output_location,overwrite=True)
            logging.info(f"wrote se joined hastrom to {output_location}")            
    """
    Next we match GAIA objects for this image using sky coordinates provided we have a
    plate solved images.
    """
    if iwcs is not None:
        limiting_mag = 18.0
        gaia_cat = get_gaia(center.ra.deg,center.dec.deg,fov * 1.1,limiting_mag,iwcs,xmax,ymax)        
        if sejoin is None:
            # we did not match hastrom
            sejoin = setab
            sejoin.add_column("N", name='hmatch', index=0) 
        
        # Create RA/DEC for hastrom not matched
        for i in range(len(sejoin)):
            if sejoin['hmatch'][i] == ' ':
                # this hastrom has no se, use hastrom RA/DEC
                sejoin['ALPHA_SKY'] = sejoin['rax']
                sejoin['DELTA_SKY'] = sejoin['decx']
        
        sejoin.sort('NUMBER')
        c = SkyCoord(ra=setab['ALPHA_SKY']*u.degree, dec=setab['DELTA_SKY']*u.degree)
        catalog = SkyCoord(ra=gaia_cat['ra'], dec=gaia_cat['dec'])
        idx, d2d, d3d = c.match_to_catalog_sky(catalog)
        maxadist = maxdist * plate_scale
        sep_constraint = d2d < (maxadist * u.arcsecond)
        c_matches = c[sep_constraint]
        logging.info(f"setab len {len(setab)} # match GAIA bright {limiting_mag} {len(c_matches)}")                                                                                    
        logging.info(f"nbr matches between setab and GAIA within {maxadist:.2f} arcsec is {len(c_matches)} out of {len(setab)}")
        # Now we extend the setab with the GAIA match
        if sejoin is None:
            # we did not match hastrom
            sejoin = setab
        sejoin['gdist'] = d2d * 3600    # deg to arcsec
        sejoin['g_key'] = idx
        # rename GAIA columns to avoid hit during join
        c = list(gaia_cat.columns)
        for orig in c:
            gaia_cat.rename_column(orig,"g_"+orig)
        gaia_cat['g_key'] = [i for i in range(0,len(gaia_cat))]
        segaia = join(sejoin,gaia_cat)
        segaia.add_column("Y", name='gmatch', index=0) 
        for i in range(len(segaia)):
            if segaia['gdist'][i] > maxadist:
                segaia['gmatch'][i] = 'N'    
        
        # Add sexagesimal sky coordinates
        segaia.add_column("", name='Sexag_Sky', index=2) 
        coord = SkyCoord(segaia['ALPHA_SKY'],segaia['DELTA_SKY'],unit='deg')
        segaia['Sexag_Sky'] = coord.to_string(style='hmsdms',sep=":")

        output_location = img[:6] + "sj" + img[6:] + ".csv"    
        segaia.write(output_location,overwrite=True)
        logging.info(f"wrote se joined gaia to {output_location}")
        
        gcat = join(gaia_cat,sejoin)
        scoords =  SkyCoord(gcat['g_ra'],gcat['g_dec'],unit=u.deg)
        pcoords = utils.skycoord_to_pixel(scoords, iwcs) 
        gx = pcoords[0] + 1.0
        gy = pcoords[1] + 1.0       
        gcat.add_column("Y", name='gmatch', index=0) 
        gcat.add_column(gx, name='g_x', index=3) 
        gcat.add_column(gy, name='g_y', index=4) 
        for i in range(len(gcat)):
            if (gcat['gdist'][i] > maxadist):
                gcat['gmatch'][i] = 'N'    
        # Since we get a slightly larger sky area from GAIA than the image
        # we want to ignore the "extra" GAIA objects, setting gmatch to blank
        for i in range(len(gcat)):
            if (1 <= gcat['g_x'][i] <= xmax) and (1 <= gcat['g_y'][i] <= ymax):
                continue
            else:
                gcat['gmatch'][i] = ' '          
        output_location = img[:6] + "gcat" + img[6:] + ".csv"    
        gcat.write(output_location,overwrite=True)
        logging.info(f"wrote gaia joined se to {output_location}")                                