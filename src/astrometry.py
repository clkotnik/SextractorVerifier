'''
Created on Aug 7, 2023

@author: clkot
'''

if __name__ == '__main__':
    pass

#!/usr/bin/python3 
"""
Created on Jul 10, 2021

Based on code from Matt Craig at
https://github.com/mwcraig/msumastro/blob/master/msumastro/header_processing/astrometry.py

CLKotnik 
Call astrometry.net "solve-field" command line program.

CLKotnik 230228
We recently were digging into an issue with one of these images in VPhot.  The crux of the issue seems to be CRPIX1 & 2.
Astronmetry.net is placing zero in these headers in spite of this code asking for the center of the image to be used.
This seems to be an issue with version 0.78 of Astrometry.net.  

A workaround is implemented here.  We no longer specify --crpix-center.  Instead, we calculate the center of the image
pixel coordinates from NAXIS1 and 2 in the header since we already read it here for other reasons.
  
CLKotnik 230301
Workaround from 230228 removed.

This version is setup to work with Astrometry.net ver 0.93.  It was compiled from source and installed to /usr/local/astrometry/bin.
It is expected that this version will be used to test and will replace the original version when testing completes.
  


"""
import logging
import subprocess
import sys
from os import path, remove, rename, stat, getuid
from glob import glob
import gzip
import bz2
import shutil

import math
import tempfile
from textwrap import dedent
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import wcs

import warnings

_svn_id_ = "$Id: astrometry.py 1343 2023-03-02 15:06:55Z aavsonet $"


# suppress runtime and astropy warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings('ignore', category=wcs.FITSFixedWarning)
warnings.filterwarnings('ignore', category=fits.column.VerifyWarning)
warnings.filterwarnings('ignore', category=fits.card.VerifyWarning)




__all__ = ['call_astrometry', 'add_astrometry']

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
#log_level = logging.DEBUG

def call_astrometry(filename, sextractor=False,
                    custom_sextractor_config=False, 
                    no_plots=True, minimal_output=True,
                    save_wcs=False, verify=None,
                    ra_dec=None, radius=None,
                    overwrite=False,
                    wcs_reference_image_center=True,
                    odds_ratio=None,
                    astrometry_config=None,
                    additional_args=None,
                    solve_bin=None):
    """
    Wrapper around astrometry.net solve-field.

    Parameters
    ----------
    sextractor : bool or str, optional
        ``True`` to use `sextractor`, or a ``str`` with the
        path to sextractor.
    custom_sextractor_config : bool, optional
        If ``True``, use a sexractor configuration file customized for BSM
        images.
    no_plots : bool, optional
        ``True`` to suppress astrometry.net generation of
        plots (pngs showing object location and more)
    minimal_output : bool, optional
        If ``True``, suppress, as separate files, output of: WCS
        header, RA/Dec object list, matching objects list, but see
        also `save_wcs`
    save_wcs : bool, optional
        If ``True``, save WCS header even if other output is suppressed
        with `minimial_output`
    verify : str, optional
        Name of a WCS header to be used as a first guess
        for the astrometry fit; if this plate solution does not work
        the solution is found as though `verify` had not been specified.
    ra_dec : list or tuple of float
        (RA, Dec); also limits search radius to 2 degree unless radius
        also is specified.
    radius : float, optional
        Search radius around ra_dec which must also be specified.
    overwrite : bool, optional
        If ``True``, perform astrometry even if astrometry.net files from a
        previous run are present.
    wcs_reference_image_center :
        If ``True``, force the WCS reference point in the image to be the
        image center.
    odds_ratio : float, optional
        The odds ratio to use for a successful solve. Default is to use the
        default in `solve-field`.
    astrometry_config : str, optional
        Name of configuration file to use for SExtractor.
    additional_args : str or list of str, optional
        Additional arguments to pass to `solve-field`
    """
    # Allow override the binary filename
    if solve_bin is None:
        solve_field = ["solve-field"]
    else:
        solve_field = [solve_bin]
        
    option_list = []

    option_list.append("--obj 100")

    if additional_args is not None:
        if isinstance(additional_args, str):
            add_ons = [additional_args]
        else:
            add_ons = additional_args
        option_list.extend(add_ons)

    if isinstance(sextractor, str):
        option_list.append("--source-extractor-path " + sextractor)
    elif sextractor:
        option_list.append("--use-source-extractor")

    if no_plots:
        option_list.append("--no-plot")

    if minimal_output:
        option_list.append("--corr none --rdls none --match none")
        if not save_wcs:
            option_list.append("--wcs none")

    if ra_dec is not None:
        if radius is None:
            radius = 2.0
        option_list.append("--ra %s --dec %s " % ra_dec)
        option_list.append("--radius %s " % radius)

    if overwrite:
        option_list.append("--overwrite")

    if wcs_reference_image_center:
        option_list.append("--crpix-center")

    options = " ".join(option_list)

    solve_field.extend(options.split())

    if custom_sextractor_config:
        tmp_location = tempfile.mkdtemp()
        param_location = path.join(tmp_location, 'default.param')
        config_location = path.join(tmp_location, 'feder.config')
        config_contents = SExtractor_config.format(param_file=param_location)
        with open(config_location, 'w') as f:
            f.write(config_contents)
        with open(param_location, 'w') as f:
            contents = """
                X_IMAGE
                Y_IMAGE
                MAG_AUTO
                FLUX_AUTO
            """
            f.write(dedent(contents))

        additional_solve_args = [
            '--source-extractor-config', config_location,
            '--x-column', 'X_IMAGE',
            '--y-column',  'Y_IMAGE',
            '--sort-column', 'MAG_AUTO',
            '--sort-ascending'
        ]

        solve_field.extend(additional_solve_args)
    else:
        tmp_location = None
        
    if odds_ratio is not None:
        solve_field.append('--odds-to-solve')
        solve_field.append(odds_ratio)

    if astrometry_config is not None:
        solve_field.append('--config')
        solve_field.append(astrometry_config)

    # kludge to handle case when path of verify file contains a space--split
    # above does not work for that case.

    if verify is not None:
        if verify:
            solve_field.append("--verify")
            solve_field.append("%s" % verify)
        else:
            solve_field.append("--no-verify")

    solve_field.extend([filename])
    logger.debug(' '.join(solve_field))
    """
    try:
        solve_field_output = subprocess.check_outputsubprocess.check_output(solve_field,
                                                     stderr=subprocess.STDOUT,
                                                     universal_newlines=False).decode("utf-8")
        print("solve_field_output",type(solve_field_output),len(solve_field_output))
        print(solve_field_output)
        log_level = logging.DEBUG
        logging.info("SUBPROCESS OUTPUT >>>>>>>\n{}".format(solve_field_output))
        logging.info("<<<<<<<<SUBPROCESS OUTPUT")
        return_status = 0
    except subprocess.CalledProcessError as e:
        return_status = e.returncode
        solve_field_output = 'Output from astrometry.net:\n' + str(e.output)
        log_level = logging.WARN
        logger.warning('Adding astrometry failed for %s', filename)
        raise e
    """
    try:
        completion = subprocess.run(solve_field,
                                     stdout = subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
        solve_field_output = str(completion.stdout)
        solve_field_error = str(completion.stderr)
        logging.debug("call_astrometry return code: {}".format(completion.returncode))
        logging.debug(">>>>>>>>>>>ERROR>>>>>>>>>>>>\n{}\n<<<<<<<<<<<<<<<<<<<<<<<<<\n".format(solve_field_error))
        logging.debug(">>>>>>>>>>>OUTPUT>>>>>>>>>>>>\n{}\n<<<<<<<<<<<<<<<<<<<<<<<<<\n".format(solve_field_output))
        return_status = completion.returncode
    except subprocess.CalledProcessError as e:
        return_status = e.returncode
        logger.exception('call_astrometry failed for %s with exception', filename)
        if tmp_location is not None:
            shutil.rmtree(tmp_location)
        raise e
    
    #logger.log(log_level, solve_field_output)
    if return_status != 0:
        logging.info("call_astrometry: solve-field return code {}".format(return_status))
    else:
        logging.debug("call_astrometry: solve-field return code {}".format(return_status))
    if tmp_location is not None:
        shutil.rmtree(tmp_location)
    return return_status


def add_astrometry(filename, overwrite=False, 
                   ra_dec=None, radius=None,
                   note_failure=False, save_wcs=False,
                   replace_wcs = False,
                   verify=None, try_builtin_source_finder=False,
                   custom_sextractor=False,
                   odds_ratio=None,
                   astrometry_config=None,
                   plate_scale=None,
                   avoid_pyfits=False,
                   no_source_extractor=False,
                   solve_field_args=None):
    """Add WCS headers to FITS file using astrometry.net

    Parameters
    ----------
    overwrite : bool, optional
        Set ``True`` to overwrite the original file. If `False`,
        the file astrometry.net generates is kept.

    ra_dec : list or tuple of float
        (RA, Dec); also limits search radius to 2 degree unless radius
        also is specified.  If not specified, the FITS header RA/DEC
        is used.

    radius : float, optional
        Search radius around ra_dec which must also be specified or
        obtainable from FITS header.

    note_failure : bool, optional
        If ``True``, create a file with extension "failed" if astrometry.net
        fails. The "failed" file contains the error messages genreated by
        astrometry.net.

    try_builtin_source_finder : bool
        If true, try using astrometry.net's built-in source extractor if
        sextractor fails.

    save_wcs :
    
    replace_wcs: bool, if false and image already has WCS, nothing is done
    
    verify :
        See :func:`call_astrometry`

    plate_scale : image plate scale in arcsec/pix. If specified as a float
        the range 80%-120% is searched.  If specified as a list or tuple of
        floats, the range from the first to second is searched.

    avoid_pyfits : bool
        Add arguments to solve-field to avoid calls to pyfits.BinTableHDU.
        See https://groups.google.com/forum/#!topic/astrometry/AT21x6zVAJo

    Returns
    -------
    bool
        ``True`` on success.

    Notes
    -----

    Tries a couple strategies before giving up: first sextractor,
    then, if that fails, astrometry.net's built-in source extractor.

    It also cleans up after astrometry.net, keeping only the new FITS
    file it generates, the .solved file, and, if desired, a ".failed" file
    for fields which it fails to solve.

    For more flexible invocation of astrometry.net, see :func:`call_astrometry`
    """
    # Separate filename and extension 
    # also determine if image is compressed
    base, ext = path.splitext(filename)
    compressed = None
    if ext.lower() in [".gz",".bz2"]:
        compressed = ext[1:]
        base,ext = path.splitext(base)
    elif ext.lower() == ".zip":
        logging.error("ZIP compression not supported, only gzip and bzip2")
        return False
    logging.debug("add_astrometry on filename {}, base {}, ext {}, compressed {}".format(filename,base,ext,compressed))
    
    # If ra and dec not provided, see if they can be found in FITS header
    try:
        hdulist = fits.open(filename, mode='readonly', ignore_missing_end=True)
        hdr = hdulist[0].header
        if 'RA' not in hdr or 'DEC' not in hdr:
            logging.warning("add astrometry: do not have RA/DEC in header, plate solving skipped".format(filename))
            return False
        radec = "{} {}".format(hdr['RA'],hdr['DEC'])
        objcoord = SkyCoord(radec ,unit=(u.hourangle,u.deg))
        binning = float(hdr['XBINNING']) if 'XBINNING' in hdr else None
        pixsiz = float(hdr['XPIXSZ'])   if 'XPIXSZ' in hdr else None
        focallen = float(hdr['FOCALLEN']) if 'FOCALLEN' in hdr else None
        if focallen < 0.1:          # zero in focallen at times
            focallen = None
        logging.debug("FITS hdr objcoord {}, binning {}, pixsiz {}, focallen {}".format(objcoord,binning,pixsiz,focallen))
        if pixsiz is None or focallen is None:
            logging.debug("Cannot determine plate scale from FITS header")
            hdr_scale = None
        else:
            hdr_scale = 180.0/math.pi*(math.atan(pixsiz/1000.0/focallen))*3600.0
            logging.debug("Plate scale from FITS header: {}".format(hdr_scale))            
        fitswcs = wcs.WCS(hdr)
        """
        naxis1 = float(hdr['NAXIS1']) if 'NAXIS1' in hdr else 0.0
        naxis2 = float(hdr['NAXIS2']) if 'NAXIS2' in hdr else 0.0
        crpix1 = naxis1/2.0
        crpix2 = naxis2/2.0
        crpix_opt = f"--crpix-x {crpix1} --crpix-y {crpix2}"
        """
        #fitswcs.has_celestial        
        hdulist.close()
    except Exception:
        logging.exception("Failed to parse FIT header from {}".format(filename))
        return False
    
    # Do we need to plate solve?
    if fitswcs.has_celestial and not replace_wcs:
        logging.warning("add astrometry: {} already has WCS, plate solving skipped".format(filename))
        return True

    # Use RA/DEC from FITS header if available and not specified by caller
    if ra_dec is None:
        ra_dec = (objcoord.ra.deg,objcoord.dec.deg)

    # All are in arcsec per pixel, values are approximate
    # Plate scale is specified for binning used for this image
    if not plate_scale is None:
        if isinstance(plate_scale, float):
            scale_options = ("--scale-low {low} --scale-high {high} "
                             "--scale-units arcsecperpix".format(low=0.8*plate_scale, high=1.2 * plate_scale))
        elif isinstance(plate_scale, list) or isinstance(plate_scale, tuple):
            scale_options = ("--scale-low {low} --scale-high {high} "
                             "--scale-units arcsecperpix".format(low=plate_scale[0], high=plate_scale[1]))
        else:
            logging.warning("plate scale must be float of list/tuple of floats")
    # Header values are not reliable.  Better to skip this and take the processing time hit.
    #elif hdr_scale is not None:
    #    scale_options = ("--scale-low {low} --scale-high {high} "
    #                     "--scale-units arcsecperpix".format(low=0.8*hdr_scale*binning, high=1.2 * hdr_scale*binning))
    else:
        scale_options = ''

    if avoid_pyfits:
        pyfits_options = '--no-remove-lines --uniformize 0'
    else:
        pyfits_options = ''

    additional_opts = ' '.join([scale_options,
                                pyfits_options])

    if solve_field_args is not None:
        additional_opts = additional_opts.split()
        additional_opts.extend(solve_field_args)
        
        
    logger.info('BEGIN add_astrometry on {}, {}, {}, addl {}'.format(filename,scale_options,ra_dec,additional_opts))
    try:
        logger.debug('About to call call_astrometry')
        solved_field = (call_astrometry(filename,
                                        sextractor=not no_source_extractor,
                                        ra_dec=ra_dec, radius=radius,
                                        save_wcs=save_wcs, verify=verify,
                                        custom_sextractor_config=custom_sextractor,
                                        odds_ratio=odds_ratio,
                                        astrometry_config=astrometry_config,
                                        ## --crpix-center is not working.  Workaround is to turn it off
                                        ## and specify --crpix-x and -y in additional_opts
                                        ## calculated from NAXIS1 and 2 in FITS header
                                        wcs_reference_image_center=True,
                                        additional_args=additional_opts,
                                        solve_bin ="/usr/local/astrometry/bin/solve-field")
                        == 0)
    except subprocess.CalledProcessError as e:
        logger.exception('Failed with exception')
        failed_details = e.output
        solved_field = False

    """
    if (not solved_field) and try_builtin_source_finder:
        log_msg = 'Astrometry failed using sextractor, trying built-in '
        log_msg += 'source finder'
        logger.info(log_msg)
        try:
            solved_field = (call_astrometry(filename, ra_dec=ra_dec,
                                            overwrite=True,
                                            save_wcs=save_wcs, verify=verify)
                            == 0)
        except subprocess.CalledProcessError as e:
            failed_details = e.output
            solved_field = False
    """

    if solved_field and path.exists(base+'.solved'):
        logger.info('add_astrometry: plate solve succeeded for file {}'.format(filename))
    else:
        solved_field = False            # in case command returned 0, but solved file not creating indicating failure
        logger.warning('add_astrometry: plate solve failed for file {}'.format(filename))

    if overwrite and solved_field:
        logger.info('add_astrometry: updating original file with image with wcs')
        try:
            # Since Pinpoint sets this we set it here
            with fits.open(base + '.new', 'update') as f:
                for hdu in f:
                    hdu.header['PLTSOLVD'] = True           
            # Now rename/compress the new version just created
            if compressed is None:
                rename(base + '.new', filename)
            elif compressed == 'gz':
                logger.debug('gzip compressing {} to {}'.format(base + '.new',filename))
                with open(base + '.new', 'rb') as f_in:
                    with gzip.open(filename, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)       
                remove(base + '.new')         
            elif compressed == 'bz2':
                logger.debug('bz2 compressing {} to {}'.format(base + '.new',filename))
                with open(base + '.new', 'rb') as f_in:
                    with bz2.open(filename, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)                
                remove(base + '.new')         
            else:
                logging.error('Logic error')
        except OSError as e:
            logger.error(e)
            return False
    
    # whether we succeeded or failed, clean up
    try:
        remove(base + '.axy')
    except OSError:
        pass

    if solved_field:
        try:
            remove(base + '-indx.xyls')
            remove(base + '.solved')
        except OSError:
            pass
        
    # If image was compressed, the uncompressed file hangs around
    # find and remove it
    uncs = glob('/tmp/tmp.uncompressed.*')
    # Iterate over the list of files and remove individually
    myuid = getuid()
    for file in uncs:
        if stat(file).st_uid == myuid:
            remove(file) 
    
    """
    logger.info("CLEANUP OF FILES DISABLED*******************")
    """
    
    
    
    logger.info('END add_astrometry for {} with status {}'.format(filename,solved_field))
    return solved_field


SExtractor_config = """
# Configuration file for SExtractor 2.19.5 based on default by EB 2014-11-26
#

# modification was to change DETECT_MINAREA and turn of filter convolution

#-------------------------------- Catalog ------------------------------------

PARAMETERS_NAME  {param_file}  # name of the file containing catalog contents

#------------------------------- Extraction ----------------------------------

DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)
DETECT_MINAREA   15              # min. # of pixels above threshold
DETECT_THRESH    1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
ANALYSIS_THRESH  1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2

FILTER           N              # apply filter for detection (Y or N)?
FILTER_NAME      default.conv   # name of the file containing the filter

DEBLEND_NTHRESH  32             # Number of deblending sub-thresholds
DEBLEND_MINCONT  0.005          # Minimum contrast parameter for deblending

CLEAN            Y              # Clean spurious detections? (Y or N)?
CLEAN_PARAM      1.0            # Cleaning efficiency

MASK_TYPE        CORRECT        # type of detection MASKing: can be one of
                                # NONE, BLANK or CORRECT

#------------------------------ Photometry -----------------------------------

PHOT_APERTURES   10              # MAG_APER aperture diameter(s) in pixels
PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius>
PHOT_PETROPARAMS 2.0, 3.5       # MAG_PETRO parameters: <Petrosian_fact>,
                                # <min_radius>

SATUR_LEVEL      50000.0        # level (in ADUs) at which arises saturation
SATUR_KEY        SATURATE       # keyword for saturation level (in ADUs)

MAG_ZEROPOINT    0.0            # magnitude zero-point
MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)
GAIN             0.0            # detector gain in e-/ADU
GAIN_KEY         GAIN           # keyword for detector gain in e-/ADU
PIXEL_SCALE      1.0            # size of pixel in arcsec (0=use FITS WCS info)

#------------------------- Star/Galaxy Separation ----------------------------

SEEING_FWHM      1.2            # stellar FWHM in arcsec
STARNNW_NAME     default.nnw    # Neural-Network_Weight table filename

#------------------------------ Background -----------------------------------

BACK_SIZE        64             # Background mesh: <size> or <width>,<height>
BACK_FILTERSIZE  3              # Background filter: <size> or <width>,<height>

BACKPHOTO_TYPE   GLOBAL         # can be GLOBAL or LOCAL

#------------------------------ Check Image ----------------------------------

CHECKIMAGE_TYPE  NONE           # can be NONE, BACKGROUND, BACKGROUND_RMS,
                                # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,
                                # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,
                                # or APERTURES
CHECKIMAGE_NAME  check.fits     # Filename for the check-image

#--------------------- Memory (change with caution!) -------------------------

MEMORY_OBJSTACK  3000           # number of objects in stack
MEMORY_PIXSTACK  300000         # number of pixels in stack
MEMORY_BUFSIZE   1024           # number of lines in buffer

#----------------------------- Miscellaneous ---------------------------------

VERBOSE_TYPE     NORMAL         # can be QUIET, NORMAL or FULL
HEADER_SUFFIX    .head          # Filename extension for additional headers
WRITE_XML        N              # Write XML file (Y/N)?
XML_NAME         sex.xml        # Filename for XML output

"""

def setup_logging(verbose,filename=None):
    """
    Setup the logging subsystem configuration
    """
    if (verbose):
        lev = logging.DEBUG
    else:
        lev = logging.INFO
    mode = 'a'
    form = '%(asctime)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s'
    if (filename is None):
        # Following to log to stdout
        logging.basicConfig(level=lev,stream=sys.stdout, filemode = mode,format=form)
    else:
        logging.basicConfig(level=lev,filename = filename, filemode = mode,format=form)

def usage(argv0):
    print("Plate solve image with astrometry.net")
    print("Usage: {} [-v] [-f] <fits file> [<fits file> ...]".format(argv0))
    print("   -v for verbose output")
    print("   -f to force existing WCS to be overwritten")
    exit()
"""

Main processing - testing
"""

if __name__ == "__main__":
    # Get command line args
    if len(sys.argv) >= 2:
        alist = sys.argv[1:]
        overwrite_wcs = False
        verbose = False
        
        while alist[0] == "-f" or alist[0] == "-v":
            if alist[0] == "-f":
                overwrite_wcs = True
            else:
                verbose = True
            if len(alist) >= 2:
                alist = alist[1:]
            else:
                usage(sys.argv[0])
                exit()        
            
        # Executed from command line
        fitsfiles = alist
    else:
        usage(sys.argv[0])
        exit()        
    
    setup_logging(verbose)
    
    logging.debug("Begin")
        
    for fitsfile in fitsfiles:
        if not path.exists(fitsfile):
            logging.error("FITS file not found {}".format(fitsfile))
        else:
            add_astrometry(fitsfile,overwrite=True
                           ,replace_wcs=overwrite_wcs
                           ,custom_sextractor=True)
