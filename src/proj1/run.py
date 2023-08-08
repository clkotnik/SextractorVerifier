'''
Created on Aug 7, 2023

@author: clkot
'''

if __name__ == '__main__':



#!/usr/bin/python3
'''
Created on Mar 23, 2020

@author: CLKotnik

This module handles the AAVSOnet pipeline processing to select a night's images, process
all calibration files collected and setup the science images for processing with the
most current calibration master files. 

Apr 2021
Phase 3 is a major release.  The entire pypline is now python.  As part of that, this
module now takes on this functionality:
    - update the image_info table.  Initially, this table is in a different DB than cal_info.
      This may change, but requires changes to the RoM web service to do so.  The new logic will
      replace all existing rows for a night to avoid the inaccurate data resulting from the
      current logic.
    - call the thumbnail module
    - call the astrometry and photometry fortran programs
    - compress the images
    - there are a couple of minor changes to the config file so that a single config file
      can support the entire pypline
    - the logging now goes to a file
    
    
Nov 2020
Phase 2 is a major release.  It contains this additional functionality:

    Calibration master creation and image calibration is performed with python routines
    rather than IRAF.  This provides a number of enhanced features:
        - darks no longer need be at fixed exposure durations - just a reasonable spread
        - image duplication will be minimized
        - the a9999.fits files are no longer created

    TO DO: When masterdir option is used, the HTML report incorrectly identifies the cals
    in use.  See for example bsm_nh2/201109.
    
    Handle bad FITS images and do a more controlled HALT.
    
    Do not include raw files with names like "*oldzip"
       
    Provide explicit HALT message when web service to get telescopes is  not available  
    
CLKotnik
Jul 1 2023
Add configuration option to enable/disable FITS header update of OBJCTRA/DEC.  If
selected, call for the update after astrometry.net is run.

Also, add scriptsdir and ftpdir to config file options as documentation so this
program has a complete list.  These are used elsewhere.
    
'''
import subprocess
import sys
import os
import shutil
import logging
import configargparse
from collections import Counter
from datetime import date,datetime
from astropy.io import fits
from astropy.nddata.blocks import block_reduce,block_replicate
import bz2

from pypline.filesys import FileSys
from pypline.filelist import FileList
from pypline.observatory import Observatories
from pypline.database import CalInfo,ImageInfo
from pypline.imgproc_py import ImgProc
from pypline.thumbnail import make_all_tn
from pypline.astrometry import add_astrometry
from pypline.sextractor import extract_processing
# suppress runtime and astropy warnings
import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings('ignore', category=fits.column.VerifyWarning)
warnings.filterwarnings('ignore', category=fits.card.VerifyWarning)


_svn_id_ = "$Id: pypline_calibration.py 1457 2023-07-19 21:31:16Z kotnik $ :"
######################################################################
# This section contains parameters that control the creation and
# application of master files.  The are not input from a config file,
# but might be in the future is the need arises.
######################################################################

# The number of days back and forward to go to get images to combine for a master
_FLAT_DAY_RANGE = [2,2]

# The minimum number of raw images needed to make a master
_FLAT_MIN_RAW_PER_MASTER = 4

######################################################################

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

def get_pgm_parms():
    """
    Get the programs config file and command line arguments.
    
    Return a configargparse arguments object
    """
    parser = configargparse.ArgumentParser(description='Perform calibration file processing for AAVSOnet pipeline')
    parser.add_argument('-c', '--my-config', required=True, is_config_file=True, help='config file path', env_var="MY_CONFIG")
    parser.add_argument('-o', '--observatory', required=True, help='observatory name', env_var ='OBSERVATORY')
    parser.add_argument('-d', '--yyyymmdd',    required=True, help='datenite process date yyyymmdd', env_var = 'YYYYMMDD')
    parser.add_argument('-D', '--procdir',     required=True, help='base directory for processed images')
    parser.add_argument('-m', '--homedir',     required=True, help='home directory')
    parser.add_argument('-R', '--rawdir',      required=True, help='base directory for raw images')
    parser.add_argument('-t', '--thumbdir',    required=True, help='base directory for thumbnails')
    parser.add_argument('-u', '--scriptsdir',  required=True, help='base directory for scripts')
    parser.add_argument('-w', '--ftpdir',  required=True, help='base directory for anonymous ftp')
    parser.add_argument( '-M', '--masterdir',                 help='override calibration master directory')
    parser.add_argument('-b', '--no-bias'   ,                 help='Do not process raw bias and dark', action='store_false',default=True)
    parser.add_argument('-s', '--no-science'  ,               help='Do not process raw science', action='store_false',default=True)
    parser.add_argument('-f', '--no-flats'  ,                 help='Do not process raw flats', action='store_false',default=True)
    parser.add_argument('-i', '--no-thumb'  ,                 help='Do not generate thumbnails', action='store_false',default=True)
    parser.add_argument('-j', '--no-photom'  ,                help='Do not run photometry/astrometry', action='store_false',default=True)
    parser.add_argument('-k', '--no-dbinfo'  ,                help='Do not update the image_info table', action='store_false',default=True)
    parser.add_argument('-Z', '--no-wcs'  ,                   help='Do not run WCS solving', action='store_false',default=True)
    parser.add_argument('-e', '--no-extract'  ,               help='Do not run source extractor', action='store_false',default=True)
    #parser.add_argument('-U', '--no-fitsup'  ,                help='Do not run FITS header updated', action='store_false',default=True)
    parser.add_argument('-l','--logfile',      required=True, help='Filename to log messages to')
    parser.add_argument('-v', '--verbose',                    help='Turn on verbose output', 
                        action='store_true',default=False,env_var='VERBOSE')
    parser.add_argument('-E', '--haltfile',    required=True, help='filename to signal HALT completion')
    parser.add_argument('-W', '--warnfile',    required=True, help='filename to signal WARNING completion')
    parser.add_argument('-F', '--filelist',    required=True, help='Filename for raw file list')
    parser.add_argument('-P', '--photometry',  required=True, help='FORTRAN photometry program')
    parser.add_argument('-Q', '--astrometry',  required=True, help='FORTRAN astrometry program')
    parser.add_argument('-B', '--db_occam',    required=True, help='host;username;password;db name')
    parser.add_argument('-A', '--db_info',     required=True, help='host;username;password;db name')
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

def edit_datenite(yyyymmdd):
    # Edit some of the attributes passed
    try:
        datenite_as_date = date(int(yyyymmdd[0:4]),int(yyyymmdd[4:6]),int(yyyymmdd[6:]))
    except Exception:
        logging.exception("Invalid date YYYYMMDD: {}".format(yyyymmdd))
        return None
    return datenite_as_date
        
def get_raw_files(observatory,file_sys,symlink,do_bias,do_flats,do_science):
    """
    Search for all the raw images to be processed for this datenite.
    Put the images in an instance of the FileList oject.
    Pull attributes from the images' FITS header, add them to the file list
    and save that list to a disk file. 
    
    Filter out any images that are bypassed - bias/dark, flats or science
    
    Return the file list object
    """
    # Setup the in memory file list
    fl = FileList(observatory,symlink)
    
    # Search the raw directory for files for this datenite
    fl.select_images(file_sys.obs_rawdir(),file_sys.yyyymmdd)
    
    # For all the files selected, pull attributes from the FITS header
    fl.get_fits_headers()
    
    # Write the list of files and attributes to the processed directory
    # for the datenite (yymmdd)
    if len(fl.imglist) > 0:
        file_sys.make_procdaydirs()   
        fl.save_filelist(file_sys.obs_filelist_path(),do_bias,do_flats,do_science)


    return fl

def cache_best_bias(binning,img_list,cal_info,file_sys):
    """
    Cache the best bias in the processed directory for the datenite
    and return the full path to it.
    
    Return the full file path for this bias master or None if error.
    """
    cnt = Counter()
    
    for img in img_list:
        cnt[img['SET-TEMP']] += 1
    
    temps = list(cnt.keys())
    temp = temps[0]
    if len(temps) > 1:
        for t in temps:
            if cnt[t] > cnt[temp]:
                temp = t
        
    logging.info("Looking for bias bin {} at temp {}".format(binning,temp))
    rel_path,yymmdd,storage_loc = cal_info.get_bias_master(binning,temp,file_sys)
    
    if rel_path is None:
        return None
    
    newname = f"bias_temp{temp}_bin{binning}_{yymmdd}.fits"
    img = {'NEW-FILENAME':newname,'FILEPATH':rel_path,'STORAGE-LOC':storage_loc}
    if not file_sys.copy_images_from_fs_s3([img],'cal',file_sys.obs_procdaydir()):
        logging.error("Cannot cache bias")
        return None
    
    # Bias master is now copied to the processed datenite directory from S3 or file-system
    # if it was not there already
    bias_master = os.path.join(file_sys.obs_procdaydir(),newname)
    return bias_master
                    
                     
def get_best_bias(binning,img_list,cal_info,file_sys):
    """
    Determine the most used set temp for the images in the list.  Get
    the best bias to use at this temp. 
    
    Return the full file path for this bias master.
    """
    cnt = Counter()
    
    for img in img_list:
        cnt[img['SET-TEMP']] += 1
    
    temps = list(cnt.keys())
    temp = temps[0]
    if len(temps) > 1:
        for t in temps:
            if cnt[t] > cnt[temp]:
                temp = t
        
    logging.info("Looking for bias bin {} at temp {}".format(binning,temp))
    rel_path,yymmdd,storage_loc = cal_info.get_bias_master(binning,temp,file_sys)
    if rel_path is not None:
        bias_master = os.path.join(file_sys.procdir,rel_path)
        return bias_master
    else:
        return None

def get_calibration_masters(temp,binning,cal_info,file_list,file_sys):
    """
    Get a list of the best bias, dark and flat master calibration images
    for the given temperature, binning and set of filters on the current
    datenite.
    
    Return a bias image, list of dictionaries representing the darks and flats
    or None if error
    """

    rel_path,yymmdd,storage_loc = cal_info.get_bias_master(binning,temp,file_sys)
    if rel_path is None:
        logging.error("Cannot locate an appropriate bias master")
        return None,None,None
    
    bias_img = dict()
    bias_img['FILEPATH'] = rel_path
    ignore,bias_img['FILENAME'] = os.path.split(rel_path)
    bias_img['IMAGETYP'] = 'bias'
    bias_img['NEW-FILENAME'] = f"bias_temp{temp}_bin{binning}_{yymmdd}.fits"
    bias_img['COMPRESSED'] = ''
    bias_img['STORAGE-LOC'] = storage_loc
    

    darks = cal_info.get_dark_masters(binning,temp,yymmdd,file_sys,file_list)
    if darks is None:
        logging.error("Error getting dark masters")
        return  None,None,None
    for dark in darks:
        dark['NEW-FILENAME'] = f"dark_{int(dark['EXPTIME'])}sec_temp{temp}_bin{binning}_{yymmdd}.fits"

    filters = cal_info.get_flat_filter_list(binning,file_sys)
    if filters is None:
        logging.error("Error getting filter list")
        return  None,None,None
 
    flats = cal_info.get_flat_masters(binning,filters,file_sys,file_list)
    if flats is None:
        logging.error("Error getting flat masters")
        return  None,None,None
    for flat in flats:
        flat['NEW-FILENAME'] = f"flat{flat['FILTER']}_bin{binning}_{flat['DATENITE']}.fits"
   
    return bias_img,darks,flats

def check_darks(masters):
    """
    Check if the dark exposure times are the standard steps:
    1,3,10,30,100,300
    or the alternate used at SRO
    15,30,...
    """
    for master in masters:
        if (master['IMAGETYP'] == 'dark') and (master['EXPTIME'] == 15):
            return True
        
    return False

def get_image_attr(filename):
    """
    Get a list of FITS header attribute values of interest.
    
    Input:
    filename : full path of FITS image file, may be compressed
    
    Returns :
    Dictionary of FITS header values for the attributes given as the input.
    The format is consistent with the image dictionary used in filelist. 
    None if error opening file
    
    """
    img = dict()
    
    try:
        hdul = fits.open(filename)
        hdr = hdul[0].header
    except Exception:
        logging.exception("Cannot open FITS file {}".format(filename))
        return None
        
        
    for col in ('SET-TEMP','FILTER','XBINNING','DATE-OBS','IMAGETYP',
                'EXPTIME','NCOMBINE',):
        try: 
            img[col] = hdr[col]
        except KeyError:
            img[col] = None
                    
    hdul.close()
    
    return img


def get_override_masters(masterdir,procdir):
    """
    Get a list of the bias, dark and flat master calibration images
    from the override master directory masterdir.
    
    Return:
        list of image dictionaries in same format at filelist images
        return None if error
    """
    masters = list()        # image list to return
    

    
    if not os.path.isdir(masterdir):
        logging.error("Override directory is not a directory: {}".format(masterdir))
        return None
    
    for root, dirs, files in os.walk(masterdir,followlinks=True):
        for file in files:
            name,ext = os.path.splitext(file)
            if (ext.lower() in ['.fit','.fits','.fts']):
                filepath = os.path.join(root, file)   
                logging.info("Found FITS file {}".format(filepath))
                # Looks like a FITS file based on name - examine FITS header
                img = get_image_attr(filepath)
                if img is not None:
                    if img['NCOMBINE'] is None:
                        logging.warning("Image is not a master file {}".format(filepath))
                        continue
                    img['FILEPATH'] = filepath[len(procdir)+1:]
                    img['FILENAME'] = file
                    img['SET-TEMP'] = int(float(img['SET-TEMP']))
                    img['COMPRESSED'] = ''
                    typ = img['IMAGETYP']
                    if 'DARK' in typ.upper():
                        typ = 'dark'
                        et = int(img['EXPTIME'])
                        img['NEW-FILENAME'] = f"dark_{et}sec_temp{img['SET-TEMP']}_bin{img['XBINNING']}_override.fits"
                        logging.info("dark {} bin {} temp {} exp {}".format(file,img['XBINNING'],img['SET-TEMP'],img['EXPTIME']))
                    elif 'FLAT' in typ.upper():
                        typ = 'flat'
                        img['NEW-FILENAME'] = f"Flat{img['FILTER']}_bin{img['XBINNING']}_override.fits"
                        logging.info("flat {} bin {} filter {}".format(file,img['XBINNING'],img['FILTER']))
                    elif 'BIAS' in typ.upper():
                        typ = 'bias'
                        img['NEW-FILENAME'] = f"bias_temp{img['SET-TEMP']}_bin{img['XBINNING']}_override.fits"
                        logging.info("bias {} bin {} temp {}".format(file,img['XBINNING'],img['SET-TEMP']))
                    else:
                        logging.warning("Invalid imagetyp for cali master {}".format(typ))
                        continue
                    img['IMAGETYP'] = typ
                    img['STORAGE-LOC'] = 'fs'
                    masters.append(img)
                    
    logging.info("Found {} master calibration override images".format(len(masters)))
    return masters

def subset_override_masters(override,temp,binning,filts):
    """
    Get a list of the bias, dark and flat master calibration images
    from the override master list previously loaded by get_override_masters.
    Select bias/dark for the given temp and binning.
    Select flats for the given binning and filter
    
    Return:
        list of image dictionaries in bias,darks and flats
        return None if error
    """    
    darks = list()
    flats = list()
    
    for img in override:
        if img['IMAGETYP'] =='bias':
            if img['XBINNING'] == binning and img['SET-TEMP'] == temp:
                bias_image = img
                logging.info("Selected {} for bin {}, temp {}".format(img['FILEPATH'],binning,temp))
        elif img['IMAGETYP'] == 'dark':
            if img['XBINNING'] == binning and img['SET-TEMP'] == temp:
                darks.append(img)
                logging.info("Selected {} for bin {}, temp {}".format(img['FILEPATH'],binning,temp))
        elif img['IMAGETYP'] == 'flat':
            if img['XBINNING'] == binning and img['FILTER'] in filts:
                flats.append(img)
                logging.info("Selected {} for bin {}, filters {}".format(img['FILEPATH'],binning,filts))
        else:
            logging.error("Logic error - imagetype {}".format(img['IMAGETYP']))
            return None
    logging.info("Selected  images from override masters for bin {}, temp {}, filters {}".format(binning,temp,filts))
    return bias_image,darks,flats
    
def save_master(file_sys,cal_info,pathname,typ,temp,binning,filt=None,exp=None):
    """
    """
    img = dict()
    img['OBS-PGM'] = file_sys.observatory
    img['SYMLINK'] = file_sys.symlink
    img['IMAGETYP'] = typ
    img['FILEPATH'] = pathname
    img['SET-TEMP'] = temp
    img['FILTER'] = filt
    img['XBINNING'] = binning
    img['EXPTIME'] = exp
    
    try:
        filepath = os.path.join(file_sys.procdir,pathname)
        hdul = fits.open(filepath)
        hdr = hdul[0].header
        if 'DATE-OBS' in hdr:
            img['DATE-OBS'] = hdr['DATE-OBS']
        else:
            img['DATE-OBS'] = None
        hdul.close()     
    except Exception as e:
        logging.exception("Cannot open FITS header for file {}".format(filepath))
        # Testing with imgproc_noop we end up here and need to make
        # a unique date-time
        img['DATE-OBS'] = datetime.today().strftime('%Y-%m-%dT%H:%M:%S.%f') 
                

    
    status,desc = cal_info.insert_master(img,True,file_sys.yymmdd)
    if status:
        logging.info("Inserted master {} into cal_info".format(pathname))
        return True
    else:
        logging.error("Failed to insert master {}, error: {}".format(pathname,desc))
        return False
    
    
def create_bias_master(img_list,temp,binning,file_sys,file_list,image_proc,cal_info):
    """
    Create a bias master from the images in the image list.   Create a ZIP archive
    of the raw images.
    
    Input:
        img_list - list of image dictionaries as defined in class FileList.  These are
                   a subset of all the images being processed for the night.
        temp - set temperature at which the images were taken
        binning - the CCD binning mode
        file_sys - instance of class FileSys for the current run
        file_list - instance of class FileList for the current run
        image_proc - instance of ImageProc
    
    Return the name of the master image created if successful, None otherwise
           the name is the pathname relative to the observatory processed directory
    """
    # Create the raw bias image list
    biaslist = list(img['FILEPATH'] for img in img_list)
    master_name = os.path.join(file_sys.obs_caldaydir(),f"bias_temp{temp}_bin{binning}.fits")
    if not image_proc.create_bias_master(biaslist,master_name,
                          indir=file_sys.obs_rawdir(),
                          overwrite=False,cache_bias=True):
        logging.error(f"Error creating bias {master_name}")
        return None
    
    archive_name = os.path.join(file_sys.obs_caldaydir(),f"raw_bias_temp{temp}_bin{binning}.zip")
    if not file_sys.zip_filelist(file_sys.obs_rawdir(),biaslist,archive_name,compress=False):
        logging.error(f"Error creating bias archive{archive_name}")
        return None

    # Save the master bias to the DB
    master_relpath = master_name[len(file_sys.procdir)+1:]
    if not save_master(file_sys,cal_info,master_relpath,'bias',temp,binning):
        return None
    
    return master_name


def create_dark_master(img_list,bias_master_file,temp,binning,exposure,file_sys,file_list,image_proc,cal_info):
    """
    Create a dark master from the images in the image list.   Archive the raw images to
    a ZIP archive.
    
    Input:
        img_list - list of image dictionaries as defined in class FileList.  These are
                   a subset of all the images being processed for the night.
        bias_master_file - full path of bias master file
        temp - set temperature at which the images were taken
        binning - the CCD binning mode
        exposure - dark exposure time
        file_sys - instance of class FileSys for the current run
        file_list - instance of class FileList for the current run
        image_proc - instance of ImageProc
    
    Return the name of the master image created if successful, None otherwise
           the name is the pathname relative to the observatory processed directory
    """
    # Create the raw bias image list
    darklist = list(img['FILEPATH'] for img in img_list)
    master_name = os.path.join(file_sys.obs_caldaydir(),f"dark_{exposure}sec_temp{temp}_bin{binning}.fits")
    # Here we assume a bias master for the same temp and binning was just created and cached
    # so we do not pass it in.
    if not image_proc.create_dark_master(darklist,master_name,
                          indir=file_sys.obs_rawdir(),
                          overwrite=False):
        logging.error(f"Error creating dark {master_name}")
        return None
    
    archive_name = os.path.join(file_sys.obs_caldaydir(),f"raw_dark_{exposure}sec_temp{temp}_bin{binning}.zip")
    if not file_sys.zip_filelist(file_sys.obs_rawdir(),darklist,archive_name,compress=False):
        logging.error(f"Error creating dark archive{archive_name}")
        return None
   
    # Save the master dark to the DB
    master_relpath = master_name[len(file_sys.procdir)+1:]
    if not save_master(file_sys,cal_info,master_relpath,'dark',temp,binning,exp=exposure):
        return None
    
    return master_name

def create_flat_master(img_list,binning,filt,file_sys,file_list,image_proc,cal_info):
    """
    Create a flat master from the images in the image list.   Archive raw images from
    this and any adjacent nights to a ZIP file.
    
    Input:
        img_list - list of image dictionaries as defined in class FileList.  These are
                   a subset of all the images being processed for the night.
        binning - the CCD binning mode
        filt - flat filter 
        file_sys - instance of class FileSys for the current run
        file_list - instance of class FileList for the current run
        image_proc - instance of ImageProc
    
    Return the name of the master image created if successful
           The name is the pathname relative to the observatory processed directory
           None if not enough raw images to make a master
           False if error
    """
    # Copy files from raw to calibration yymmdd.  
    # We may use them on another night's processing.
    status = file_sys.copy_images(img_list,file_sys.obs_caldaydir())
    if not status:
        return False

    # Insert raw flats into cal_info table
    rel_dir = file_sys.obs_caldaydir()
    rel_dir = rel_dir[len(file_sys.procdir)+1:]      # strip off absolute protion
    status,desc = cal_info.insert_raw_list(img_list,False,file_sys.yymmdd,rel_dir)
    if status:
        logging.info("{} raw {} flats inserted into cal_info".format(len(img_list),filt))
    else:
        logging.error("Error {} inserting raw {} flats into cal_info".format(desc,filt))
        return False
    

    # Get a list of raw flats including adjacent days that will be combined in the master
    status,adj_list = cal_info.get_image_list(file_sys,file_list,False,'flat',binning,_FLAT_DAY_RANGE,filt=filt)
    if status:
        logging.info("{} raw {} flats found including adjacent days".format(filt,len(adj_list)))
    else:
        logging.error("Error {} searching for {} flats from adjacent days".format(filt,desc))
        return False

    if (len(adj_list) < _FLAT_MIN_RAW_PER_MASTER):
        return None
        
    # Here we get a bias master to use in creating the flat master.  If the
    # set of raw flats have multiple set temps, pick the often used temp for
    # bias selection
    

    bias_master_file = cache_best_bias(binning,img_list,cal_info,file_sys)
    if bias_master_file is None:
        return False
    
    
    # Check to see if any of the adjacent flats are in S3 and copy them back to their
    # original location if so
    s3_list = [f for f in adj_list if f['STORAGE-LOC'] == 's3']
    for f in s3_list:
        f['NEW-FILENAME'] = f['FILEPATH']
    if file_sys.copy_images_from_fs_s3(s3_list,'cal',file_sys.procdir):
        logging.info("Restored {} raw flats from S3".format(len(s3_list)))
    else:
        logging.error("Cannot restore raw flats from S3")
        return False
    
    # Create the flat master
    master_name = os.path.join(file_sys.obs_caldaydir(),f"flat{filt}_bin{binning}.fits")
    flatlist = list(img['FILEPATH'] for img in adj_list)
    if not image_proc.create_flat_master(flatlist,master_name,indir=file_sys.procdir,biasmast=bias_master_file):
        logging.error(f"Error creating flat {master_name}")
        return False

    # Insert the flat master into the database
    master_relpath = master_name[len(file_sys.procdir)+1:]    
    if not save_master(file_sys,cal_info,master_relpath,'flat',None,binning,filt=filt):
        return False
    
    # Create a ZIP archive containing the raw flats used in the master
    archive_name = os.path.join(file_sys.obs_caldaydir(),f"raw_flat{filt}_bin{binning}.zip")
    if not file_sys.zip_filelist(file_sys.procdir,flatlist,archive_name,compress=False):
        logging.error(f"Error creating flat archive{archive_name}")
        return None

    # Remove the raw flats we restored from S3
    if (len(s3_list) + 0):
        n = 0
        for f in s3_list:
            os.remove(os.path.join(file_sys.procdir,f['NEW-FILENAME']))
            n += 1
        logging.info("Removed {} raw flats restored from S3".format(n))
    
    return master_name

def create_testflat_master(img_list,binning,filt,file_sys,file_list,image_proc,cal_info):
    """
    Create a test flat master from the images in the image list.  This master
    consists of only the flats taken this night and not adjacent nights.  It
    is used to check the quality of the flats for the night.  Since this 
    master is just for testing, it is not inserted into the DB.   
    
    Input:
        img_list - list of image dictionaries as defined in class FileList.  These are
                   a subset of all the images being processed for the night.
        binning - the CCD binning mode
        filt - flat filter 
        file_sys - instance of class FileSys for the current run
        file_list - instance of class FileList for the current run
        image_proc - instance of ImageProc
    
    Return the name of the master image created if successful
           The name is the pathname relative to the observatory processed directory
           None if not enough raw images to make a master
           False if error

    """

    # NOTE: the testflat is created no matter how many individual flats are available

    # Here we get a bias master to use in creating the flat master.  If the
    # set of raw flats have multiple set temps, pick the often used temp for
    # bias selection
    bias_master_file = cache_best_bias(binning,img_list,cal_info,file_sys)
    if bias_master_file is None:
        return False
    
    # Create the flat master
    flatlist = list(img['FILEPATH'] for img in img_list)    
    master_name = os.path.join(file_sys.obs_caldaydir(),f"testflat{filt}_bin{binning}.fits")
    if not image_proc.create_flat_master(flatlist,master_name,indir=file_sys.obs_rawdir(),biasmast=bias_master_file):
        logging.error(f"Error creating flat {master_name}")
        return False
    
    # Copy test flat master to datenite subdir
    # for use in thumbnails and trouble shooting
    try:
        shutil.copy(master_name,file_sys.obs_procdaydir())
    except Exception:
        logging.exception("Failed to copy {} to {}".format(master_name,file_sys.obs_procdaydir()))
        return False
            
    return master_name

def create_calibration_masters(file_sys,file_list,image_proc,cal_info,do_bias,do_flats):
    """
    Create master bias, dark and flat masters.  Create separate masters for each
    binning mode in the input calibration images.  For bias and dark create separate masters
    for each set temperature.   
    
    Return True for success, False for failure
    """
    # Setup the in memory file list
    temps = file_list.get_temps_list()
    bins = file_list.get_bins_list()
    ffilts = file_list.get_flat_filters_list()
    dexps = file_list.get_dark_exposures_list()
    
    # Loop through all the calibration images
    got_darks = False           # so we can decide whether to archive
    for t in temps:
        for b in bins:
            bias_master_file = None
            # Biases
            images = file_list.get_images(t,b,'bias')
            if (len(images) > 0) and do_bias:
                got_darks = True
                bias_master_file = create_bias_master(images,t,b,file_sys,file_list,image_proc,cal_info)
                if bias_master_file is None:
                    logging.error("Error while creating bias master - terminating")
                    return False
                else:
                    logging.info(f"Bias master {bias_master_file} from {len(images)} images at temp {t} bin {b}")
            else:
                if len(images) == 0:
                    logging.info(f"No bias images this night for temp {t} bin {b}")
                else:
                    logging.warning("Bias images not processed per program args")            # Darks
            for e in dexps:
                images = file_list.get_images(t,b,'dark',exp=e)
                if (len(images) > 0) and do_bias:
                    got_darks = True
                    dark_master_file = create_dark_master(images,bias_master_file,t,b,e,file_sys,file_list,image_proc,cal_info)
                    if dark_master_file is None:
                        logging.error("Error while creating dark master - terminating")
                        return False
                    else:
                        logging.info(f"Dark master {dark_master_file} from {len(images)} images at temp {t} bin {b} exposure {e}")
                else:
                    if len(images) == 0:
                        logging.info(f"No dark images this night for temp {t} bin {b} exposure {e}")
                    else:
                        logging.warning("Dark images not processed per program args")            # Darks
            
    # Flats
    for b in bins:
        for f in ffilts:
            images = file_list.get_images(None,b,'flat',filt=f)
            if (len(images) > 0) and do_flats:
                logging.info(f"Creating flat master from {len(images)} images bin {b} filter {f}")
                # First create master to calibrate science images that includes flats from
                # this and adjacent nights.
                flat_master_file = create_flat_master(images,b,f,file_sys,file_list,image_proc,cal_info)
                if flat_master_file is False:
                    logging.critical("Error while creating flat master - terminating")
                    return False
                elif flat_master_file is not None:
                    logging.info(f"Flat master {flat_master_file} from {len(images)} images bin {b} filter {f}")
                # Note: None is returned if  not flat was created due to too few raw images
                   
                # Next create master to test this night's images 
                testflat_master_file = create_testflat_master(images,b,f,file_sys,file_list,image_proc,cal_info)
                if testflat_master_file is None:
                    logging.critical("Error while creating test flat master - terminating")
                    return False
                else:
                    logging.info(f"Test flat master {testflat_master_file} from {len(images)} images bin {b} filter {f}")
            else:
                if len(images) == 0:
                    logging.info(f"No flat images for bin {b} filter {f}")
                else:
                    logging.warning("Flat images not processed per program args")            # Darks
                        
    return True

def calibrate_images(file_sys,file_list,image_proc,cal_info,override=None):
    """
    Calibrate the science images.   
    
    Return True for success, False for failure
    """
    # Setup the in memory file list
    temps = file_list.get_temps_list()
    bins = file_list.get_bins_list()
    filts = file_list.get_light_filters_list()
    
    # Loop through all the science images by bin and temp
    got_images = False
    for t in temps:
        for b in bins:
            # Get science images
            images = file_list.get_images(t,b,'light')
            if len(images) == 0:
                logging.info("No science images this night for temp {} bin {}".format(t,b))
            else:
                got_images = True
                logging.info("Calibrating {} science images for temp {} bin {}".format(len(images),t,b))
                               
                # Locate master calibration files for these science images
                if override is not None:
                    bias_image,darks,flats = subset_override_masters(override,t,b,filts)
                else:
                    bias_image,darks,flats = get_calibration_masters(t,b,cal_info,file_list,file_sys)
                if bias_image is None or len(darks) == 0 or len(flats) == 0:
                    logging.error("Error finding master(s) for science images")
                    return False

                masters = list()
                masters.append(bias_image)
                masters.extend(darks)
                masters.extend(flats)
                if file_sys.copy_images_from_fs_s3(masters,'cal',file_sys.obs_procdaydir()):
                    logging.info("copied {} master images to datenite dir".format(len(masters)))
                else:
                    logging.error("Error uncompress/copying master images")
                    return False
                
                imgfiles = list([img['FILEPATH'],img['NEW-FILENAME']] for img in images)
                if image_proc.calibrate_images(imgfiles,
                                               file_sys.obs_procdaydir(),
                                               bias_image['NEW-FILENAME'],
                                               list(img['NEW-FILENAME'] for img in darks),
                                               list(img['NEW-FILENAME'] for img in flats),
                                               masters_dir=file_sys.obs_procdaydir(),
                                               indir=file_sys.obs_rawdir()):
                    logging.info("Science images calibrated")
                else:
                    logging.error("Error calibrating science images")
                    return False
                
                
    if got_images:            
        # Create headlog.txt and improc_parm.txt files
        file_list.save_headlog(os.path.join(file_sys.obs_procdaydir(),"headlog.txt"))
        file_list.save_improc_parm(os.path.join(file_sys.obs_procdaydir(),"improc_parm.txt"),file_sys.yymmdd)
        logging.info("Wrote photometry parameter file")
        if file_sys.copy_photometry_init():
            logging.info("Copied photometry initialization file")
        else:
            return False
             
    return True

def launch_pgm(cmd,args,homedir,inp=None,timeout=60000):
    """
    Launch the script and wait for the response.  Capture and print the output.
    Return the process exit code.
    """
    if not os.path.exists(cmd):
        logging.error("*** launch_pgm {} does not exist".format(cmd))
        return -99

    try:
        cmdline = [cmd,]
        cmdline.extend(args)
        logging.info("launching subprocess {}, timeout {}".format(cmdline,timeout))
        completion = subprocess.run(cmdline,
                                    input=inp,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True,
                                    timeout=timeout,
                                    cwd=homedir)

    except subprocess.SubprocessError as e:
        logging.exception("Subprocess exeption {}".format(e))
        return -99
    
    logging.info("Subprocess returned {}".format(completion.returncode))
    logging.info (completion.stdout)
    return completion.returncode

def launch_shell(cmdline,homedir,inp=None,timeout=36000):
    """
    Launch the command line in a shell and wait for the response.  Capture and print the output.
    Return the process exit code.
    """
    try:

        logging.info("launching subprocess shell {}, timeout {}".format(cmdline,timeout))
        completion = subprocess.run(cmdline,
                                    shell=True,
                                    input=inp,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True,
                                    timeout=timeout,
                                    cwd=homedir)

    except subprocess.SubprocessError as e:
        logging.exception("Subprocess exeption {}".format(e))
        return -99
    
    logging.info("Subprocess returned {}".format(completion.returncode))
    logging.info (completion.stdout)
    return completion.returncode


def update_wcs(images,imgdir):
    """
    Run the Astrometry.net plate solve on all the science images
    passed in the list images.
    Return the number of images with a solution. 

    """
    logging.info("Begin plate solving")
    
    # Get a list of all science images
    nbrSolved = 0
    endtime = datetime.now()
    
    for img in images:
        fitspath = os.path.join(imgdir,img['NEW-FILENAME'])
        begtime = endtime
        status = add_astrometry(fitspath,overwrite=True
                           ,replace_wcs=True
                           ,custom_sextractor=True)
        endtime = datetime.now()
        elapsed = (endtime - begtime).total_seconds()
        
        if status:
            nbrSolved += 1
            logging.info("WCS replaced/added for {} in {} sec".format(fitspath,elapsed))
            img['PLTSOLVD']='True'
        else:
            logging.warning("WCS solution failed for {} in {} sec".format(fitspath,elapsed))
            img['PLTSOLVD']='False'
       
    return nbrSolved

def sextractor(args,images,imgdir):
    """
    Run the source extractor on all the science images
    passed in the list of images.
    
    Update the image dictionary in the list to include
    the metrics from source extractor.
    
    Return the number of images with successful extraction.
    
    """
    logging.info("Begin source extractor processing")
    
    # Get a list of all science images
    nbrSolved = 0
    endtime = datetime.now()
    
    for img in images:
        fitspath = os.path.join(imgdir,img['NEW-FILENAME'])
        begtime = endtime
        srcs_se,srcs_fort,xmatched_se = extract_processing(fitspath,args.thresh_sextractor,args.area_sextractor)
        endtime = datetime.now()
        elapsed = (endtime - begtime).total_seconds()
        
        if srcs_se is not None:
            nbrSolved += 1
            logging.info("Sources extracted for {} in {} sec".format(fitspath,elapsed))
            img['SRCS_SE'] = srcs_se
            img['SRCS_FORT'] = srcs_fort
            img['XMATCHED_SE'] = xmatched_se
        else:
            logging.warning("Source extraction failed for {} in {} sec".format(fitspath,elapsed))
       
    return nbrSolved

def photom_astrom(photpgm,astrpgm,imgdir,yymmdd,hphot_input):
    """
    Run the FORTRAN photometry and astrometry programs in the processed datenite directory.
    Compress the images with bzip2.
    Return True if all programs return code is zero.  False otherwise.
    """
    logging.info("Begin photometry")
    rtncode = launch_pgm(photpgm,[],imgdir,inp=hphot_input)
    if rtncode != 0:
        logging.error("Photometry program abnormal return code {}".format(rtncode))
        return False
    else:
        logging.info("Photometry program normal return")

    logging.info("Begin astrometry")
    rtncode = launch_pgm(astrpgm,[],imgdir)
    if rtncode != 0:
        logging.error("Astrometry program abnormal return code {}".format(rtncode))
        return False
    else:
        logging.info("Astrometry program normal return")
        
    logging.info("Begin image compression")
    rtncode = launch_shell('/bin/bzip2 {}.[0-9][0-9][0-9][0-9]'.format(yymmdd),imgdir)
    if rtncode != 0:
        logging.error("Compression abnormal return code {}".format(rtncode))
        return False
    else:
        logging.info("Compression program normal return")
        
    return True
        
"""
Mainline processing begins
"""
if __name__ == "__main__":
    # Get the program parameters from config file and command line
    # Configure the logging subsystem
    args = get_pgm_parms()

    # Print some msgs as well as log show they show up in cron job output
    logging.info("BEGIN *** pypline calibration processing begins")
    logging.info(_svn_id_)
    print("BEGIN *** pypline calibration processing begins")
    
    args = get_pgm_parms()
    # Due to difficulties with argparse, names there are inverted
    # Create meaningful names
    do_flats =  args.no_flats
    do_bias =  args.no_bias
    do_science =  args.no_science
    do_thumb = args.no_thumb
    do_photom = args.no_photom
    do_dbinfo = args.no_dbinfo
    do_wcs = args.no_wcs
    do_sextractor = args.no_extract
    #do_fitsup = args.no_fitsup
    do_fitsup = False
    
    # Setup the file system singleton object
    try:
        fs = FileSys(args.rawdir,args.procdir,args.homedir,args.thumbdir,args.filelist,
                     args.haltfile,args.warnfile,args.observatory,args.yyyymmdd)
    except Exception as e:
        logging.exception("Exiting due to fatal error in configuration: FileSys")
        raise e      
        # program will exit and unhandled exception will be noticed in shell
      
    # Here we allow the overriding of selection of master calibration
    # files.  If masterdir is defined, we search the specified directory
    # for master bias, dark and flat images and create a list of them.
    # During an override, processing of raw calibration images for the
    # datenite is turned off.
    if args.masterdir is not None:
        logging.info("Overriding calibration masters from dir {}".format(args.masterdir))
        do_flats = False
        do_bias = False
        cal_override = True
        override_masters = get_override_masters(args.masterdir,args.procdir)
        l = len(args.procdir)
        if args.masterdir[:l] != args.procdir:
            haltmsg = "HALT: Override masters must be subdir under {}".format(args.procdir)
            fs.write_haltfile(haltmsg)
            logging.critical(haltmsg)
            sys.exit(99)
        if override_masters is None:
            haltmsg = "HALT: Cannot locate override masters"
            fs.write_haltfile(haltmsg)
            logging.critical(haltmsg)
            sys.exit(99)
    else:
        cal_override = False
       
    logging.info("will process - bias/dark: {}, flats: {}, science: {}".format(do_bias,do_flats,do_science))

    # Setup the image processing singleton object
    ip = ImgProc()
    
    # Connect to the database
    try:
        cal_info = CalInfo(args.db_occam)
    except ValueError:
        haltmsg = "HALT: Cannot connect to the cal_info database"
        fs.write_haltfile(haltmsg)
        logging.critical(haltmsg)
        sys.exit(99)
        
    # Check datenite
    datenite_as_date = edit_datenite(args.yyyymmdd)
    if (datenite_as_date is None):
        haltmsg = "HALT: Cannot process with invalid datenite"
        fs.write_haltfile(haltmsg)
        logging.critical(haltmsg)
        sys.exit(99)
        
    # Setup singleton observatory object and get the telescope information for
    # the images being processed
    obs = Observatories(args.observatory)
    telescope = obs.get_telescope_datenite(args.yyyymmdd)
    if (telescope is None):
        haltmsg = "HALT: telescopes web service down or invalid datenite {}".format(args.yyyymmdd)
        fs.write_haltfile(haltmsg)
        logging.critical(haltmsg)
        sys.exit(99)
        
    logging.info("Processing with observatory {}, telescope {}".format(args.observatory,telescope))
    
    result = fs.switch2telescop(telescope['symlink'],False)
    if (result):
        logging.info("switch successful for observatory {}, symlink {}".format(args.observatory,telescope['symlink']))
    else:
        haltmsg = "HALT: switch failed for observatory {}, symlink {}".format(args.observatory,telescope['symlink'])
        logging.critical(haltmsg)
        fs.write_haltfile(haltmsg)
        sys.exit(99)     
    
    # If this is a rerun, we need to clear out old contents of directories
    # and the database in addition to making the directories in the case of
    # a first time run
    cal_info.delete_datenite(args.observatory,telescope['symlink'],fs.yymmdd)
    fl = get_raw_files(args.observatory,fs,telescope['symlink'],do_bias,do_flats,do_science)
    if len(fl.imglist) == 0:
        logging.info("No images for {}.  End run.".format(args.yyyymmdd))
        sys.exit(2)

    
    # Halt if images are not all binned the same
    # CLKotnik 2020-10-17 remove this check
    """
    if (len(fl.binning_cnts) > 1):
        # Writing the msg to haltfile tells caller we had a fatal error
        fs.write_haltfile('HALT: Images are not all binned the same way')
        sys.exit(99)
    """

    # Halt if images have invalid filters    
    if (not fl.filtersValid):
        # Writing the msg to haltfile tells caller we had a fatal error
        fs.write_haltfile('HALTING: Image(s) have invalid filter')
        sys.exit(99)      
        
        
    # Halt if images have no set-temp   
    if ('' in fl.temp_cnts):
        # Writing the msg to haltfile tells caller we had a fatal error
        fs.write_haltfile('HALTING: Image(s) must have SET_TEMP: Was cooler on?')
        sys.exit(99)        
    
    # Calibration Masters
    if ('bias' in fl.type_cnts) or ('dark' in fl.type_cnts) or ('flat' in fl.type_cnts):
        if do_flats or do_bias:
            fs.make_caldaydirs()
            if create_calibration_masters(fs,fl,ip,cal_info,do_bias,do_flats):
                logging.info("Calibration master creation complete")
                print("Calibration master creation complete")
            else:
                fs.write_haltfile('HALT: Calibration master creation failed')
                sys.exit(99)
    else:
        logging.info("No calibration files this run")
        print("No calibration files this run")

    # So post calibration processing updates the values in the rawimages.txt file
    # so it would need to be rewritten.  Assume no rewrite.
    rewrite_rawimages = False

        
    # If we will have science images, create the thumbnail directories
    if do_science and ('light' in fl.type_cnts):
        fs.make_thumbdaydirs() 
        
    # Calibrate science images
    if do_science:
        if cal_override:
            status = calibrate_images(fs,fl,ip,cal_info,override=override_masters)
        else:
            status = calibrate_images(fs,fl,ip,cal_info)
        if status:
            logging.info("Calibration of science images complete")
            print("Calibration of science images complete")
        else:
            fs.write_haltfile('HALT: Calibration of science images failed')
            sys.exit(99)
    else:
        logging.warning("Science images not processed per program args")
        print("Science images not processed per program args")
        sys.exit(99)
        
    
    if do_thumb:
        make_all_tn(fs.thumbdir,
                                      fs.procdir,
                                      args.observatory,
                                      telescope['symlink'],
                                      args.yyyymmdd[2:],
                                      None,
                                      cal_info,
                                      filelist=fl)
        logging.info("Created thumbnail images")
        print("Created thumbnail images")
    else:
        logging.warning("Thumbnails disabled in config")
        print("Thumbnails disabled in config")
        
    # Done with this table/database
    cal_info.disconnect()
    
    # The rest of the processing is only necessary if we have science
    # images
    got_images = os.path.exists(os.path.join(fs.obs_procdaydir(),"headlog.txt"))
    if (not do_science) or (not got_images):
        logging.info("END *** pypline calibration processing ends - no science images")
        print("END *** pypline calibration processing ends - no science images")    
        sys.exit(99)
        
    if do_wcs:
        images = fl.get_images(None,None,'light')
        n = update_wcs(images,fs.obs_procdaydir())
        logging.info("Found WCS for {} out of {} science images".format(n,len(images)))
        print("Found WCS for {} out of {} science images".format(n,len(images)))
        rewrite_rawimages = True
        
    if do_fitsup:
        # Update FITS header for target off center.  In this case, ACP was told "object" was
        # where the image was to be centered.  ACP therefore put these coordinates in the FITS
        # header RA and DEC.
        #  The OBJECT FITS header is the celestial object
        # of interest.  The FileList object has found the coordinates for this in VSX and we
        # now update the RA/DEC we now have that object update the images with these values.
        fl.update_fits_headers(fs.obs_procdaydir())
        logging.info("FITS headers updated for off-center objects")        
        
        
    if do_photom:
        hphot_input = fl.get_new_improc_parm(os.path.join(fs.obs_procdaydir(),"improc_parm.txt"))
        result = photom_astrom(args.photometry,args.astrometry,fs.obs_procdaydir(),fs.yymmdd,hphot_input)
        if not result:
            fs.write_warnfile('WARN: Photometry/astrometry/compression of science images failed')
            #sys.exit(99)
        logging.info("Performed photometry, astrometry and compression")
        print("Performed photometry, astrometry and compression")
            
    else:
        logging.warning("Photometry and astrometry disabled in config")
        print("Photometry and astrometry disabled in config")
        
    if do_sextractor:
        images = fl.get_images(None,None,'light')
        n = sextractor(args,images,fs.obs_procdaydir())
        logging.info("Extracted sources for {} out of {} science images".format(n,len(images)))
        print("Extracted sources for {} out of {} science images".format(n,len(images)))
        rewrite_rawimages = True
    else:
         logging.info("sextractor disabled by configuration parameter")
         print("sextractor disabled by configuration parameter")
        
    if do_dbinfo:
        # Connect to the database
        try:
            img_info = ImageInfo(args.db_info)
        except ValueError:
            haltmsg = "HALT: Cannot connect to the image_info database"
            fs.write_haltfile(haltmsg)
            logging.critical(haltmsg)
            sys.exit(99)  
                  
        # If this is a rerun, we need to clear out the database
        nbrdel = img_info.delete_datenite(args.observatory,fs.yymmdd)
        if nbrdel is None:
            haltmsg = "HALT: Error deleting image_info"
            fs.write_haltfile(haltmsg)
            logging.critical(haltmsg)
            sys.exit(99)        
                
            
        imgs = fl.get_images(None,None,'light')
        status,desc = img_info.insert_processed_list(imgs,fs.yymmdd)
        if not status:
            haltmsg = "HALT: Error inserting into image_info - {}".format(desc)
            fs.write_haltfile(haltmsg)
            logging.critical(haltmsg)
            sys.exit(99)        
        logging.info("Added images to image_info table")
        print("Added images to image_info table")
    else:
        logging.warning("Update of image_info disabled in config")
        print("Update of image_info disabled in config")
    

    if rewrite_rawimages:
        # Since we updated the science images with astrometry.new and/orsextractor results
        # rewrite the raw images text file
        fl.save_filelist(fs.obs_filelist_path(),do_bias,do_flats,do_science)
        logging.info("rawimages.txt rewritten")

        
    logging.info("END *** pypline calibration processing ends")
    print("END *** pypline calibration processing ends")

    sys.exit(99)