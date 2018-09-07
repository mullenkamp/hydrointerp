# -*- coding: utf-8 -*-

import os, glob, datetime, subprocess
import rasterio
import numpy as np
from netCDF4 import Dataset

'''
Convert TRMM netcdf to GTiff for specified period of time and project and resample to finer resolution if required
'''

#-Authorship information-###################################################################
__author__ = 'Wilco Terink'
__copyright__ = 'Wilco Terink'
__version__ = '1.0'
__email__ = 'wilco.terink@ecan.govt.nz'
__date__ ='May 2018'
############################################################################################

#-Paths
ncDir = r'C:\Active\Projects\Rakaia\Data\Atmospheric\TRMM'
tifDir = r'C:\Active\Projects\Rakaia\Data\Atmospheric\TRMM\tif'

#-Period
sYear = 1998
sMonth = 1
sDay = 1

eYear = 2017
eMonth = 12
eDay = 31

#-Bounding box (clipping) of area of interest
xmin = 1401683.2619799999520183
ymin = 5118270.5192900002002716
xmax = 1560183.2619799999520183
ymax = 5264870.5192900002002716

#-output crs (EPSG number)
CRS = 2193

#-resampling
r = 'bilinear'
res = 100 #-m


#################################################################################################

#-Dumbs information about the netcdf file
def dump_nc_info(file):
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key
    
        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print "\t\ttype:", repr(nc.variables[key].dtype)
            for ncattr in nc.variables[key].ncattrs():
                print '\t\t%s:' % ncattr,\
                      repr(nc.variables[key].getncattr(ncattr))
        except KeyError:
            print "\t\tWARNING: %s does not contain variable attributes" % key
    
    #-Create dataset instance of netcdf file
    nc = Dataset(file, 'r')
    
    #-Global attributes of NetCDF file
    nc_attrs = nc.ncattrs()
    for nc_attr in nc_attrs:
        print '\t%s:' % nc_attr, repr(nc.getncattr(nc_attr))
     
    #-Dimension information
    nc_dims = [dim for dim in nc.dimensions]  # list of nc dimensions
    # Dimension shape information.
    print "NetCDF dimension information:"
    for dim in nc_dims:
        print "\tName:", dim 
        print "\t\tsize:", len(nc.dimensions[dim])
        print_ncattr(dim)
             
     #-Variable information.
    nc_vars = [var for var in nc.variables]  # list of nc variables
    print "NetCDF variable information:"
    for var in nc_vars:
        if var not in nc_dims:
            print '\tName:', var
            print "\t\tdimensions:", nc.variables[var].dimensions
            print "\t\tsize:", nc.variables[var].size
            print_ncattr(var)
            
#-Write Numpy Array to GeoTiff file
def writeGTiff(array, lats, lons, tifDir, outFile, bbox, CRS, resamp=False):
    source_crs = rasterio.crs.CRS.from_epsg(4326)
    tempFile = os.path.join(tifDir, 'temp.tif')
    #-remove tempFile if already exist on disk    
    if os.path.isfile(tempFile):
        os.remove(tempFile)
    #-Raster Geo properties
    cols = array.shape[1]
    rows = array.shape[0]
    xmin, ymin, xmax, ymax = [lons.min(), lats.min(), lons.max(), lats.max()]
    xres = (xmax - xmin) / float(cols)
    yres = (ymax - ymin) / float(rows)
    geotransform = [xmin, xres, 0.0, ymax, 0.0, -yres]
    #-write the raster to a temporary GTIff file
    with rasterio.open(tempFile, 'w',
                       driver='GTIff',
                       height=rows,
                       width=cols,
                       count=1,
                       dtype=rasterio.dtypes.float32,
                       crs = source_crs,
                       transform = geotransform
                       ) as dataset:
        dataset.write(array, 1)
        dataset.nodata = -9999

    #-set outFile
    outFile = os.path.join(tifDir, outFile)
    #-remove outFile if already exist on disk    
    if os.path.isfile(outFile):
        os.remove(outFile)
    inFile = tempFile
    #-check if resampling to finer resolution is needed
    if resamp:
        r = resamp[0]
        res = resamp[1]
        #-Project to desired CRS, clip from bounding box, and resample to resolution (res) using resample method (r)
        subprocess.Popen('gdalwarp -r ' + r + ' -tr ' + str(res) + ' ' + str(res) + ' -s_srs EPSG:4326 -t_srs EPSG:' + str(CRS) + 
                         ' -te ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' +str(bbox[2]) + ' ' +str(bbox[3]) + ' ' + 
                         inFile + ' ' + outFile, shell=True).wait()
    else:
        #-Project to desired CRS and clip from bounding box
        subprocess.Popen('gdalwarp -s_srs EPSG:4326 -t_srs EPSG:' + str(CRS) + ' -te ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' +str(bbox[2]) + \
                         ' ' +str(bbox[3]) + ' ' + inFile + ' ' + outFile, shell=True).wait()

#-Get the arrays from the netCDF file
def getNetCDFArrays(f, v):
    ncDataSet = Dataset(f,'r')
    data = ncDataSet[v][:]
    data = np.rot90(data)
    #data = np.flipud(data)
    #print data[:5,:5]
    #-replace fillValue with NaN
    if v == 'precipitation':
        data[data==-9999.9004] = np.nan
    #-dimensions
    lats = ncDataSet['lat'][:]
    lons = ncDataSet['lon'][:]

    #print v + ' has:\n\t%d rows\n\t%d cols' %(data.shape[0],data.shape[1])
    
    return data, lons, lats

#-Convert netCDFs to GTiffs for a specified period and resample if needed
def convert(ncDir, tifDir, sYear, sMonth, sDay, eYear, eMonth, eDay, bbox, CRS, resamp=False):
    sDate = datetime.date(sYear, sMonth, sDay)
    eDate = datetime.date(eYear, eMonth, eDay)
    curDate = sDate
    while curDate<=eDate:
        print curDate
        strDate = curDate.strftime('%Y%m%d')
        inf = glob.glob(os.path.join(ncDir, '*%s*' %strDate))[0]
        outf = strDate + '_3B42_prec.tif'
        
        [prec, lons, lats] = getNetCDFArrays(inf, 'precipitation')
        writeGTiff(prec, lats, lons, tifDir, outf, bbox, CRS, resamp)
        
        curDate += datetime.timedelta(days=1)


bbox = [xmin, ymin, xmax, ymax]
convert(ncDir, tifDir, sYear, sMonth, sDay, eYear, eMonth, eDay, bbox, CRS, resamp = [r, res])
