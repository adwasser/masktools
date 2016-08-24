'''
Module for dealing with multi-object spectroscopy slitmasks
'''
from __future__ import (print_function, absolute_import,
                        division, unicode_literals)

import numpy as np
import pandas as pd
from scipy import signal
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord

def offset(measured_edges, predicted_edges, size=2048):
    '''
    Calculates the constant offset between two lists of edges.

    Parameters
    ----------
    measured_edges : array of pixel values
    predicted_edges : array of pixel values
    size : number of pixels to use

    Returns
    -------
    lag : float, pixel offset between the two inputs, predicted + lag = measured
    '''
    n = size
    gauss = signal.gaussian(51, std=5)
    # pixmax = max(np.amax(measured_edges), np.amax(predicted_edges))
    pixels = np.arange(n)
    # n = pixels.size
    m_pix = np.zeros(n)
    m_pix[map(int, measured_edges)] = np.ones(len(measured_edges))
    m_pix_smooth = signal.convolve(m_pix, gauss, mode='same')
    p_pix = np.zeros(n)
    p_pix[map(int, predicted_edges)] = np.ones(len(predicted_edges))
    p_pix_smooth = signal.convolve(p_pix, gauss, mode='same')
    corr = signal.correlate(m_pix_smooth, p_pix_smooth, mode='full')
    lags = np.arange(corr.size) - m_pix.size + 1
    lag = lags[np.argmax(corr)]
    return lag


class Slitmask(object):
    '''
    Generic slitmask class, should be sub-classed for specific instruments.
    '''

    def __init__(self, mask_name, mask_ra, mask_dec, mask_pa, slits):
        '''
        Parameters
        ----------
        mask_name : str, name of mask
        mask_ra : float, right ascension of mask origin, in degrees
        mask_dec : float, declination of mask origin, in degrees
        mask_pa : position angle of mask, in sky coordinates (degrees ccw from North)
        slits : record array (or pandas.DataFrame) of slits with attributes (fields/columns) of
                ['name', 'x', 'y', 'left_edge', 'right_edge', 'ra', 'dec', 'pa', 'width', 'length', 'isalign']
                name : str, unique identifier
                x : float, arcsec, mask coordinate
                y : float, arcsec, mask coordinate
                left_edge : float, arcsec, left edge x-coordinate 
                right_edge : float, arcsec, right edge x-coordinate
                ra : float, degree of sky coordinate
                dec : float, degree of sky coordinate
                pa : float, degree, ccw from +y (i.e., East of North)
                width : float, arcsec, slit width along the dispersion direction (for pa = mask_pa)
                length : float, arcsec, slit length along spatial direction (for pa = mask_pa)
                isalign : bool, true if slit is an alignment box
        '''
        self._mask_name = mask_name
        try:
            self._mask_coord = SkyCoord(mask_ra, mask_dec, unit=('deg', 'deg'))
            # self._slit_coords = ...
        except TypeError:
            self._mask_coord = None
        self._mask_pa = mask_pa
        columns = ['name', 'x', 'y', 'left_edge', 'right_edge', 'pa', 'width', 'length', 'isalign']
        if slits is None:
            self.slits = pd.DataFrame(columns=columns)
        else:
            self.slits = pd.DataFrame(slits)


    def __len__(self):
        return len(self.slits)
    
    def __repr__(self):
        s = ('<' + type(self).__name__ + ': ' + self.mask_name +  ' (' +
             str(len(self.slits)) + ' slits)>')
        return s
    
    @property
    def mask_name(self):
        return self._mask_name

    @property
    def mask_coord(self):
        return self._mask_coord

    @property
    def mask_pa(self):
        return self._mask_pa

    def slit_edges(self, idx=slice(None), coord_transform=None):
        '''
        Parameters
        ----------
        idx : bool array, selects a subset of slits
        coord_transform : f: float -> float

        Returns
        -------
        edges : [N by 2] float array, each row has [left_edge, right_edge]
        '''
        if coord_transform is None:
            f = lambda x: x
        else:
            f = coord_transform
        return np.column_stack([f(self.slits.left_edge[idx]), f(self.slits[idx].right_edge)])

    def slit_lengths(self, idx=slice(None)):
        return np.abs(self.slits[idx].left_edge - self.slits[idx].right_edge)

    def match_traces(self, left_traces, right_traces, idx=slice(None)):
        '''
        Matches the given traces with slits.  
        Assumes that the passed traces have the shape (n_pix, n_traces),
        where n_pix is the number of pixels along the dispersion direction.

        Parameters
        ----------
        left_traces : float array of left (lower) edge in mask coords
        right_traces : float array of right (upper) edge in mask coords
        idx : bool array, selects a subset of slits

        Returns
        -------
        slits : record array of slits from Slitmask
        '''
        assert left_traces.shape == right_traces.shape
        n_pix, n_traces = left_traces.shape
        
        slits = self.slits[idx]
        n_slits = len(slits)
        # get indices of slitmask.slits that match the input traces
        # left_edges and right_edges are from the mask design file
        # tile up to shape (n_slits, n_pix)
        left_edges = np.tile(slits.left_edge, (n_pix, 1)).T
        right_edges = np.tile(slits.right_edge, (n_pix, 1)).T
        slit_idx = []
        # left/right trace have size n_pix
        for trace_idx, left_trace, right_trace in enumerate(zip(left_traces.T, right_traces.T)):
            # tile traces up to shape (n_slits, n_pix)
            left_trace = np.tile(left_trace, (n_slits, 1))
            right_trace = np.tile(right_trace, (n_slits, 1))
            overlaps = np.minimum(right_edges, right_trace) - np.maximum(left_edges, left_trace)
            # collapse over n_pix
            overlap = np.sum(overlaps, axis=1)
            slit_idx.append(np.argmax(overlap))
        return slits[slit_idx]
        
    
class DEIMOS_slitmask(Slitmask):
    '''
    DEIMOS slit mask design.

    DEIMOS multi-extension fits files should have at least the following named 
    binary tables:
        MaskDesign - a single line containing the mask name and other details
        ObjectCat - a record of accepted objects from the DSIMULATOR input file
        DesiSlits - design parameters for each slit
        SlitObjMap - a record of the mapping between slit IDs and object IDs
        BluSlits - contains the slit corners (in mm on mask)
    See http://deep.ps.uci.edu/DR4/bintabs.html for more details.
    '''

    def __init__(self, hdulist, plate_scale=0.1185, mm_per_arcsec=0.73089981, pix_per_chip=2048):
        '''
        Parameters
        ----------
        hdulist : either a string, in which case it corresponds to 
                  the fits filename, or an HDUList instance from
                  astropy's fits module
        plate_scale : the DEIMOS plate scale, arcseconds per pixel
        mm_per_arcsec : mm on mask per arcsec subtended on sky
        pix_per_chip : number of pixels along the spatial direction of each CCD
        '''
        if isinstance(hdulist, str):
            hdulist = fits.open(hdulist)
        else:
            assert isinstance(hdulist, fits.hdu.hdulist.HDUList)
        
        # mask attributes
        self._plate_scale = plate_scale # arcsec per pix
        self._mm_per_arcsec = mm_per_arcsec
        self._pix_per_chip = pix_per_chip
        self._pix_per_mm = 1 / (plate_scale * mm_per_arcsec) # pix per mm

        # binary tables, loaded as numpy record arrays
        mask_design = hdulist['MaskDesign'].data
        desi_slits = hdulist['DesiSlits'].data
        blu_slits = hdulist['BluSlits'].data
        obj_cat = hdulist['ObjectCat'].data
        slit_obj_map = hdulist['SlitObjMap'].data

        # sort by dSlitId
        desi_slits = desi_slits[np.argsort(desi_slits['dslitid'])]
        blu_slits = blu_slits[np.argsort(blu_slits['dslitid'])]
        slit_obj_map =  slit_obj_map[np.argsort(slit_obj_map['dslitid'])]
        
        mask_name = mask_design[0]['DesName']
        mask_pa = mask_design[0]['PA_PNT']             # in degrees ccw of North
        mask_ra = mask_design[0]['RA_PNT']             # in degrees
        mask_dec = mask_design[0]['DEC_PNT']           # in degrees

        # construct slits as numpy record array
        nslits = len(desi_slits)
        obj_idx = np.array([(i in slit_obj_map['ObjectId']) for i in obj_cat['ObjectId']])
        name = obj_cat[obj_idx]['OBJECT'].strip()
        ra = desi_slits['slitRA']                      # in degrees
        dec = desi_slits['slitDEC']                    # in degrees
        pa = desi_slits['slitLPA']                     # in degrees ccw of North
        isalign = (desi_slits['slitTyp'] == 'A')
        length = desi_slits['slitLen']
        width = desi_slits['slitWid']
        # make sure to convert from mm on mask to arcsec!
        left_edge = blu_slits['SlitX2'] / mm_per_arcsec
        right_edge = blu_slits['SlitX1'] / mm_per_arcsec
        x = (blu_slits['SlitX1'] + blu_slits['SlitX2'] + blu_slits['SlitX3'] + blu_slits['SlitX4'])/ (4 * mm_per_arcsec)
        y = (blu_slits['SlitY1'] + blu_slits['SlitY2'] + blu_slits['SlitY3'] + blu_slits['SlitY4'])/ (4 * mm_per_arcsec)
        # check for correct-ish slit length... don't trust to more than 0.3 arcsec, or ~2 pix
        assert np.all(np.isclose(np.abs(left_edge - right_edge), length, atol=0.5))
        df = pd.DataFrame(columns=['name', 'x', 'y', 'ra', 'dec', 'left_edge', 'right_edge',
                                   'pa', 'width', 'length', 'isalign'],
                             data=np.array([name, x, y, ra, dec, left_edge, right_edge, pa, width, length, isalign]).T)
        dtypes = [str, float, float, float, float, float, float, float, float, float, bool]
        for i, col in enumerate(df.columns):
            if dtypes[i] is bool:
                df[col] = map(lambda s: True if s == 'True' else False, df[col])
            df[col] = df[col].astype(dtypes[i])
        super(type(self), self). __init__(mask_name, mask_ra, mask_dec, mask_pa, df)

    @property
    def pix_per_chip(self):
        return self._pix_per_chip

    @property
    def mm_per_arcsec(self):
        return self._mm_per_arcsec
    
    @property
    def plate_scale(self):
        return self._plate_scale

    def offset_for_det(self, det):
        '''
        Fudging the coordinate transforms from mask to pixel values.
        Add the offset to the measured slit pixel values for the "correct" answer.
        '''
        if det == 1:
            return 85
        elif det == 2:
            return -3
        elif det == 3:
            return -95
        elif det == 4:
            return -174
        elif det == 5:
            return -55
        elif det == 6:
            return 29
        elif det == 7:
            return 123
        elif det == 8:
            return 201
        else:
            raise ValueError('det needs to be in 1-8!')
        
    def mosaic_location(self, det):
        '''
        Locates the chip within the DEIMOS detector mosaic.

        Parameters
        ----------
        det : int, detector index, measured from 1, matching the 'detXX' syntax 
              in the settings.deimos file

        Returns
        -------
        row : int, bottow (0) is blue side, top (1) is red side
        col : int, left to right
        '''
        # check that the requested detector(s) make(s) sense
        try:
            good = det in range(1, 9)
        except ValueError:
            # det should be iterable
            good = all([d in range(1, 9) for d in det])
            det = np.array(det)
        assert good
        return divmod(det - 1, 4)
    

    def mask_to_det_coords(self, x, det):
        '''
        Transform mask spatial arcsec coordinates to pixel coordinates of the 
        detector.

        Parameters
        ----------
        x : float, arcsec in mask coord, can be array
        det : int, detector index, measured from 1, matching the 'detXX' syntax 
              in the settings.deimos file

        Returns
        -------
        pix_x : float, detector pixel value of mask coord, will be array if 
                input is an array
        '''
        row, col = self.mosaic_location(det)
        # bottom row detectors are oriented with origin in blc right-handed
        pix_x = x / self.plate_scale - (col - 2) * self.pix_per_chip
        if row == 1:
            # top row detectors are oriented with origin in trc, right-handed
            pix_x = self.pix_per_chip - pix_x
        return pix_x + self.offset_for_det(det)


    def det_to_mask_coords(self, pix_x, det):
        '''
        Transforms detector pixel coordinate to mask coordinate in arcsec.

        Parameters
        ----------
        pix_x : float, detector pixel value of mask coord, can be array
        det : int, detector index, measured from 1, matching the 'detXX' syntax 
              in the settings.deimos file

        Returns
        -------
        x : float, arcsec in mask coord, will be array if pix_x is one
        '''

        row, col = self.mosaic_location(det)
        pix_x -= self.offset_for_det(det)
        if row == 1:
            pix_x = self.pix_per_chip - pix_x
        x = (pix_x + (col - 2) * self.pix_per_chip) * self.plate_scale
        return x 
            

    def slits_in_det(self, det):
        '''
        Gets a boolean array corresponding to slits falling on the specified
        detector.

        Parameters
        ----------
        det : int, detector index, measured from 1, matching the 'detXX' syntax 
              in the settings.deimos file
        
        Returns
        -------
        slits : boolean array of size nslits
        '''
        row, col = self.mosaic_location(det)
        if row == 0:
            left = 0
            right = self.pix_per_chip
        else:
            left = self.pix_per_chip
            right = 0
        chip_left_edge = self.det_to_mask_coords(left, det)
        chip_right_edge = self.det_to_mask_coords(right, det)

        left_edges, right_edges = super(type(self), self).slit_edges().T
        slits = (chip_left_edge < left_edges) & (right_edges < chip_right_edge)
        return slits
    
    def slit_edges(self, det, idx=None):
        '''
        Gets the pixel values of slit edges in a detector.

        Parameters
        ----------
        det : int, detector index, measured from 1, matching the 'detXX' syntax 
              in the settings.deimos file

        Returns
        -------
        edges : [N by 2] float array, each row has [left_edge, right_edge] in pixel values
        '''
        if idx is None:
            idx = self.slits_in_det(det)
        coord_transform = lambda x: self.mask_to_det_coords(x, det)
        edges = super(type(self), self).slit_edges(idx, coord_transform)
        # flip L/R if det is in upper row
        if det >= 5:
            return edges[:, ::-1]
        return edges

