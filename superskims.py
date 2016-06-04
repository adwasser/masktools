'''
superskims.py

This script is for creating distributions of SKiMS (Stellar Kinematics with Multiple Slits) 
which optimally sample the integrated stellar light of a galaxy.

Original author: Nicola Pastorello
Contributing author: Asher Wasserman
   contact: adwasser@ucsc.edu
'''

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np
from astropy.coordinates import SkyCoord

def b_cb(n):
    '''
    'b' parameter in the Sersic function, from the Ciotti & Bertin (1999) 
    approximation.
    '''
    return - 1. / 3 + 2. * n + 4 / (405. * n) + 46 / (25515. * n**2)


def I_sersic(R, I0, Re, n):
    '''
    Sersic surface brightness profile.
    R is the projected radius at which to evaluate the function.
    I0 is the central brightness.
    Re is the effective radius (at which half the luminosity is enclosed).
    n is the Sersic index.
    '''
    try:
        I = I0 * np.exp(-b_cb(n) * (R / Re)**(1. / n))
    except FloatingPointError, e:
        print(e)
        I = 0
    return I

def I_n4551(r, theta):
    # NGC4551   188.908249    12.264010   2     5     1     1176   16.1  -22.18    0.17   -4.9    1.22
    distance = 16.1e6 # pc
    Mk = -22.18
    mk = Mk + 5 * (np.log10(distance) - 1)
    mk_vega = 
    Reff = 10**1.22
    I0 = 1.53569e+11

    
class Galaxy:
    '''This is a representation of a galaxy which needs slitmasks.'''

    def __init__(self, name, center, r_eff, axial_ratio, position_angle, brightness_profile):
        '''
        Parameters
        ----------
        name: str, name of galaxy, e.g. 'n4551', used in file output labeling
        center: SkyCoord object of central position
        r_eff: float, in arcseconds, giving the effective radius of galaxy
        axial_ratio: float, ratio of minor axis to major axis, equal to (1 - flattening)
        position_angle: float, in degrees, giving the position angle
                        measured counter-clockwise from north (i.e. positive declination)
        brightness_profile: f: radius in arcsec, position angle in degrees -> 
                            surface brightness in mag/arcsec^2
        '''
        self.name = name
        self.center = center
        self.r_eff = r_eff
        self.axial_ratio = axial_ratio
        self.position_angle = position_angle
        self.brightness_profile = brightness_profile
        self.masks = []

    def __repr__(self):
        return self.name + ': ' + self.center.to_string('hmsdms')
        
    def create_masks(self, num_masks):
        '''
        Parameters
        ----------
        num_masks: int, number of masks to make for the galaxy
        '''
        self.masks = []
        cone_angle = 180. / num_masks
        for i in range(num_masks):
            delta_pa = i * cone_angle
            mask_pa = self.position_angle + delta_pa
            mask_r_eff = np.sqrt((self.r_eff * np.cos(np.radians(delta_pa)))**2 +
                                 (self.r_eff * self.axial_ratio * np.sin(np.radians(delta_pa)))**2)
            self.masks.append(Mask(mask_pa, mask_r_eff, cone_angle, self.brightness_profile))

    def slit_positions(self, best=False):
        '''
        Returns the slit positions (x, y), rotated to the galaxy frame, (i.e., the x-axis
        is along the major axis and the y-axis is along the minor axis).

        if best, then get slit positions for the best fitting slits
        '''
        # list of positions rotated to the major axis of galaxy
        x_positions = np.array([])
        y_positions = np.array([])
        for mask in self.masks:
            theta = np.radians(mask.mask_pa - self.position_angle)
            if best:
                x = np.array([slit.x for slit in mask.best_slits])
                y = np.array([slit.x for slit in mask.best_slits])
            else:
                x, y = mask.slit_positions()
            x_rot = x * np.cos(theta) - y * np.sin(theta)
            y_rot = x * np.sin(theta) + y * np.cos(theta)
            x_positions = np.concatenate([x_positions, x_rot, -x_rot])
            y_positions = np.concatenate([y_positions, y_rot, -y_rot])                
        return x_positions, y_positions
            
    def sampling_metric(self, xx, yy, resolution):
        '''
        Evaluates how well the given points sample the 2d space.

        Parameters
        ----------
        xx: float array, arcsec along long end of mask
        yy: float array, arcsec along short end of mask
        resolution: float, arcsec, spacing between points in spatial grid

        Returns
        -------
        metric: float, sum of minimum distances between spatial grid and slit samples
        '''

        assert len(xx) == len(yy)
        # take only points on one side of minor axis
        mask = xx >= 0
        xx = xx[mask]
        yy = yy[mask]
        num_slits = len(xx)
        
        # make grid samples
        x_samples = np.linspace(0, np.amax(xx), int(np.amax(xx) / resolution))
        y_samples = np.linspace(np.amin(yy), np.amax(yy), int(np.ptp(yy) / resolution))

        # flatten grid
        x_flat = np.tile(x_samples, y_samples.shape)
        y_flat = np.tile(y_samples, (x_samples.size, 1)).T.flatten()
        num_points = len(x_flat)
        
        # tile grid to n_points by n_slits
        x_points = np.tile(x_flat, (num_slits, 1))
        y_points = np.tile(y_flat, (num_slits, 1))

        # tile slit positions to n_points by n_slits
        x_slits = np.tile(xx, (num_points, 1)).T
        y_slits = np.tile(yy, (num_points, 1)).T        
        
        distances = np.amin(np.sqrt((x_slits - x_points)**2 +
                                    (y_slits - y_points)**2),
                            axis=0)
        return np.sum(distances)

    def optimize(self, num_masks=2, num_iter=100, resolution=0.5):
        '''
        Find the optimal spatial sampling of mask slits.

        To do: figure out how to best reference entire galaxy spatial extent, not just by mask coords.

        Parameters
        ----------
        num_masks: int, number of masks to make for the galaxy
        num_iter: int, number of iterations in MC
        resolution: float, arcsec, spatial resolution of mask area to sample for MC
        '''
        self.create_masks(num_masks)
        # iteratively randomize slit distribution and check spatial sampling
        best_result = np.inf
        for i in range(num_iter):
            # print(i)
            # randomize slits
            for mask in self.masks:
                mask.random_slits()
            # list of positions rotated to the major axis of galaxy
            x_positions, y_positions = self.slit_positions()
            metric = self.sampling_metric(x_positions, y_positions, resolution)
            # minimize metric
            if metric < best_result:
                # copy current slit configuration to best setup
                for mask in self.masks:
                    # cleanup first
                    # del mask.best_slits[:]
                    # storage next
                    mask.best_slits = mask.slits
                best_result = metric
        return best_result

class Mask:
    '''Represents a slitmask'''
    
    def __init__(self, mask_pa, mask_r_eff, cone_angle, brightness_profile,
                 slit_separation=0.5, slit_width=1, min_slit_length=3,
                 max_radius_factor=5, sky_spacing=50):
        '''
        Parameters
        ----------
        mask_pa: float, degrees east of north
        mask_r_eff: float, arcsec, effective radius along the mask position angle
        cone_angle: float, degrees, the opening angle of the slit spatial distribution
        brightness_profile: f: radius in arcsec, position angle in degrees -> 
                            surface brightness in mag/arcsec^2
        slit_separation: float, arcsec, minimum separation between slits
        slit_width: float, arcsec, width of slit, should not be less than 1 arcsec
        min_slit_length: float, arcsec, the minimum slit length
        max_radius_factor: float, factors of Reff to which to extend the skims
        sky_spacing: float, arcsec, how far from the edge of the mask to place the sky slits
        '''
        # x_max, y_max, are the maximum spatial extent of the masks, in arcsec
        self.x_max = 498
        self.y_max = 146
        self.mask_pa = mask_pa
        self.mask_r_eff = mask_r_eff
        self.cone_angle = cone_angle
        self.brightness_profile = brightness_profile
        self.slit_separation = slit_separation
        self.slit_width = slit_width
        self.min_slit_length = min_slit_length
        self.max_radius_factor = max_radius_factor
        self.sky_spacing = sky_spacing
        self.slits = []
        self.best_slits = []

        
    def __repr__(self):
        mask_params_str = '<Mask -- PA: {0:.2f}, Reff: {1:.2f}, Cone angle: {2:.2f}>'
        return mask_params_str.format(self.mask_pa, self.mask_r_eff, self.cone_angle)

    
    def get_slit_length(self, x, y, snr=35., sky=19, integration_time=7200, plate_scale=0.1185,
                        gain=1.2, read_noise=2.5, dark_current=4, imag_count_20=1367):
        '''
        Determine how long the slit should be, based on the required signal-to-noise ratio.

        Default signal-to-noise ratio is set by kinematic requirements.
        Default sky background is for dark sky conditions in I band.
        Default time is for two hours.
        Plate scale is set for DEIMOS.
        Gain, read noise, and dark current are rough estimates from 
            http://www2.keck.hawaii.edu/inst/deimos/deimos_detector_data.html
        I band counts at I = 20 are from LRIS, but should be close to DEIMOS, see
            http://www2.keck.hawaii.edu/inst/deimos/lris_vs_deimos.html

        To do: calibrate what value counts should have for a desired signal-to-noise ratio

        Parameters
        ----------
        x: float, arcsec, x coordinate
        y: float, arcsec, y coordinate
        snr: float, desired signal-to-noise ratio
        sky: float, brightness of sky in mag/arcsec^2, default (sky=19) is a wild and crazy guess
        integration_time: float, seconds
        plate_scale: float, arcsec per pixel
        gain: float, e- counts per ADU
        read_noise: float, e- counts
        dark_current: float, e- counts per pix per hour
        imag_count_20: float, e- counts per second at I = 20 mag
        '''
        radius = np.sqrt(x**2 + y**2)
        angle = self.mask_pa + np.degrees(np.arctan(y/x))
        source_sb = self.brightness_profile(radius, angle)

        # convert to e- per second per pix^2
        source_flux = imag_count_20 * 10**(0.4 * (20 - source_sb)) * plate_scale**2
        sky_flux =  imag_count_20 * 10**(0.4 * (20 - sky)) * plate_scale**2

        dark = dark_current * 3600.
        denominator = (read_noise**2 + (gain / 2)**2 +
                       integration_time * (source_flux + sky_flux + dark))
        npix = snr**2 * denominator * integration_time**2 * source_flux
        area = npix * plate_scale**2
        length = area / self.slit_width
        return length
    

    def slit_positions(self):
        '''
        Returns arrays with x, y positions of slits.
        '''
        xx = np.array([slit.x for slit in self.slits]) 
        yy = np.array([slit.y for slit in self.slits])
        return xx, yy
        
        
    def random_slits(self):
        '''
        Produce a random alignment (satisfying the opening angle restriction), with slit lengths
        satisfying a signal-to-noise requirement.
        '''
        # reset slits
        self.slits = []
        x = self.slit_separation / 2.
        count = 0
        while x < self.mask_r_eff * self.max_radius_factor:
            y_cone = np.tan(np.radians(self.cone_angle / 2.)) * x
            y = np.random.uniform(-y_cone, y_cone)
            length = max(self.min_slit_length, self.get_slit_length(x, y))
            self.slits.append(Slit(x, y, length, self.mask_pa + 5.,
                                   'skims{0:02d}'.format(count)))
            count += 1
            x += length + self.slit_separation

    def add_sky_slits(self):
        '''
        Place sky slits on the mask
        '''
        x = self.x_max - self.sky_spacing
        count = 0
        while x < self.x_max:
            y_cone = np.tan(np.radians(self.cone_angle / 2.)) * x
            y = np.random.uniform(-y_cone, y_cone)
            self.slits.append(Slit(x, y, self.min_slit_length, self.mask_pa + 5.,
                                   name='sky{0:02d}'.format(count)))
            count += 1
            x += length + self.slit_separation

    def within_mask(self, x, y):
        x = np.abs(x)
        a = (y < -1) | (x < 360)
        b = (360 <= x) & (x < 420) & (y < -0.85*x+452.)
        c = (420. <= x) & (x < 460.) & (y < -1.075*x+546.5)
        d = (460. <= x) & (x < 498.) & (y < -1.9347368421*x+693.5789473684)
        return a | b | c | d


    def within_cones(self, x, y):
        x = np.abs(x)
        yline1 = np.tan(self.cone_angle / 2. * np.pi/180.) * np.array(x)
        yline2 = -np.tan(self.cone_angle / 2. * np.pi/180.) * np.array(x)
        return (yline2 < y) & (y < yline1)


    def within_slits(self, x, y):
        return np.sqrt(x**2 + y**2) <= self.mask_r_eff * self.max_radius_factor 

class Slit:
    
    def __init__(self, x, y, length, pa, name):
        '''
        Representation of a slit in a mask.  Coordinates are relative to the mask, so that
        the x-axis is along the long end and the y-axis is along the short end.
        
        Parameters
        ----------
        x: float, arcsec along long end of mask
        y: float, arcsec along short end of mask
        length: float, arcsec, slit length, should be a minimum of 3
        pa: float, degrees, position angle of slit, relative to sky (i.e., 0 is north, 90 is east)
        name: string, unique (within mask) identifier
        '''
        self.x = x
        self.y = y
        self.length = length
        self.pa = pa
        self.name = name

    def __repr__(self):
        info_str = ': length of {0:.2f}, PA of {1:.2f} at ({2:.2f}, {3:.2f})'
        return '<' + self.name + info_str.format(self.length, self.pa, self.x, self.y) + '>'
        
