from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np

from masktools.slitmask import Slitmask
# from .slit import Slit

class Slit(object):
    def __init__(self, x, y, length, width, pa, name):
        '''
        Representation of a slit in a mask.  Coordinates are relative to the mask, so that
        the x-axis is along the long end and the y-axis is along the short end.
        
        Parameters
        ----------
        x: float, arcsec along long end of mask
        y: float, arcsec along short end of mask
        length: float, arcsec, slit length (along spatial axis), should be a minimum of 3
        width: float, arcsec, width of slit (along dispersion axis)
        pa: float, degrees, position angle of slit, relative to sky (i.e., 0 is north, 90 is east)
        name: string, unique (within mask) identifier
        '''
        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.pa = pa
        self.name = name

    def __repr__(self):
        info_str = ': length of {0:.2f}, PA of {1:.2f} at ({2:.2f}, {3:.2f})'
        return '<Slit: ' + self.name + info_str.format(self.length, self.pa, self.x, self.y) + '>'

class Slits(object):

    def __init__(self, slits):
        self.reset()
        for slit in slits:
            self.add_slit(slit)

    def __len__(self):
        return len(self._x)

    def __getitem__(self, index):
        return self._x[index], self._y[index], self._length[index], self._width[index], self._pa[index], self._name[index]
    
    def reset(self):
        self._x = []
        self._y = []
        self._length = []
        self._width = []
        self._pa = []
        self._name = []
        
    def add_slit(self, slit):
        self._x.append(slit.x)
        self._y.append(slit.y)
        self._length.append(slit.length)
        self._width.append(slit.width)            
        self._pa.append(slit.pa)
        self._name.append(slit.name)

    def append(self, x, y, length, width, pa, name):
        self._x.append(x)
        self._y.append(y)
        self._length.append(length)
        self._width.append(width)            
        self._pa.append(pa)
        self._name.append(name)
    
    @property
    def x(self):
        return np.array(self._x)

    @property
    def y(self):
        return np.array(self._y)

    @property
    def length(self):
        return np.array(self._length)

    @property
    def width(self):
        return np.array(self._width)

    @property
    def pa(self):
        return np.array(self._pa)

    @property
    def name(self):
        return np.array(self._name)

    
class SKiMS_slitmask(object):

    '''Represents a slitmask'''
    
    def __init__(self, mask_name, mask_pa, mask_r_eff, cone_angle, brightness_profile,
                 slit_separation=0.5, slit_width=1, min_slit_length=3,
                 max_radius_factor=4, angle_offset=5):
        '''
        Parameters
        ----------
        name: str, gui name of mask
        mask_pa: float, degrees east of north
        mask_r_eff: float, arcsec, effective radius along the mask position angle
        cone_angle: float, degrees, the opening angle of the slit spatial distribution
        brightness_profile: f: radius in arcsec, position angle in degrees -> 
                            surface brightness in mag/arcsec^2
        slit_separation: float, arcsec, minimum separation between slits
        slit_width: float, arcsec, width of slit, should not be less than 1 arcsec
        min_slit_length: float, arcsec, the minimum slit length
        max_radius_factor: float, factors of Reff to which to extend the skims
        angle_offset: float, degrees, rotate the slits from the mask_pa by this amount
        '''
        # x_max, y_max, are the maximum spatial extent of the masks, in arcsec
        self.x_max = 498
        self.y_max = 146
        self.mask_name = mask_name
        self.mask_pa = mask_pa
        self.mask_r_eff = mask_r_eff
        self.cone_angle = cone_angle
        self.brightness_profile = brightness_profile
        self.slit_separation = slit_separation
        self.slit_width = slit_width
        self.min_slit_length = min_slit_length
        self.max_radius_factor = max_radius_factor
        self.angle_offset = angle_offset
        self.slits = Slits([])
        self.best_slits = None
        
    def __repr__(self):
        mask_params_str = '<SKiMS_slitmask: ' + self.mask_name + ': PA: {0:.2f}, Reff: {1:.2f}, Cone angle: {2:.2f}>'
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
        mag_plate_scale = - 2.5 * np.log10(plate_scale**2)
        source_flux = imag_count_20 * 10**(0.4 * (20 - source_sb - mag_plate_scale))
        sky_flux = imag_count_20 * 10**(0.4 * (20 - sky - mag_plate_scale))

        # dark = dark_current / 3600.
        # denominator = (read_noise**2 + (gain / 2)**2 +
        #                integration_time * (source_flux + sky_flux + dark))
        npix = snr**2 * sky_flux / integration_time / source_flux**2
        area = npix * plate_scale**2
        length = area / self.slit_width
        return length

    def _test_slits(self):
        # reset slits
        self.slits.reset()
        # x is at the left edge of the slit
        x = self.slit_separation / 2.
        count = 0
        while x < self.mask_r_eff * self.max_radius_factor:
            # y_cone = np.tan(np.radians(self.cone_angle / 2.)) * x
            # y = np.random.uniform(-y_cone, y_cone)
            y = 0
            length = max(self.min_slit_length, self.get_slit_length(x, y))
            # first run gets even indices, rotated copy gets odd indices
            name = 'skims{0:02d}'.format(2 * count)
            self.slits.append(x=x + length / 2, y=y, length=length, width=self.slit_width,
                              pa=self.mask_pa + self.angle_offset, name=name)
            count += 1
            x += length + self.slit_separation
        
    def random_slits(self):
        '''
        Produce a random alignment (satisfying the opening angle restriction), with slit lengths
        satisfying a signal-to-noise requirement.
        '''
        # reset slits
        self.slits.reset()
        # x is at the left edge of the slit
        x = self.slit_separation / 2.
        count = 0
        while x < self.mask_r_eff * self.max_radius_factor:
            y_cone = np.tan(np.radians(self.cone_angle / 2.)) * x
            y = np.random.uniform(-y_cone, y_cone)
            length = max(self.min_slit_length, self.get_slit_length(x, y))
            # first run gets even indices, rotated copy gets odd indices
            name = 'skims{0:02d}'.format(2 * count)
            self.slits.append(x=x + length / 2, y=y, length=length, width=self.slit_width,
                              pa=self.mask_pa + self.angle_offset, name=name)
            count += 1
            x += length + self.slit_separation

    def add_sky_slits(self, num_sky_slits=10, sky_spacing=100):
        '''
        Place sky slits on the mask
        
        Parameters
        ----------
        num_sky_slits, int, maximum number of sky slits (per half of max) to place
        sky_spacing, float, arcsec, start placing slits this far from mask edge
        '''
        x = self.x_max - sky_spacing
        count = 0
        while x < self.x_max and count < num_sky_slits:
            y = np.random.uniform(-self.y_max, self.y_max)
            length = self.min_slit_length
            name = 'sky{0:02d}'.format(2 * count)
            self.slits.append(x=x + length / 2, y=y, length=length, width=self.slit_width,
                              pa=self.mask_pa + self.angle_offset, name=name)
            count += 1
            x += length + self.slit_separation

    def mirror_slits(self):
        '''
        Adds the mirror image of the current slits to the mask.
        '''
        nslits = len(self.slits)
        for i, slit in enumerate(self.slits):
            x, y, length, width, pa, name = slit
            slit_type = name[:-2]
            index = name[-2:]
            print(index, slit_type)
            new_name = slit_type + '{0:02d}'.format(index + 1)
            self.append(-x, -y, length, width, pa, new_name)
                    
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
    
