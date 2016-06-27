from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from astropy.coordinates import SkyCoord

from .utils import sersic_profile_function
from .mask import Mask

class Galaxy:
    '''This is a representation of a galaxy which needs slitmasks.'''

    def __init__(self, name, center, r_eff, axial_ratio, position_angle,
                 mu_eff=22.0, brightness_profile=None):
        '''
        Parameters
        ----------
        name: str, name of galaxy, e.g. 'n4551', used in file output labeling
        center: SkyCoord object of central position
        r_eff: float, in arcseconds, giving the effective radius of galaxy
        axial_ratio: float, ratio of minor axis to major axis, equal to (1 - flattening)
        position_angle: float, in degrees, giving the position angle
                        measured counter-clockwise from north (i.e. positive declination)
        mu_eff: float, in mag [per arcsec2], giving the surface brightness at r_eff
        brightness_profile: f: radius in arcsec, position angle in degrees -> 
                            surface brightness in mag/arcsec^2
                            If None, default to de Vaucouleurs' profile
        '''
        self.name = name
        assert isinstance(center, SkyCoord)
        self.center = center
        self.r_eff = r_eff
        self.mu_eff = mu_eff
        self.axial_ratio = axial_ratio
        self.position_angle = position_angle
        if brightness_profile is not None:
            self.brightness_profile = brightness_profile
        else:
            # default to deVaucouleurs' profile
            self.brightness_profile = sersic_profile_function(mu_eff, r_eff, 4, position_angle, axial_ratio)
        self.masks = []

    def __repr__(self):
        return '<Galaxy ' + self.name + ': ' + self.center.to_string('hmsdms') + '>'
        
    def create_masks(self, num_masks, mask_cone_angles=None, cone_overlap=180.):
        '''
        Parameters
        ----------
        num_masks: int, number of masks to make for the galaxy
        mask_cone_angles: list, if not None, sets the individual opening angles for each mask
        cone_overlap: float, degrees, = cone_angle * num_masks
        '''
        self.masks = []
        if mask_cone_angles is not None:
            assert len(mask_cone_angles) == num_masks
            cone_angles = mask_cone_angles
        else:
            cone_angle = cone_overlap / num_masks
            cone_angles = [cone_angle] * num_masks
        sep_angle = 180. / num_masks
        for i in range(num_masks):
            delta_pa = i * sep_angle
            mask_pa = self.position_angle + delta_pa
            mask_r_eff = np.sqrt((self.r_eff * np.cos(np.radians(delta_pa)))**2 +
                                 (self.r_eff * self.axial_ratio * np.sin(np.radians(delta_pa)))**2)
            name = str(i + 1) + self.name
            self.masks.append(Mask(name, mask_pa, mask_r_eff, cone_angles[i], self.brightness_profile))

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
        metric: float, mean of minimum distances between spatial grid and slit samples
        '''

        assert len(xx) == len(yy)
        # take only points on one side of minor axis
        # mask = xx >= 0
        # xx = xx[mask]
        # yy = yy[mask]
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
        return np.amax(distances)

    def optimize(self, num_masks=4, num_iter=100, resolution=1, cone_angles=None, cone_overlap=180):
        '''
        Find the optimal spatial sampling of mask slits.

        Parameters
        ----------
        num_masks: int, number of masks to make for the galaxy
        num_iter: int, number of iterations in MC
        resolution: float, arcsec, spatial resolution of mask area to sample for MC
        '''
        self.create_masks(num_masks, cone_angles, cone_overlap)
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
        # add sky slits and mirror the final results
        for mask in self.masks:
            mask.add_sky_slits()
            mask.mirror_slits()
        return best_result
