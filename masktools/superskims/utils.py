'''
Utility functions for superskims mask making.
'''

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from itertools import cycle
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u

def run(name, r_eff, axial_ratio, position_angle, num_masks,
        mu_eff=22.0, ra=None, dec=None):
    '''
    Wrapper to run the superskims mask making code on a given galaxy.

    Parameters
    ----------
    name: str, galaxy name
    
    ra: float, degrees J2000
    dec: float, degrees J2000
    
    '''
    from .outputs import save_to_dsim
    from .galaxy import Galaxy
    try:
        from astroquery.ned import Ned
        t = Ned.query_object(name)
        ra = t['RA(deg)'][0]
        dec = t['DEC(deg)'][0]
    except ImportError:
        ra = ra
        dec = dec
    center = SkyCoord(ra, dec, unit='deg')
    galaxy = Galaxy(name, center, r_eff, axial_ratio, position_angle, mu_eff)
    galaxy.optimize(num_masks, num_iter=100, resolution=0.5)
    for mask in galaxy.masks:
        output_file = mask.name + '_PA{:0.1f}_superskims.dsim'.format(mask.mask_pa)
        save_to_dsim(mask, galaxy.center, output_file)


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


def mu_sersic(R, mu_eff, Re, n):
    '''Sersic surface brightness profile in mag/arcsec2'''
    return mu_eff + 2.5 * b_cb(n) / np.log(10) * ((R / Re)**(1. / n) - 1)

    
def a_ellipse(r, theta, axial_ratio):
    '''
    Finds the semi-major axis of the ellipse with axial_ratio corresponding to polar coords (r, theta).
    This is de-projecting from the sky coordinate (r, theta) to the circularized radial coordinate in the
    galaxy's frame of reference.
    '''
    return r / axial_ratio * np.sqrt(np.sin(theta)**2 + axial_ratio**2 * np.cos(theta)**2)


def sersic_profile_function(mu_eff, r_eff, n, position_angle, axial_ratio):
    '''
    Creates a function, f: (radius, angle) -> surface brightness.
    I0, Re, n are the Sersic parameters, axial_ratio is the ratio of the minor to major axes,
    position angle is in degrees east of north, surface brightness is in mag / arcsec2

    To be used in the evaluation of surface brightness profiles in mask making.
    r should be in arcsec, and theta should be in degrees
    '''
    def func(r, theta):
        # angle relative to major axis, in radians
        theta_canonical = np.radians(theta - position_angle)
        R = a_ellipse(r, theta_canonical, axial_ratio)
        return mu_sersic(R, mu_eff, r_eff, n)
    return func


def mask_to_sky(x, y, mask_pa):
    '''Convert mask x, y coordinates (along major and minor axis) to sky ra and dec coordinates.'''
    theta = np.pi + np.radians(mask_pa + 90)
    ra = -np.cos(theta) * x + -np.sin(theta) * y
    dec = -np.sin(theta) * x + np.cos(theta) * y
    return ra, dec
    

