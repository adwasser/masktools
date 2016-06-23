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
import superskims

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
    
    
def slit_patches(mask, color=None, sky_coords=False, reverse=False):
    '''Constructs mpl patches for the slits of a mask.  If sky_coords is true, output in relative ra/dec'''
    assert isinstance(mask, superskims.Mask)
    patches = []
    for slit in mask.slits:
        if reverse:
            x = -slit.x
            y = -slit.y
        else:
            x = slit.x
            y = slit.y
        dx = slit.length / 2
        dy = slit.width / 2
        # bottom left-hand corner
        if sky_coords:
            blc = tuple(mask_to_sky(x - dx, y - dy, mask.mask_pa))
            angle = slit.pa + 90
        else:
            blc = (x - dx / 2, y - dy / 2)
            angle = slit.pa - mask.mask_pa
        patches.append(mpl.patches.Rectangle(blc, dx, dy, angle=angle,
                                             fc=color, ec='k', alpha=0.5))
    return patches


def plot_mask(mask, color=None, writeto=None, annotate=False):
    '''Plot the slits in a mask, in mask coords'''

    assert isinstance(mask, superskims.Mask)
    fig, ax = plt.subplots()
    
    for p in slit_patches(mask, color=color):
        ax.add_patch(p)
    # for p in slit_patches(mask, color=color, reverse=True):
    #     ax.add_patch(p)
    # ax.add_collection(pc)
    if annotate:
        for slit in mask.slits:
            ax.text(slit.x - 3, slit.y + 1, slit.name, size=8)
    xlim = mask.x_max / 2
    ylim = mask.y_max / 2
    lim = min(xlim, ylim)
    ax.set_title(mask.name)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('RA offset (arcsec)', fontsize=16)
    ax.set_ylabel('Dec offset (arcsec)', fontsize=16)
    if writeto is not None:
        fig.savefig(writeto)
    return fig, ax


def plot_galaxy(galaxy, writeto=None):
    '''Plot all slit masks'''
    assert isinstance(galaxy, superskims.Galaxy)
    fig, ax = plt.subplots()
    colors = cycle(['r', 'b', 'm', 'c', 'g'])
    handles = []
    for i, mask in enumerate(galaxy.masks):
        color = next(colors)
        label = str(i + 1) + galaxy.name + ' (PA = {:.2f})'.format(mask.mask_pa)
        handles.append(mpl.patches.Patch(fc=color, ec='k', alpha=0.5, label=label))
        for p in slit_patches(mask, color=color, sky_coords=True):
            ax.add_patch(p)
        # for p in slit_patches(mask, color=color, sky_coords=True, reverse=True):
        #     ax.add_patch(p)
    xlim = galaxy.masks[0].x_max / 2
    ylim = galaxy.masks[0].y_max / 2
    lim = min(xlim, ylim)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(galaxy.name, fontsize=16)
    ax.set_xlabel('RA offset (arcsec)', fontsize=16)
    ax.set_ylabel('Dec offset (arcsec)', fontsize=16)
    ax.legend(handles=handles, loc='best')
    if writeto is not None:
        fig.savefig(writeto) #, bbox_inches='tight')
    return fig, ax


def save_to_regions(mask, writeto=None):
    pass


def save_to_dsim(mask, center, writeto=None):
    '''
    mask is a Mask, center is a SkyCoord, writeto is the output file name
    '''
    assert isinstance(mask, superskims.Mask)
    with open(writeto, 'w') as f:
        ra_str, dec_str = center.to_string('hmsdms').split(' ') 
        header = '\t'.join([mask.name, ra_str, dec_str, '2000.0', 'PA={:.2f}'.format(mask.mask_pa)]) + '\n'
        f.write(header)
        x, y = mask.slit_positions()
        ra_offsets, dec_offsets = mask_to_sky(x, y, mask.mask_pa)
        ra = (ra_offsets / np.cos(center.dec.radian) + center.ra.arcsec) * u.arcsec
        dec = (dec_offsets + center.dec.arcsec) * u.arcsec
        coords = SkyCoord(ra, dec)
        for i, slit in enumerate(mask.slits):
            ra, dec = coords[i].to_string('hmsdms').split()
            pa = '{:.2f}'.format(slit.pa)
            half_len = '{:.2f}'.format(slit.length / 2)
            width = '{:.2f}'.format(slit.width)
            line = '\t'.join([slit.name, ra, dec, '2000.0', '0', 'R', '100', '1', pa, half_len, width]) + '\n'
            f.write(line)
