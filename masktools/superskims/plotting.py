from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from itertools import cycle
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from .utils import mask_to_sky

__all__ = ['plot_mask', 'plot_galaxy']

def slit_patches(mask, color=None, sky_coords=False, center=None):
    '''
    Constructs mpl patches for the slits of a mask.  If sky_coords is true, 
    output in relative ra/dec.  galaxy center is necessary for sky_coords
    '''
    patches = []
    for slit in mask.slits:
        x = slit.x
        y = slit.y
        dx = slit.length
        dy = slit.width
        # bottom left-hand corner
        if sky_coords:
            L = np.sqrt(dx**2 + dy**2) / 2
            alpha = np.tan(dy / dx)
            phi = np.pi / 2 - np.radians(slit.pa)
            delta_x = L * (np.cos(alpha + phi) - np.cos(alpha))
            delta_y = L * (np.sin(alpha + phi) - np.sin(alpha))
            ra, dec = mask_to_sky(x - dx / 2, y - dy / 2, mask.mask_pa)
            blc0 = (ra, dec)
            angle = (90 - slit.pa)
            blc = (ra + delta_x, dec - delta_y)
            # blc = (ra + x1 + x2, dec - y1 + y2)            
        else:
            blc = (x - dx / 2, y - dy / 2)
            angle = slit.pa - mask.mask_pa
        patches.append(mpl.patches.Rectangle(blc, dx, dy, angle=angle,
                                             fc=color, ec='k', alpha=0.5))
        # patches.append(mpl.patches.Rectangle(blc0, dx, dy, angle=0,
        #                                      fc=color, ec='k', alpha=0.1))
    return patches


def plot_mask(mask, color=None, writeto=None, annotate=False):
    '''Plot the slits in a mask, in mask coords'''

    fig, ax = plt.subplots()
    
    for p in slit_patches(mask, color=color, sky_coords=False):
        ax.add_patch(p)
        
    if annotate:
        for slit in mask.slits:
            ax.text(slit.x - 3, slit.y + 1, slit.name, size=8)
    xlim = mask.x_max / 2
    ylim = mask.y_max / 2
    lim = min(xlim, ylim)
    ax.set_title(mask.name)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('x offset (arcsec)', fontsize=16)
    ax.set_ylabel('y offset (arcsec)', fontsize=16)
    if writeto is not None:
        fig.savefig(writeto)
    return fig, ax


def plot_galaxy(galaxy, writeto=None):
    '''Plot all slit masks'''
    fig, ax = plt.subplots()
    colors = cycle(['r', 'b', 'm', 'c', 'g'])
    handles = []
    for i, mask in enumerate(galaxy.masks):
        color = next(colors)
        label = str(i + 1) + galaxy.name + ' (PA = {:.2f})'.format(mask.mask_pa)
        handles.append(mpl.patches.Patch(fc=color, ec='k',
                                         alpha=0.5, label=label))
        for p in slit_patches(mask, color=color,
                              sky_coords=True, center=galaxy.center):
            ax.add_patch(p)
    xlim = galaxy.masks[0].x_max / 2
    ylim = galaxy.masks[0].y_max / 2
    lim = min(xlim, ylim)
    # reverse x axis so it looks like sky
    ax.set_xlim(lim, -lim)
    # ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(galaxy.name, fontsize=16)
    ax.set_xlabel('RA offset (arcsec)', fontsize=16)
    ax.set_ylabel('Dec offset (arcsec)', fontsize=16)
    ax.legend(handles=handles, loc='best')
    if writeto is not None:
        fig.savefig(writeto) #, bbox_inches='tight')
    return fig, ax
