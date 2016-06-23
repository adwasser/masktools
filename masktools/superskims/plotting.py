from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from itertools import cycle
import matplotlib as mpl
from matplotlib import pyplot as plt

from .utils import mask_to_sky

__all__ = ['plot_mask', 'plot_galaxy']

def slit_patches(mask, color=None, sky_coords=False, reverse=False):
    '''Constructs mpl patches for the slits of a mask.  If sky_coords is true, output in relative ra/dec'''
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
