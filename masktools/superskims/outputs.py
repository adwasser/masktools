from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

__all__ = ['save_to_regions', 'save_to_dsim']

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from .utils import mask_to_sky

def save_to_regions(mask, writeto=None):
    raise NotImplementedError


def save_to_dsim(mask, center, writeto=None):
    '''
    mask is a Mask, center is a SkyCoord, writeto is the output file name
    '''
    with open(writeto, 'w') as f:
        ra_str, dec_str = center.to_string('hmsdms').split(' ')
        name = mask.name + '_PA{:0.1f}'.format(mask.mask_pa)
        header = '\t'.join([name, ra_str, dec_str, '2000.0', 'PA={:.2f}'.format(mask.mask_pa)]) + '\n\n'
        f.write(header)
        x, y = mask.slit_positions()
        ra_offsets, dec_offsets = mask_to_sky(x, y, mask.mask_pa)
        ra = (ra_offsets / np.cos(center.dec.radian) + center.ra.arcsec) * u.arcsec
        dec = (dec_offsets + center.dec.arcsec) * u.arcsec
        coords = SkyCoord(ra, dec)
        for i, slit in enumerate(mask.slits):
            name = slit.name + ' ' * (16 - len(slit.name))
            ra, dec = coords[i].to_string('hmsdms', sep=':').split()
            pa = '{:.2f}'.format(slit.pa)
            half_len = '{:.2f}'.format(slit.length / 2)
            width = '{:.2f}'.format(slit.width)
            line = name + '\t'.join([ra, dec, '2000.0', '0', 'R', '100', '1',
                                     pa, half_len, width]) + '\n'
            f.write(line)
