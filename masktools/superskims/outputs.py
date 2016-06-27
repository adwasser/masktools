from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__all__ = ['save_to_regions', 'save_galaxy_to_regions', 'save_to_dsim']

from itertools import cycle
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from .utils import mask_to_sky

def save_to_regions(mask, center, writeto=None):
    '''
    mask is a Mask, center is a SkyCoord, writeto is the output file name
    '''
    with open(writeto, 'w') as f:
        f.write('# Region file format: DS9 version 4.1\n')
        f.write('global color=red move=0 select=0\n')
        f.write('j2000\n')        
        ra_str, dec_str = center.to_string('hmsdms').split(' ')
        name = mask.name + '_PA{:0.1f}'.format(mask.mask_pa)
        x, y = mask.slit_positions()
        ra_offsets, dec_offsets = mask_to_sky(x, y, mask.mask_pa)
        ra = (ra_offsets / np.cos(center.dec.radian) + center.ra.arcsec) * u.arcsec
        dec = (dec_offsets + center.dec.arcsec) * u.arcsec
        coords = SkyCoord(ra, dec)
        for i, slit in enumerate(mask.slits):
            name = slit.name
            ra, dec = coords[i].to_string('hmsdms', sep=':').split()
            pa = '{:.2f}'.format(slit.pa)
            height = '{:.2f}'.format(slit.length) + '\"'
            width = '{:.2f}'.format(slit.width) + '\"'
            line = 'box(' + ', '.join([ra, dec, width, height, pa]) + ') # text={' + name + '}\n'
            f.write(line)

def save_galaxy_to_regions(galaxy, writeto=None, annotate=False):
    with open(writeto, 'w') as f:
        f.write('# Region file format: DS9 version 4.1\n')
        f.write('global color=red move=0 select=0\n')
        f.write('j2000\n')
        center = galaxy.center
        ra_str, dec_str = center.to_string('hmsdms').split(' ')
        colors = cycle(['red', 'green', 'blue', 'magenta', 'cyan', 'yellow'])
        for mask in galaxy.masks:
            color = next(colors)
            x, y = mask.slit_positions()
            ra_offsets, dec_offsets = mask_to_sky(x, y, mask.mask_pa)
            ra = (ra_offsets / np.cos(center.dec.radian) + center.ra.arcsec) * u.arcsec
            dec = (dec_offsets + center.dec.arcsec) * u.arcsec
            coords = SkyCoord(ra, dec)
            for i, slit in enumerate(mask.slits):
                name = mask.name[0] + slit.name
                ra, dec = coords[i].to_string('hmsdms', sep=':').split()
                pa = '{:.2f}'.format(slit.pa)
                height = '{:.2f}'.format(slit.length) + '\"'
                width = '{:.2f}'.format(slit.width) + '\"'
                line = 'box(' + ', '.join([ra, dec, width, height, pa])
                if annotate:
                    line += ') # color=' + color + ' text={' + name + '}\n'
                else:
                    line += ') # color=' + color + '\n'
                f.write(line)
    

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
            line = name + '\t'.join([ra, dec, '2000.0', '0', 'R', '100', '1', '1',
                                     pa, half_len, width]) + '\n'
            f.write(line)
