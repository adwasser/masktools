from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from astropy.coordinates import SkyCoord
from astropy import units as u

__all__ = ['table_to_regions']

def table_to_regions(table, writeto=None, color='red'):
    '''
    Parameters
    ----------
    table  : astropy Table object with USNO entries
    writeto : str, filename for output
    color : str, as recognized by ds9

    Returns
    -------
    None
    '''
    coords = SkyCoord(table['RAJ2000'], table['DEJ2000'])
    names = table['USNO-B1.0']
    width = '4"'
    height = '4"'
    pa = '0'
    with open(writeto, 'w') as f:
        f.write('# Region file format: DS9 version 4.1\n')
        f.write('global color=' + color + ' move=0 \n')
        f.write('j2000\n')
        for i in range(len(coords)):
            ra, dec = coords[i].to_string('hmsdms', sep=':').split()
            name = names[i]
            line = 'box(' + ', '.join([ra, dec, width, height, pa]) + ') # text={' + name + '}\n'
            f.write(line)
