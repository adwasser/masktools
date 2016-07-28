'''
Module for querying data from the USNO-B1.0 catalog.
'''
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from astroquery.vizier import Vizier
from astropy.coordinates import Angle

__all__ = ['get_table']

def get_table(name, radius=10, pick=None,
              pm_threshold=40, align_mag_bounds=(14, 19), guide_mag_bounds=(12, 17)):
    '''
    Parameters
    ----------
    name : str, galaxy name around which to query Vizier for align/guide stars,
           e.g., 'NGC 1052'
    radius : float, arcminutes around which to search for stars
    pick : if None, return all within radius, if "align", choose for align stars,
           if "guide", choose for guide stars
    pm_threshold : float, milliarcsec per year, the maximum proper motion allowed
    align_mag_bounds : tuple of (lower i mag, upper i mag) allowed for align stars
    guide_mag_bounds : tuple of (lower i mag, upper i mag) allowed for guide stars
    
    Returns
    -------
    table : astropy Table object with USNO entries
    '''

    if pick not in [None, 'align', 'guide']:
        raise ValueError('pick needs to be one of {None, "align", "guide"}')
    
    # to select all objects instead of just the first 50
    Vizier.ROW_LIMIT = -1 
    result = Vizier.query_region(name, radius=Angle(radius, 'arcmin'), catalog='USNO-B1.0')
    assert len(result) == 1
    table = result[0]
    if pick is None:
        return table
    pmRA = table['pmRA']
    pmDec = table['pmDE']
    # remove high PM objects
    pm_okay = np.sqrt(pmRA**2 + pmDec**2) < 40 # units should be mas / yr
    table = table[pm_okay]
    
    imag = table['Imag']
    if pick == 'align':
        imag_okay = (align_mag_bounds[0] < imag) & (imag < align_mag_bounds[1])
    elif pick == 'guide':
        imag_okay = (guide_mag_bounds[0] < imag) & (imag < guide_mag_bounds[1])

    return table[imag_okay]
    
