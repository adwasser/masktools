from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

class Slit:
    
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
