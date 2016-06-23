from __future__ import division
import numpy as np
from astropy.coordinates import SkyCoord
from superskims import Galaxy

name = 'N4551'
center = SkyCoord('12h35m37.9s +12d15m50s')
r_eff = 16.6 # arcsec
axial_ratio = 0.75
position_angle = 70.5

g = Galaxy(name, center, r_eff, axial_ratio, position_angle)
g.create_masks(2)
m = g.masks[0]
