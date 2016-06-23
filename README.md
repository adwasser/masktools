masktools
=========

Utilities for designing DEIMOS slitmasks

Contact: <adwasser@ucsc.edu>
Website: <https://github.com/adwasser/masktools>

Required packages
-----------------
* numpy (tested on 1.11.0)
* astropy (tested on 1.1.1)
* matplotlib (tested on 1.5.0)

Recommended packages
--------------------
* astroquery (tested on 0.3.1)


masktools.superskims
--------------------

This package is for creating distributions of SKiMS (Stellar Kinematics with Multiple Slits) 
which optimally sample the integrated stellar light of a galaxy.

Authors: Nicola Pastorello and Asher Wasserman

This packages installs the `superskims` script, which creates dsim input files for a given galaxy.

> usage: superskims [-h] [-m MU_EFF] [-r RA] [-d DEC] [-q]
>                   name R_eff axial_ratio position_angle num_masks
> 
> This script generates dsim input files from the inputted galaxy parameters
> 
> positional arguments:
>   name                  Name of galaxy for output and NED lookup
>   R_eff                 Effective radius of galaxy in arcseconds
>   axial_ratio           Ratio of minor axis to major axis
>   position_angle        In degrees counter-clockwise from North
>   num_masks             Number of masks for the galaxy
> 
> optional arguments:
>   -h, --help            show this help message and exit
>   -m MU_EFF, --mu_eff MU_EFF
>                         Effective surface brightness in mag/arcsec^2, default
>                         is 22
>   -r RA, --ra RA        Right ascension, in degrees J2000. If name is findable
>                         by NED, will use that one instead.
>   -d DEC, --dec DEC     Declination, in degrees J2000. If name is findable by
>                         NED, will use that one instead.
>   -q, --quiet           If toggled, suppress console output.

For more fine-grained control, you can access the Mask and Galaxy classes through the `masktools.superskims` package while within a python environment, e.g.,::

    from masktools import superskims
    from astropy.coordinates import SkyCoord
    center = SkyCoord('12h35m37.9s +12d15m50s')
    galaxy = superskims.Galaxy(name='N4551', center=center, r_eff=16.6, 
                               axial_ratio=0.75, position_angle=70.5)
    galaxy.optimize(num_masks=4)
    for mask in galaxy.masks:
        output_file = mask.name + '_PA{:0.1f}_superskims.dsim'.format(mask.mask_pa)
        superskims.outputs.save_to_dsim(mask, galaxy.center, output_file)


