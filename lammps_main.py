#!/usr/bin/env python
"""
main.py

the program starts here.

"""

# Turn on keyword expansion to get revision numbers in version strings
# in .hg/hgrc put
# [extensions]
# keyword =
#
# [keyword]
# lammps_main.py =
#
# [keywordmaps]
# Revision = {rev}


try:
    __version_info__ = (0, 0, 0, int("$Revision$".strip("$Revision: ")))
except ValueError:
    __version_info__ = (0, 0, 0, 0)
__version__ = "%i.%i.%i.%i"%__version_info__

import sys
from structure_data import CIF, Structure, Atom
from uff import UFF_DATA
from atomic import MASS, ATOMIC_NUMBER

def main():
    print __version__

    cif = CIF(name="test")
    # NB can add the filename as the second argument of the class instance,
    # or from a separate function

    # set as the first argument for testing
    cif.read(sys.argv[1])

    struct = Structure(name='test')
    struct.from_CIF(cif)

if __name__ == "__main__": 
    main()

