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



def main():
    print __version__
    

if __name__ == "__main__": 
    main()

