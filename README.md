[![Build Status](https://travis-ci.org/peteboyd/lammps_interface.svg?branch=master)](https://travis-ci.org/peteboyd/lammps_interface)
# LAMMPS Interface
## Authors

-   Peter Boyd
-   Mohamad Moosavi
-   Matthew Witman

## Description
This program was designed for easy interface between the crystallographic
information file (.[cif]) and the Large-scale Atomic Molecular Massively
Parallel Simulator ([Lammps]).

## Installation
Simply install from [PyPI](https://pypi.org/project/lammps-interface/):
```
pip install lammps-interface
```

For development purposes, clone the repository and install it from source:
```
pip install -e .
```

Note: In both cases, this adds `lammps-interface` to your `PATH`.

## Usage

### Command line interface
To see the optional arguments type:
```
lammps-interface --help
```
To create [Lammps] simulation files for a given cif file type:
```
lammps-interface cif_file.cif
```
This will create [Lammps] simulation files with UFF parameters.

### Jupyter notebook
In order to implement module to your project check out Jupyter notebooks provided in this repository in `/notebooks` for usage examples.

## Licence
MIT licence (see LICENCE file)

## Citation
The publication associated with this code is found here:

Boyd, P. G., Moosavi, S. M., Witman, M. & Smit, B. Force-Field Prediction of Materials Properties in Metal-Organic Frameworks. J. Phys. Chem. Lett. 8, 357–363 (2017).

https://dx.doi.org/10.1021/acs.jpclett.6b02532

[Lammps]: http://lammps.sandia.gov/
[cif]: https://en.wikipedia.org/wiki/Crystallographic_Information_File
