## v0.2.2

### Bug fixes
 * fix issue with unicode on windows [1cde93ed57a3293455a40a1adc2b69881116b1dc](https://github.com/peteboyd/lammps_interface/commit/1cde93ed57a3293455a40a1adc2b69881116b1dc)

## v0.2.1

### Bug fixes
 * fix for accurate step reporting in min.csv file [11bd2e5c7b3a03a1b1f2b0387a4e76ad64b0f1ef](https://github.com/peteboyd/lammps_interface/commit/11bd2e5c7b3a03a1b1f2b0387a4e76ad64b0f1ef)
 * The dihedral force constant was underestimated for UFF and UFF4MOF #44 [e9349627e60481c98b3ac15da0e06183e06d2192](https://github.com/peteboyd/lammps_interface/commit/e9349627e60481c98b3ac15da0e06183e06d2192)
 * Fix UFF4MOF bug where it can't find the correct metal ff type #46 [2b53caab48ca3610cc05cd5cf444df6b2c79579c](https://github.com/peteboyd/lammps_interface/commit/2b53caab48ca3610cc05cd5cf444df6b2c79579c)
 * Adjust for undocumented factor of 2 in discovered in the LAMMPS source code leading to too large angle bending constants #47 [cc12a4233f9657a69e9c64a26deca76e5bdd0b42](https://github.com/peteboyd/lammps_interface/commit/cc12a4233f9657a69e9c64a26deca76e5bdd0b42)

## v0.2.0

### Bug fixes
 * Drop python 3.5 support as it is not compatible with networkx 2.4
 * Fix for dict key change during iteration [0fe4b6fb3d785a0bfe9456363b37b8493b7ec9ab](https://github.com/peteboyd/lammps_interface/commit/0fe4b6fb3d785a0bfe9456363b37b8493b7ec9ab)
 * Replace time.clock deprecated function [6f9b050b624ba94fdfacff0a34551f89cfd2f708](https://github.com/peteboyd/lammps_interface/commit/6f9b050b624ba94fdfacff0a34551f89cfd2f708)
 * Add ASE to requirements [54714b1600e9d83b1f354a0b6cafac983653f61f](https://github.com/peteboyd/lammps_interface/commit/54714b1600e9d83b1f354a0b6cafac983653f61f)
 * Start using ASE to read CIF files [b2a52c0cd1fc739d63fcfd17d65ae75cdf541f7d](https://github.com/peteboyd/lammps_interface/commit/b2a52c0cd1fc739d63fcfd17d65ae75cdf541f7d)

## v0.1.3

### Improvements

 * Optimize topology information calculation [#27](https://github.com/peteboyd/lammps_interface/pull/27)

### Bug fixes

 * fix legacy networkx commands [1065ff310172fad1f2cfb79dda7ea737de2cdab1](https://github.com/peteboyd/lammps_interface/commit/1065ff310172fad1f2cfb79dda7ea737de2cdab1)
 * fix install from source [#36](https://github.com/peteboyd/lammps_interface/pull/36)
 * fix Windows build [#32](https://github.com/peteboyd/lammps_interface/pull/32)

## v0.1.2

### Improvements

 * make compatible with networkx 2.4 [371d5d93110b349d51b5d7bfb7ef2bcc89752b7b](https://github.com/peteboyd/lammps_interface/commit/371d5d93110b349d51b5d7bfb7ef2bcc89752b7b)

### Bug fixes

 * exit on error if atoms overlap in min-img distance check [29086e5c6ed4e8cc013624875d7029f4ec9ae9db](https://github.com/peteboyd/lammps_interface/commit/371d5d93110b349d51b5d7bfb7ef2bcc89752b7b)

## v0.1.1

First release on PyPI
