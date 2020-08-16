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

 * make compatible with networkx 2.4 371d5d93110b349d51b5d7bfb7ef2bcc89752b7b

### Bug fixes

 * exit on error if atoms overlap in min-img distance check 29086e5c6ed4e8cc013624875d7029f4ec9ae9db

## v0.1.1

First release on PyPI
