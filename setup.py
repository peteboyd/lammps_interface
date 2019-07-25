from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="lammps_interface",
    version="0.1",
    description="Automatic generation of LAMMPS input files for molecular dynamics simulations of MOFs",
    install_requires=requirements,
    include_package_data=True,
    packages=find_packages(),
    scripts=['lammps-interface'],
)
