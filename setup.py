from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="lammps-interface",
    author="Peter Boyd, Mohamad Moosavi, Matthew Witman",
    version="0.1.2",
    license="MIT",
    url="https://github.com/peteboyd/lammps_interface",
    description="Automatic generation of LAMMPS input files for molecular dynamics simulations of MOFs",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    extras_require={
        'docs': [ 'sphinx>=2,<3', 'sphinx-rtd-theme>=0.4,<1' ]
    },
    include_package_data=True,
    packages=find_packages(),
    scripts=['lammps-interface'],
)
