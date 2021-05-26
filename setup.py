from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="lammps-interface",
    author="Peter Boyd, Mohamad Moosavi, Matthew Witman",
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    version="0.2.2",
    license="MIT",
    url="https://github.com/peteboyd/lammps_interface",
    description="Automatic generation of LAMMPS input files for molecular dynamics simulations of MOFs",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    extras_require={
        'docs': [ 'sphinx>=2,<3', 'sphinx-rtd-theme>=0.4,<1' ],
        'tests': [ 'pytest' ]
    },
    entry_points={
        'console_scripts': [
            'lammps-interface = lammps_interface.cli:main'
        ]
    },
    include_package_data=True,
    packages=find_packages(),
)
