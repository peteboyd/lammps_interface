#!/usr/bin/env python
from datetime import date
import numpy as np
import math
import itertools
from atomic import MASS, ATOMIC_NUMBER
from ccdc import CCDC_BOND_ORDERS
DEG2RAD=np.pi/180.

class Bond_Pot(object):
    """
    Class to hold bond styles that are implemented in lammps
    Purpose is to store info that the user wants to use to overwrite standard UFF output of lammps_interface
    """
    

    def __init__(self):
        self.name = None
        self.style = 0
        self.choices = {'none': 0, 'class2': 1, 'fene': 2, 'fene/expand': 3, 'harmonic': 4, 'morse': 5, 'nonlinear': 6, 'quartic': 7, 'table': 8}
		self.num_params = {'none': 0, 'class2': 0, 'fene': 0, 'fene/expand': 0, 'harmonic': 2, 'morse': 0, 'nonlinear': 0, 'quartic': 0, 'table': 0}
        self.params = []


    def set_type(string):
        self.name = string
        try:
            self.style = self.choices[self.name]
        except KeyError:
            raise KeyError('No LAMMPS bond_style found matching: ' + str(self.name) + '\nPlease check your overwrite_potentials.txt file')

        if self.num_params[self.name] == 0:
            raise NameError('Sorry, bond_style <' + str(self.name) + '> not implemented yet.\nPlease contact authors in README.txt')


    def set_params(inlist):
        if self.name == None:
            raise NameError('Potential parameters assigned before potential type')
        else:
            if len(inlist) != self.num_params:
                raise NameError('Incorrect num of potential parameters given (=' + str(len(inlist)) + ').  Potential of type ' + str(self.name) | 'requires ' + str(self.num_params[self.name]))
            else:
               self.params = inlist



class Angle_Pot(object) 
    """
    Class to hold angle styles that are implemented in lammps
    Purpose is to store info that the user wants to use to overwrite standard UFF output of lammps_interface
    """

    def __init__(self):
        self.name = None
        self.style = 0
        self.choices = {'none': 0, 'charmm': 1, 'class2': 2, 'cosine': 3, 'cosine/delta': 4, 'cosine/periodic': 5, 'cosine/squared': 6, 'harmonic': 7, 'table': 8, 'fourier': 9, 'fourier/simple': 10}
        self.num_params = {'none': 0, 'charmm': 0, 'class2': 0, 'cosine': 0, 'cosine/delta': 0, 'cosine/periodic': 0, 'cosine/squared': 0, 'harmonic': 2, 'table': 0}
        self.params = []


    def set_type(string):
        self.name = string
        try:
            self.style = self.choices[self.name]
        except KeyError:
            raise KeyError('No LAMMPS angle_style found matching: ' + str(self.name) + '\nPlease check your overwrite_potentials.txt file')

        if self.num_params[self.name] == 0:
            raise NameError('Sorry, angle_style <' + str(self.name) + '> not implemented yet.\nPlease contact authors in README.txt')

    def set_params(inlist):
        if self.name == None:
            raise NameError('Potential parameters assigned before potential type')
        else:
            if len(inlist) != self.num_params:
                raise NameError('Incorrect num of potential parameters given (=' + str(len(inlist)) + ').  Potential of type ' + str(self.name) | 'requires ' + str(self.num_params[self.name]))
            else:
               self.params = inlist



class Dihedral_Pot(object) 
    """
    Class to hold dihedral styles that are implemented in lammps
    Purpose is to store info that the user wants to use to overwrite standard UFF output of lammps_interface
    """
                

    def __init__(self):
        self.name = None
        self.style = 0
        self.choices = {'none': 0, 'charmm': 1, 'class2': 2, 'harmonic': 3, 'helix': 4, 'multi/harmonic': 5, 'opls': 6}
        self.num_params = {'none': 0, 'charmm': 4, 'class2': 0, 'harmonic': 3, 'helix': 0, 'multi/harmonic': 5, 'opls': 4}
        self.params = []


    def set_type(string):
        self.name = string
        try:
            self.style = self.choices[self.name]
        except KeyError:
            raise KeyError('No LAMMPS dihedral_style found matching: ' + str(self.name) + '\nPlease check your overwrite_potentials.txt file for a typo in dihedral_style')

        if self.num_params[self.name] == 0:
            raise NameError('Sorry, dihedral_style <' + str(self.name) + '> not implemented yet.\nPlease contact authors in README.txt')


    def set_params(inlist):
        if self.name == None:
            raise NameError('Potential parameters assigned before potential type')
        else:
            if len(inlist) != self.num_params:
                raise NameError('Incorrect num of potential parameters given (=' + str(len(inlist)) + ').  Potential of type ' + str(self.name) | 'requires ' + str(self.num_params[self.name]))
            else:
               self.params = inlist




class Improper_Pot(object) 
    """
    Class to hold improper styles that are implemented in lammps
    Purpose is to store info that the user wants to use to overwrite standard UFF output of lammps_interface
    """

    def __init__(self):
        self.name = None
        self.style = 0
        self.choices = {'none': 0, 'class2': 1, 'cvff': 2, 'harmonic': 3, 'umbrella': 4}
        self.num_params = {'none': 0, 'class2': 0, 'cvff': 0, 'harmonic': 2, 'umbrella': 0}
        self.params = []


    def set_type(string):
        self.name = string
        try:
            self.style = self.choices[self.name]
        except KeyError:
            raise KeyError('No LAMMPS improper_style found matching: ' + str(self.name) + '\nPlease check your overwrite_potentials.txt file for a typo in dihedral_style')

        if self.num_params[self.name] == 0:
            raise NameError('Sorry, improper_style <' + str(self.name) + '> not implemented yet.\nPlease contact authors in README.txt')


    def set_params(inlist):
        if self.name == None:
            raise NameError('Potential parameters assigned before potential type')
        else:
            if len(inlist) != self.num_params:
                raise NameError('Incorrect num of potential parameters given (=' + str(len(inlist)) + ').  Potential of type ' + str(self.name) | 'requires ' + str(self.num_params[self.name]))
            else:
               self.params = inlist
			



###########################################
