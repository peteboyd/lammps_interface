#!/usr/bin/env python

class BondPotential(object):
    """
    Class to hold bond styles that are implemented in lammps
    Purpose is to store info that the user wants to use to overwrite standard UFF output of lammps_interface
    """
    
    class Class2(object):
        """Potential defined as

        E = K2*(r-R0)^2 + K3*(r-R0)^3 + K4*(r-R0)^4

        Input parameters: R0, K2, K3, K4
        """
        def __init__(self):
            self.name = "class2"
            self.R0 = 0.
            self.K2 = 0.
            self.K3 = 0.
            self.K4 = 0.

        def __str__(self):
            return ""

    class Fene(object):
        """Potential defined as

        E = -0.5*K*R0^2 * ln[1-(r/R0)^2] + 4*eps*[(sig/r)^12 - (sig/r)^6] + eps

        Input parameters: K, R0, eps, sig
        """

        def __init__(self):
            self.name = "fene" # NB: fene/omp and fene/kk exist
            self.K = 0.
            self.R0 = 0.
            self.eps = 0.
            self.sig = 0.

        def __str__(self):
            return ""

    class FeneExpand(object):
        """Potential defined as

        E = -0.5*K*R0^2 * ln[1-(r-del/R0)^2] + 
               4*eps*[(sig/r-del)^12 - (sig/r-del)^6] + eps

        Input parameters: K, R0, eps, sig, del
        """
        def __init__(self):
            self.name = "fene/expand" # NB: fene/expand/omp exists
            self.K = 0.
            self.R0 = 0.
            self.eps = 0.
            self.sig = 0.
            self.del = 0.

        def __str__(self):
            return ""
        
    class Harmonic(object):
        """Potential defined as

        E = K*(r - R0)^2

        Input parameters: K, R0
        """

        def __init__(self):
            self.name = "harmonic" # harmonic/kk and harmonic/omp exist
            self.K = 0.
            self.R0 = 0.

        def __str__(self):
            return ""

    class Morse(object):
        """Potential defined as

        E = D*[1 - e^(-alpha*(r-R0))]^2

        Input parameters: D, alpha, R0
        """
        def __init__(self):
            self.name = "morse" # morse/omp exists
            self.K = 0.
            self.R0 = 0.

        def __str__(self):
            return ""

    class NonLinear(object):
        """Potential defined as

        E = eps*(r-R0)^2 / [lamb^2 - (r-R0)^2] 

        Input parameters: eps, R0, lamb
        """
        def __init__(self):
            self.name = "nonlinear" # nonlinear/omp exists
            self.eps = 0.
            self.R0 = 0.
            self.lamb = 0.

        def __str__(self):
            return ""

    class Quartic(object):
        """Potential defined as

        E = K*(r-Rc)^2 * (r - Rc - B1) * (r - Rc - B2) + U0 + 
                  4*eps*[(sig/r)^12 - (sig/r)^6] + eps

        Input parameters: K, B1, B2, Rc, U0
        """
        def __init__(self):
            self.name = "quartic" # quartic/omp exists
            self.K = 0.
            self.B1 = 0.
            self.B2 = 0.
            self.Rc = 0.
            self.U0 = 0.

        def __str__(self):
            return ""

    class Table(object):
        """Potential read from file."""
        raise NotImplementedError ("Have not implemented the table funtion for lammps yet.")

    class HarmonicShift(object):
        """Potential defined as

        E = Umin/(R0 - Rc)^2 * [(r-R0)^2 - (Rc - R0)^2] 

        Input parameters: Umin, R0, Rc 
        """
        def __init__(self):
            self.name = "harmonic/shift" # harmonic/shift/omp exists
            self.Umin = 0.
            self.R0 = 0.
            self.Rc = 0.

        def __str__(self):
            return ""

    class HarmonicShiftCut(object):
        """Potential defined as

        E = Umin/(R0 - Rc)^2 * [(r-R0)^2 - (Rc - R0)^2] 

        Input parameters: Umin, R0, Rc 
        """
        def __init__(self):
            self.name = "harmonic/shift/cut" # harmonic/shift/cut/omp exists
            self.Umin = 0.
            self.R0 = 0.
            self.Rc = 0.

        def __str__(self):
            return ""

class AnglePotential(object) 
    """
    Class to hold angle styles that are implemented in lammps
    """
    
    class Charmm(object):
        """Potential defined as

        E = K*(theta - theta0)^2 + Kub*(r - Rub)^2

        Input parameters: K, theta0, Kub, Rub
        """
        def __init__(self):
            self.name = "charmm" # charmm/kk and charmm/omp exist
            self.K = 0.
            self.theta0 = 0.
            self.Kub = 0.
            self.Rub = 0.

        def __str__(self):
            return ""

    class Class2(object):
        raise NotImplementedError ("Will get on this..")

    class Cosine(object):
        """Potential defined as

        E = K*[1 - cos(theta)] 

        Input parameters: K
        """
        def __init__(self):
            self.name = "cosine" # cosine/omp exists 
            self.K = 0.

        def __str__(self):
            return ""

    class CosineDelta(object):
        """Potential defined as

        E = K*[1 - cos(theta-theta0)] 

        Input parameters: K, theta0
        """
        def __init__(self):
            self.name = "cosine/delta" # cosine/delta/omp exists 
            self.K = 0.
            self.theta0 = 0.

        def __str__(self):
            return ""

    class CosinePeriodic(object):
        """Potential defined as

        E = C*[1 - B*(-1)^n*cos(n*theta)] 

        Input parameters: C, B, n
        """
        def __init__(self):
            self.name = "cosine/periodic" # cosine/periodic/omp exists 
            self.C = 0.
            self.B = 0
            self.n = 0

        def __str__(self):
            return ""

    class CosineSquared(object):
        """Potential defined as

        E = K*[cos(theta) - cos(theta0)]^2

        Input parameters: K, theta0
        """
        def __init__(self):
            self.name = "cosine/squared" # cosine/squared/omp exists 
            self.K = 0.
            self.theta0 = 0.

        def __str__(self):
            return ""

    class Harmonic(object):
        """Potential defined as

        E = K*(theta - theta0)^2

        Input parameters: K, theta0
        """
        def __init__(self):
            self.name = "harmonic" # harmonic/kk and harmonic/omp exist 
            self.K = 0.
            self.theta0 = 0.

        def __str__(self):
            return ""

    class Table(object):
        raise NotImplementedError ("Have not implemented the table funtion for lammps yet.")

    class CosineShift(object):
        """Potential defined as

        E = -Umin/2 * [1 + cos(theta - theta0)]

        Input parameters: Umin, theta0
        """
        def __init__(self):
            self.name = "cosine/shift" # cosine/shift/omp exists 
            self.Umin = 0.
            self.theta0 = 0.

        def __str__(self):
            return ""
    
    class CosineShiftExp(object):
        """Potential defined as

        E = -Umin * [e^{-a*U(theta,theta0)} - 1] / [e^a - 1]

        where U(theta,theta0) = -0.5*(1 + cos(theta - theta0))

        Input parameters: Umin, theta0, a
        """
        def __init__(self):
            self.name = "cosine/shift/exp" # cosine/shift/exp/omp exists 
            self.Umin = 0.
            self.theta0 = 0.
            self.a = 0.

        def __str__(self):
            return ""

    class Dipole(object):
        """Potential defined as

        E = K*(cos(gamma) - cos(gamma0))^2 

        Input parameters: K, gamma0
        """
        def __init__(self):
            self.name = "dipole" # dipole/omp exists 
            self.K = 0.
            self.gamma0 = 0.

        def __str__(self):
            return ""
    
    class Fourier(object):
        """Potential defined as

        E = K*[C0 + C1*cos(theta) + C2*cos(2*theta)]

        Input parameters: K, C0, C1, C2
        """
        def __init__(self):
            self.name = "fourier" # fourier/omp exists 
            self.K = 0.
            self.C0 = 0.
            self.C1 = 0.
            self.C2 = 0.

        def __str__(self):
            return ""

    class FourierSimple(object):
        """Potential defined as

        E = K*[1 + c*cos(n*theta)]

        Input parameters: K, c, n
        """
        def __init__(self):
            self.name = "fourier/simple" # fourier/simple/omp exists 
            self.K = 0.
            self.c = 0.
            self.n = 0.

        def __str__(self):
            return ""

    class Quartic(object):
        """Potential defined as

        E = K2*(theta - theta0)^2 + K3*(theta - theta0)^3 + K4(theta - theta0)^4

        Input parameters: theta0, K2, K3, K4
        """
        def __init__(self):
            self.name = "quartic" # quartic/omp exists 
            self.theta0 = 0.
            self.K2 = 0.
            self.K3 = 0.
            self.K4 = 0.

        def __str__(self):
            return ""
    
    class Sdk(object):
        """Potential defined as

        E = K*(theta - theta0)^2 

        Input parameters: K, theta0 
        """
        def __init__(self):
            self.name = "sdk" 
            self.K = 0.
            self.theta0 = 0.

        def __str__(self):
            return ""

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
