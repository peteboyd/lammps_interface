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

        E = -0.5*K*R0^2 * ln[1-(r-delta/R0)^2] + 
               4*eps*[(sig/r-delta)^12 - (sig/r-delta)^6] + eps

        Input parameters: K, R0, eps, sig, delta
        """
        def __init__(self):
            self.name = "fene/expand" # NB: fene/expand/omp exists
            self.K = 0.
            self.R0 = 0.
            self.eps = 0.
            self.sig = 0.
            self.delta = 0.

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
            self.reduced = False

        def __str__(self):
            if self.reduced:
                return "%15.6f %15.6f"%(self.K, self.R0)
            return "%28s %15.6f %15.6f"%(self.name, self.K, self.R0)

    class Morse(object):
        """Potential defined as

        E = D*[1 - e^(-alpha*(r-R0))]^2

        Input parameters: D, alpha, R0
        """
        def __init__(self):
            self.name = "morse" # morse/omp exists
            self.D = 0.
            self.alpha = 0.
            self.R0 = 0.
            self.reduced = False

        def __str__(self):
            if self.reduced:
                return "%15.6f %15.6f %15.6f"%(self.D, self.alpha, self.R0)
            return "%28s %15.6f %15.6f %15.6f"%(self.name, self.D, self.alpha, self.R0)

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
        def __init__(self):
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

class AnglePotential(object): 
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
        def __init__(self):
            raise NotImplementedError ("Will get on this..")

    class Cosine(object):
        """Potential defined as

        E = K*[1 - cos(theta)] 

        Input parameters: K
        """
        def __init__(self):
            self.name = "cosine" # cosine/omp exists 
            self.K = 0.
            self.reduced = False

        def __str__(self):
            if self.reduced:
                return "%15.6f"%(self.K)
            return "%28s %15.6f"%(self.name,
                                  self.K)

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
            self.reduced = False
        def __str__(self):
            if self.reduced:
                return "%15.6f %15.6f"%(self.K,
                                        self.theta0)
            return "%28s %15.6f %15.6f"%(self.name,
                                         self.K,
                                         self.theta0)

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
        def __init__(self):
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
            self.reduced = False

        def __str__(self):
            if self.reduced:
                return "%15.6f %15.6f %15.6f %15.6f"%(self.K,
                                                      self.C0,
                                                      self.C1,
                                                      self.C2)
            return "%28s %15.6f %15.6f %15.6f %15.6f"%(self.name, self.K,
                                                       self.C0, self.C1, self.C2)

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
            self.reduced = False

        def __str__(self):
            if self.reduced:
                return "%15.6f %15.6f %15.6f"%(self.K,
                                               self.c,
                                               self.n)
            return "%28s %15.6f %15.6f %15.6f"%(self.name, self.K,
                                                self.c, self.n)

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

class DihedralPotential(object): 
    """
    Class to hold dihedral styles that are implemented in lammps
    """
                
    class Charmm(object):
        """Potential defined as

        E = K*[1 + cos(n*phi - d)]

        Input parameters: K, n, d, w (weighting for 1 - 4 non-bonded interactions) 
        """
        def __init__(self):
            self.name = "charmm" # charm/kk and charmm/omp exist 
            self.K = 0.
            self.n = 0
            self.d = 0
            self.w = 0. # should be kept at 0 for charmm force fields
            self.reduced = False
        def __str__(self):
            if self.reduced:
                return "%15.6f %15i %15i %15.6f"%(self.K, self.d, self.n, self.w)
            return "%28s %15.6f %15i %15i %15.6f"%(self.name, self.K, self.d, self.n, self.w)

    class Class2(object):
        def __init__(self):
            raise NotImplementedError ("Will get on this..")

    class Harmonic(object):
        """Potential defined as

        E = K*[1 + d*cos(n*phi)]

        Input parameters: K, d, n
        """
        def __init__(self):
            self.name = "harmonic" # harmonic/omp exists 
            self.K = 0.
            self.d = 0
            self.n = 0
            self.reduced = False

        def __str__(self):
            if self.reduced:
                return "%15.6f %15i %15i"%(self.K, self.d, self.n)
            return "%28s %15.6f %15i %15i"%(self.name, self.K, self.d, self.n)

    class Helix(object):
        """Potential defined as

        E = A*[1 - cos(theta)] + B*[1 + cos(3*theta)] + C*[1 + cos(theta + pi/4)] 

        Input parameters: A, B, C 
        """
        def __init__(self):
            self.name = "helix" # helix/omp exists 
            self.A = 0.
            self.B = 0.
            self.C = 0.

        def __str__(self):
            return ""

    class MultiHarmonic(object):
        """Potential defined as

        E = sum_n=1,5{ An*cos^(n-1)(theta)}

        Input parameters: A1, A2, A3, A4, A5
        """
        def __init__(self):
            self.name = "multi/harmonic" # multi/harmonic/omp exists 
            self.A1 = 0.
            self.A2 = 0.
            self.A3 = 0.
            self.A4 = 0.
            self.A5 = 0.

        def __str__(self):
            return ""
    
    class Opls(object):
        """Potential defined as

        E = 0.5*K1*[1 + cos(theta)] + 0.5*K2*[1 - cos(2*theta)] + 
            0.5*K3*[1 + cos(3*theta)] + 0.5*K4*[1 - cos(4*theta)]

        Input parameters: K1, K2, K3, K4
        """
        def __init__(self):
            self.name = "opls" # opls/kk and opls/omp exist
            self.K1 = 0.
            self.K2 = 0.
            self.K3 = 0.
            self.K4 = 0.

        def __str__(self):
            return ""

    class CosineShiftExp(object):
        """Potential defined as

        E = -Umin*[e^{-a*U(theta,theta0)} - 1] / (e^a -1) 

        where U(theta, theta0) = -0.5*(1 + cos(theta-theta0))

        Input parameters: Umin, theta0, a 
        """
        def __init__(self):
            self.name = "cosine/shift/exp" # cosine/shift/exp/omp exists 
            self.Umin = 0.
            self.theta0 = 0.
            self.a = 0.

        def __str__(self):
            return ""

    class Fourier(object):
        """Potential defined as

        E = sum_i=1,m { Ki*[1.0 + cos(ni*theta - di)] } 

        Input parameters: m, Ki, ni, di 

        NB m is an integer dictating how many terms to sum, there should be 3*m + 1 
        total parameters.

        """
        def __init__(self):
            self.name = "fourier" # fourier/omp exists
            self.m = 0
            self.Ki = []
            self.ni = []
            self.di = []

        def __str__(self):
            return ""

    class nHarmonic(object):
        """Potential defined as

        E = sum_i=1,n { Ai*cos^{i-1}(theta)

        Input parameters: n, Ai

        NB n is an integer dictating how many terms to sum, there should be n + 1 
        total parameters.

        """
        def __init__(self):
            self.name = "nharmonic" # nharmonic/omp exists
            self.n = 0
            self.Ai = []

        def __str__(self):
            return ""

    class Quadratic(object):
        """Potential defined as

        E = K*(theta - theta0)^2 

        Input parameters: K, phi0 

        """
        def __init__(self):
            self.name = "quadratic" # quadratic/omp exists
            self.K = 0.
            self.phi0 = 0.

        def __str__(self):
            return ""
    
    class Table(object):
        """Potential read from file."""
        def __init__(self):
            raise NotImplementedError ("Have not implemented the table funtion for lammps yet.")


class ImproperPotential(object): 
    """
    Class to hold improper styles that are implemented in lammps
    """
    
    class Class2(object):
        def __init__(self):
            raise NotImplementedError ("Will get on this..")

    class Cvff(object):
        """Potential defined as

        E = K*[1 + d*cos(n*theta)]

        Input parameters: K, d, n

        """
        def __init__(self):
            self.name = "cvff" # cvff/omp exists
            self.K = 0.
            self.d = 0
            self.n = 0

        def __str__(self):
            return ""
    
    class Harmonic(object):
        """Potential defined as

        E = K*(chi - chi0)^2 

        Input parameters: K, chi0 

        """
        def __init__(self):
            self.name = "harmonic" # harmonic/kk and harmonic/omp exist
            self.K = 0.
            self.chi0 = 0.

        def __str__(self):
            return ""
    
    class Umbrella(object):
        """Potential defined as

        E = 0.5*K*[1 + cos(omega0)/sin(omega0)]^2 * [cos(omega) - cos(omega0)]   if omega0 .ne. 0 (deg)
        E = K*[1 - cos(omega)]  if omega0 = 0 (deg)

        Input parameters: K, omega0 

        """
        def __init__(self):
            self.name = "umbrella" # umbrella/omp exists
            self.K = 0.
            self.omega0 = 0.
            self.reduced = True
        def __str__(self):
            if self.reduced:
                return "%15.6f %15.6f "%(self.K,
                                         self.omega0)
            return "%28s %15.6f %15.6f"%(self.name,
                                         self.K,
                                         self.omega0)
    
    class Cossq(object):
        """Potential defined as

        E = 0.5*K*cos^2(chi - chi0) 

        Input parameters: K, chi0 

        """
        def __init__(self):
            self.name = "cossq" # cossq/omp exists
            self.K = 0.
            self.chi0 = 0.

        def __str__(self):
            return ""
    
    class Fourier(object):
        """Potential defined as

        E = K*[C0 + C1*cos(omega) + C2*cos(2*omega)] 

        Input parameters: K, C0, C1, C2, a

        the parameter a allows all three angles to be taken into account in an 
        improper dihedral. It is not clear in the lammps manual what to set this 
        to to turn it off/on, but the usual assumptions are 0/1.
        """
        def __init__(self):
            self.name = "fourier" # fourier/omp exists
            self.K = 0.
            self.C0 = 0.
            self.C1 = 0.
            self.C2 = 0.
            self.a = 0
            self.reduced = False

        def __str__(self):
            if self.reduced:
                return "%15.6f %15.6f %15.6f %15i"%(self.C0,
                                                    self.C1,
                                                    self.C2,
                                                    self.a)
            return "%28s %15.6f %15.6f %15.6f %15i"%(self.name,
                                                     self.C0,
                                                     self.C1,
                                                     self.C2,
                                                     self.a)
    
    class Ring(object):
        """Potential defined as

        E = 1/6*K*(delta_ijl + delta_ijk + delta_kjl)^6 

        where delta_ijl = cos(theta_ijl) - cos(theta0)

        Input parameters: K, theta0

        """
        def __init__(self):
            self.name = "ring" # ring/omp exists
            self.K = 0.
            self.theta0 = 0.

        def __str__(self):
            return ""

class PairPotential(object): 
    """
    Class to hold Pair styles that are implemented in lammps
    NB: list here is HUGE, update as needed..

    """
    
    class LjCutCoulLong(object):
        """Potential defined as

        E = 4*eps*[(sig/r)^12 - (sig/r)^6] r < rc

        and coulombic terms dealt with a kspace solver
        """
        def __init__(self):
            self.name = "lj/cut/coul/long" 
            self.eps = 0.
            self.sig = 0.
            self.reduced = False

        def __str__(self):
            if self.reduced:
                return "%15.6f %15.6f"%(self.eps,
                                        self.sig)
            return "%28s %15.6f %15.6f"%(self.name,
                                         self.eps,
                                         self.sig)

    class BuckCoulLong(object):
        """Potential defined as

        E = A*exp{-r/rho} - C/r^{6}


        """
        def __init__(self):
            self.name = "buck/coul/long"
            self.A = 0.0
            self.rho = 0.0
            self.C = 0.0
            self.reduced = False

        def __str__(self):
            if self.reduced:
                return "%15.6f %15.6f %15.6f"%(self.A,
                                               self.rho,
                                               self.C)
            return "%28s %15.6f %15.6f %15.6f"%(self.name,
                                                self.A,
                                                self.rho,
                                                self.C)


    class HbondDreidingMorse(object):
        """Potential defined as

        E = D0*[exp{-2*alpha*(r-R0)} - 2*exp{-alpha*(r-R0)}]*cos^n(theta)

        """
        def __init__(self):
            self.name = "hbond/dreiding/morse"
            self.htype = 0
            self.donor = 'i'
            self.D0 = 0.0
            self.alpha = 0.0
            self.R0 = 0.0
            self.n = 0.0
            self.Rin = 0.0
            self.Rout = 0.0
            self.a_cut = 0.0
            self.reduced = False

        def __str__(self):
            if self.reduced:
                return "%i %s %15.6f %15.6f %15.6f %15.6f %15.6f %15.6f %15.6f"%(
                                               self.htype,
                                               self.donor,
                                               self.D0,
                                               self.alpha,
                                               self.R0,
                                               self.n,
                                               self.Rin,
                                               self.Rout,
                                               self.a_cut)
            return "%28s %i %s %15.6f %15.6f %15.6f %15.6f %15.6f %15.6f %15.6f"%(
                                               self.name,
                                               self.htype,
                                               self.donor,
                                               self.D0,
                                               self.alpha,
                                               self.R0,
                                               self.n,
                                               self.Rin,
                                               self.Rout,
                                               self.a_cut)
