from uff import UFF_DATA
from structure_data import Structure, Atom, Bond, Angle
import math

class UFF(object):
    
    def __init__(self):
        pass

    def bond_term(self, bond):
        """Harmonic assumed"""
        atom1, atom2 = bond.atoms
        fflabel1, fflabel2 = atom1.force_field_type, atom2.force_field_type
        r_1 = UFF_DATA[fflabel1][0]
        r_2 = UFF_DATA[fflabel2][0]
        chi_1 = UFF_DATA[fflabel1][8]
        chi_2 = UFF_DATA[fflabel2][8]

        rbo = -0.1332*(r_1 + r_2)*math.log(bond.order)
        ren = r_1*r_2*(((math.sqrt(chi_1) - math.sqrt(chi_2))**2))/(chi_1*r_1 + chi_2*r_2)
        r0 = (r_1 + r_2 + rbo - ren)
        K = 664.12*(UFF_DATA[fflabel1][5]*UFF_FULL[fflabel2][5])/(r0**3)
        bond.function = 'harmonic'
        bond.parameters = (K, r0)

    def angle_term(self, angle):
        """several cases exist where the type of atom in a particular environment is considered
        in both the parameters and the functional form of the term.


        A small cosine fourier expansion in (theta)
        E_0 = K_{IJK} * {sum^{m}_{n=0}}C_{n} * cos(n*theta)

        Linear, trigonal-planar, square-planar, and octahedral:
        two-term expansion of the above equation, n=0 as well as
        n=1, n=3, n=4, and n=4 for the above geometries.
        E_0 = K_{IJK}/n^2 * [1 - cos(n*theta)]

        in Lammps, the angle syle called 'fourier/simple' can be used
        to describe this functional form.
        
        general non-linear case:
        three-term Fourier expansion
        E_0 = K_{IJK} * [C_0 + C_1*cos(theta) + C_2*cos(2*theta)]
        
        in Lammps, the angle style called 'fourier' can be used to
        describe this functional form.

        Both 'fourier/simple' and 'fourier' are available from the 
        USER_MISC package in Lammps, so be sure to compile Lammps with
        this package.

        """

        # fourier/simple
        sf = ['linear', 'trigonal-planar', 'square-planar', 'octahedral']
        angle_type = self.uff_angle_type(angle)
        a_atom, b_atom, c_atom = angle.atoms
        ab_bond, bc_bond = angle.bonds

        auff, buff, cuff = a_atom.force_field_type, b_atom.force_field_type, c_atom.force_field_type

        theta0 = UFF_DATA[buff][1]
        cosT0 = math.cos(theta0*DEG2RAD)
        sinT0 = math.cos(theta0*DEG2RAD)

        c2 = 1.0 / (4.0*sinT0*sinT0)
        c1 = -4.0 * c2 * cosT0
        c0 = c2 * (2.0*cosT0*cosT0 + 1.0)

        za = UFF_DATA[auff][5]
        zc = UFF_DATA[cuff][5]
        
        r_ab = ab_bond.parameters[1]
        r_bc = bc_bond.parameters[1]
        r_ac = math.sqrt(r_ab*r_ab + r_bc*r_bc - 2.*r_ab*r_bc*cosT0)

        beta = 664.12/r_ab/r_bc
        ka = beta*(za*zc /(r_ac**5.))
        ka *= (3.*r_ab*r_bc*(1. - cosT0*cosT0) - r_ac*r_ac*cosT0)

        if angle_type in sf:
            angle.function = 'fourier/simple' 
            if angle_type == 'linear' or angle_type == 'trigonal-planar':
                kappa = ka
                c0 = -1.
                c1 = 1.

            if angle_type == 'trigonal-planar':
                kappa = ka/9.
                c0 = -1.
                c1 = 3.

            if angle_type == 'square-planar' or angle_type == 'octahedral':
                kappa = ka/16.
                c0 = -1.
                c1 = 4.

        # general-nonlinear
            angle.parameters = (kappa, c0, c1)
        else:
            angle.function = 'fourier'

            #TODO: a bunch of special cases which require molecular recognition here..
            # water, for example has it's own theta0 angle.

            theta0 = UFF_DATA[buff][1]

            c2 = 1. / (4.*math.sin(theta0)*math.sin(theta0))
            c1 = -4.*c2*math.cos(theta0)
            c0 = c2*(2.*math.cos(theta0)*math.cos(theta0) + 1)
            kappa = ka
            angle.parameters = (kappa, c0, c1, c2)

    def uff_angle_type(self, angle):
        l, c, r = angle.atoms
        """ determined by the central atom type """
        bent_types = ['O_3', 'S_3', 'O_R', 'N_2']
        if len(c.neighbours) == 2:
            if name[:3] in bent_types:
                return 'bent'
            return 'linear'

        trig_types = ['C_R', 'C_2', 'O_3_z', 'O_3']
        if len(c.neighbours) == 3:
            if name in trig_types:
                return 'trigonal-planar'
            return 'trigonal-pyramidal'

        # Need flag for Zn4O type MOFs where Zn is tetrahedral.
        # vs. Zn paddlewheel where Zn is square planar.
        sqpl_types = ['Fe6+2', 'Zn3+2', 'Cu3+1']
        if len(c.neighbours) == 4:
            if name in sqpl_types:
                return 'square-planar'
            return 'tetrahedral'

        if len(c.neighbours) == 6:
            return 'octahedral'


