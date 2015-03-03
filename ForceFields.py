class UFF(object):
    
    def __init__(self):
        pass

    def angle_term(self, left_atom, central_atom, right_atom):
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
        if self.uff_angle_type(left_atom, central_atom, right_atom) in sf:
            function = 'fourier/simple'

        # general-nonlinear
        else:
            function = 'fourier'


    def uff_angle_type(self, l, c, r):
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





