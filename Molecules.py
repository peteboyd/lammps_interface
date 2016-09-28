import numpy as np

class Water(object):
    """Water parent class, containing functions applicable
    to all water models.

    """
    @property
    def H_coord(self):
        """Define the hydrogen coords based on 
        HOH angle for the specific force field.
        Default axis for distributing the 
        hydrogen atoms is the x-axis.

        """

        try:
            return self._H_coord
        except AttributeError:
            cos_theta = np.cos(np.deg2rad(self.HOH)/2.)
            sin_theta = np.sin(np.deg2rad(self.HOH)/2.)
            mat = np.array([[ cos_theta, sin_theta, 0.],
                            [-sin_theta, cos_theta, 0.],
                            [        0.,        0., 1.]])
            cos_theta = np.cos(np.deg2rad(-self.HOH)/2.)
            sin_theta = np.sin(np.deg2rad(-self.HOH)/2.)
            mat2 = np.array([[ cos_theta, sin_theta, 0.],
                             [-sin_theta, cos_theta, 0.],
                             [        0.,        0., 1.]])
            axis = np.array([1., 0., 0.])
            length = np.linalg.norm(np.dot(axis, mat))
            self._H_coord = self.ROH/length*np.array([np.dot(axis, mat), np.dot(axis, mat2)])
            return self._H_coord
    
    def compute_midpoint_vector(self, centre_vec, side1_vec, side2_vec):
        """ Define a vector oriented away from the centre_vec which
        is half-way between side1_vec and side2_vec.  Ideal for 
        TIP4P to define the dummy atom.

        """

        v = .5* (side1_vec - side2_vec) + (side2_vec - centre_vec) 
        v /= np.linalg.norm(v)
        return v
    
    def compute_orthogonal_vector(self, centre_vec, side1_vec, side2_vec):
        """ Define a vector oriented orthogonal to two others,
        centred by the 'centre_vec'.

        Useful for other water models with dummy atoms, as
        this can be used as a '4th' vector for the 'rotation_from_vectors'
        calculation (since 3 vectors defined by O, H, and H is not enough
        to orient properly).
        The real dummy atoms can then be applied once the proper
        rotation has been found.

        """

        v1 = side1_vec - centre_vec
        v2 = side2_vec - centre_vec

        v = np.cross(v1, v2)
        v /= np.linalg.norm(v)
        return v



class Molecule(object):
    def rotation_from_vectors(self, v1, v2):
        """Obtain rotation matrix from sets of vectors.
        the original set is v1 and the vectors to rotate
        to are v2.

        """

        # v2 = transformed, v1 = neutral
        ua = np.array([np.mean(v1.T[0]), np.mean(v1.T[1]), np.mean(v1.T[2])])
        ub = np.array([np.mean(v2.T[0]), np.mean(v2.T[1]), np.mean(v2.T[2])])
        Covar = np.dot((v2 - ub).T, (v1 - ua))

        try:
            u, s, v = np.linalg.svd(Covar)
            uv = np.dot(u,v[:3])
            d = np.identity(3)
            d[2,2] = np.linalg.det(uv) # ensures non-reflected solution
            M = np.dot(np.dot(u,d), v)
            return M
        except np.linalg.linalg.LinAlgError:
            return np.identity(3)

    def rotation_matrix(self, axis, angle):
        """
        returns a 3x3 rotation matrix based on the
        provided axis and angle
        """
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle / 2.)
        b, c, d = -axis*np.sin(angle / 2.)
    
        R = np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                  [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                  [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])
    
        return R

class TIP4P_Water(Molecule, Water):
    O_mass = 15.9994
    H_mass = 1.008
    X_mass = 0.00000001
    ROH = 0.9572
    HOH = 104.52
    Rdum = 0.125
    def __init__(self):
        """ Class that provides a template molecule for TIP4P Water.

        LAMMPS has some builtin features for this molecule which
        are taken advantage of here.
        Specifically, there is no need to explicilty describe
        a dummy atom for TIP4P as this is handeled internally.

        """

        self.O_coord = np.array([0., 0., 0.])
   
    @property
    def dummy(self):
        try:
            return self._dummy
        except AttributeError:
            try:
                # following assumes the H_pos are in the right spots
                v = self.compute_midpoint_vector(self.O_coord, self.H_coord[0], self.H_coord[1])
                self._dummy = self.Rdum*v + self.O_coord
            except AttributeError:
                self._dummy = np.array([self.Rdum, 0., 0.])

            return np.reshape(self._dummy, (1,3))
   
    def approximate_positions(self, O_pos=None, H_pos1=None, H_pos2=None):
        """Input a set of approximate positions for the oxygen
        and hydrogens of water, and determine the lowest RMSD
        that would give the idealized water model.

        """
        O = self.O_coord
        H1 = self.H_coord[0]
        H2 = self.H_coord[1]
        v1 = np.array([O, H1, H2])
        v2 = np.array([O_pos, H_pos1, H_pos2])
        R = self.rotation_from_vectors(v1, v2)
        self.O_coord = O_pos
        self._H_coord = np.dot(self._H_coord, R.T) + O_pos
        try:
            del self._dummy
        except AttributeError:
            # dummy hasn't been allocated yet
            pass

class TIP5P_Water(Molecule, Water):
    O_mass = 15.9994
    H_mass = 1.008
    X_mass = 0.00000001
    ROH = 0.9572
    HOH = 104.52
    Rdum = 0.70
    DOD = 109.47
    def __init__(self):
        """ Class that provides a template molecule for TIP5P Water.

        No built in features for TIP5P so the dummy atoms must
        be explicitly described.
        Geometric features are evaluated to ensure the proper
        configuration to support TIP5P.

        """

        self.O_coord = np.array([0., 0., 0.])
    
    @property
    def dummy(self):
        """ Given that the H_coords are determined from an angle with the x-axis
        in the xy plane, here we will also use the x-axis, only project
        the dummy atoms in the xz plane.
        This will produce, if all angles are 109.5 and distances the same, a perfect
        tetrahedron, however if the angles and distances are different, will produce
        a geometry with the highest possible symmetry.

        """
        try:
            return self._dummy
        except AttributeError:
            cos_theta = np.cos(np.deg2rad(self.DOD)/2.)
            sin_theta = np.sin(np.deg2rad(self.DOD)/2.)
            mat1 = np.array([[  cos_theta,         0.,  sin_theta],
                             [         0.,         1.,         0.],
                             [ -sin_theta,         0.,  cos_theta]])
            cos_theta = np.cos(np.deg2rad(-self.DOD)/2.)
            sin_theta = np.sin(np.deg2rad(-self.DOD)/2.)
            mat2 = np.array([[  cos_theta,         0.,  sin_theta],
                             [         0.,         1.,         0.],
                             [ -sin_theta,         0.,  cos_theta]])
            axis = np.array([-1., 0., 0.])
            self._dummy = self.Rdum*np.array([np.dot(axis, mat1), np.dot(axis, mat2)])
            return self._dummy
    
    def approximate_positions(self, O_pos=None, H_pos1=None, H_pos2=None):
        """Input a set of approximate positions for the oxygen
        and hydrogens of water, and determine the lowest RMSD
        that would give the idealized water model.

        """
        O = self.O_coord
        H1 = self.H_coord[0]
        H2 = self.H_coord[1]
        v1 = np.array([O, H1, H2])
        v2 = np.array([O_pos, H_pos1, H_pos2])
        R = self.rotation_from_vectors(v1, v2)
        self.O_coord = O_pos
        self._H_coord = np.dot(self._H_coord, R.T) + O_pos
        self.dummy
        self._dummy = np.dot(self._dummy, R.T) + O_pos
