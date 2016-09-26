import numpy as np

class Molecule(object):
    def rotation_from_vectors(self, v1, v2, point=None):
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
            R = np.identity(4)
            R[:3,:3] = M
            if point is not None:
                R[:3,:3] = point - np.dot(M, point)
            return R
        except np.linalg.linalg.LinAlgError:
            return np.identity(4)


class TIP4P_Water(Molecule):
    def __init__(self):
        """ Class that provides a template molecule for TIP4P Water.

        LAMMPS has some builtin features for this molecule which
        are taken advantage of here.
        Specifically, there is no need to explicilty describe
        a dummy atom for TIP4P as this is handeled internally.

        """

        self.O_coord = np.array([0., 0., 0.])
        self.ROH = 1.0
        self.HOH = 120.0
        self.Rdum = 1.0
    
    @property
    def H_coord(self):
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
   
    @property
    def dummy(self):
        try:
            return self._dummy
        except AttributeError:
            try:
                # following assumes the H_pos are in the right spots
                v = self.compute_midpoint_vector(self.O_coord, self.H_coord[0], self.H_coord[1])
                self._dummy = -self.Rdum*v + self.O_coord
            except AttributeError:
                self._dummy = np.array([self.Rdum, 0., 0.])

            return self._dummy
   
    def compute_midpoint_vector(self, centre_vec, side1_vec, side2_vec):
        v = .5* (side1_vec - side2_vec) + (centre_vec - side1_vec) 
        v /= np.linalg.norm(v)
        return v

    def approximate_positions(self, O_pos=None, H_pos1=None, H_pos2=None):
        """Input a set of approximate positions for the oxygen
        and hydrogens of water, and determine the lowest RMSD
        that would give the idealized water model.

        """
        v = np.array([self.O_coord, self.H_coord[0], self.H_coord[1], self.dummy])
        # get approx dummy location from current coordinates
        appd = -self.compute_midpoint_vector(O_pos, H_pos1, H_pos2)*self.Rdum + O_pos
        v2 = np.array([O_pos, H_pos1, H_pos2, appd]) - O_pos
        R = self.rotation_from_vectors(v, v2)
        self.O_coord = O_pos 
        #self._H_coord = np.dot(self._H_coord, R[:3,:3]) + O_pos
        self._H_coord += O_pos
        del self._dummy

class TIP5P_Water(Molecule):
    def __init__(self):
        """ Class that provides a template molecule for TIP5P Water.

        No built in features for TIP5P so the dummy atoms must
        be explicitly described.
        Geometric features are evaluated to ensure the proper
        configuration to support TIP5P.

        """

        self.O_coord = np.array([0., 0., 0.])
        self.ROH = 1.0
        self.HOH = 120.0
        self.Rdum = 1.0
    
    @property
    def H_coord(self):
        try:
            return self._H_coord
        except AttributeError:
            cos_theta = np.cos(np.deg2rad(self.HOH)/2.)
            sin_theta = np.sin(np.deg2rad(self.HOH)/2.)
            mat = np.array([[ cos_theta, sin_theta, 0.],
                            [-sin_theta, cos_theta, 0.],
                            [        0.,        0., 0.]])

            length = np.linalg.norm(np.dot(self.dummy, mat))
            self._H_coord = self.ROH/length*np.array([np.dot(self.dummy, mat), np.dot(self.dummy, -mat)])
            return self._H_coord
   
    @property
    def dummy(self):
        try:
            return self._dummy
        except AttributeError:
            try:
                # following assumes the H_pos are in the right spots
                v = .5* (self._H_coord[1] - self._H_coord[0]) - self.O_coord
                v = np.linalg.norm(v)
                self._dummy = self.Rdum*v
            except AttributeError:
                self._dummy = np.array([0., self.Rdum, 0.])

            return self._dummy
    
    def approximate_positions(self, O_pos, H_pos):
        """Input a set of approximate positions for the oxygen
        and hydrogens of water, and determine the lowest RMSD
        that would give the idealized water model.

        """
        v = np.array([self.O_coord, self.H_coord[0], self.H_coord[1]])
        v2 = np.array([O_pos, H_pos[0], H_pos[1]])
        
        R = self.rotation_from_vectors(v, v2, point=O_pos)
        self.O_coord = np.dot(R[:3,:3], self.O_coord.T).T
        self._H_coord = np.dot(R[:3,:3], self._H_coord.T).T
        del self._dummy
