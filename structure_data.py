#!/usr/bin/env python
from datetime import date
import numpy as np
import math
import itertools
from atomic import MASS, ATOMIC_NUMBER
from ccdc import CCDC_BOND_ORDERS
DEG2RAD=np.pi/180.
class Structure(object):

    def __init__(self, name):
        self.name = name
        self.cell = Cell()
        self.atoms = []
        self.guests = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []


    def from_CIF(self, cifobj):
        """Reads the structure data from the CIF
        - currently does not read the symmetry of the cell
        - does not unpack the assymetric unit (assumes P1)
        - assumes that the appropriate keys are in the cifobj (no error checking)
        """

        data = cifobj._data
        # obtain atoms and cell
        cellparams = [float(i) for i in [data['_cell_length_a'], 
                                         data['_cell_length_b'], 
                                         data['_cell_length_c'],
                                         data['_cell_angle_alpha'], 
                                         data['_cell_angle_beta'], 
                                         data['_cell_angle_gamma']]]
        self.cell.set_params(cellparams)
        print(data.keys()) 
        x, y, z = data['_atom_site_fract_x'], data['_atom_site_fract_y'], data['_atom_site_fract_z']
        
        # Charge assignment may have to be a bit more inclusive than just setting _atom_site_charge
        # in the .cif file.. will have to think of a user-friendly way to introduce charges..
        if '_atom_site_charge' in data:
            charges = [float(j.strip()) for j in data['_atom_site_charge']]
        else:

            charges = [0. for i in range(0, len(x))]

        label, element, ff_param = (data['_atom_site_label'], 
                                    data['_atom_site_type_symbol'], 
                                    data['_atom_site_description'])
        
        index = 0
        for l,e,ff,fx,fy,fz,c in zip(label,element,ff_param,x,y,z,charges):
            fcoord = np.array([float(j) for j in (fx, fy, fz)])
            atom = Atom(element=e.strip(), coordinates = np.dot(fcoord, self.cell.cell))
            atom.force_field_type = ff.strip()
            atom.ciflabel = l.strip()
            atom.charge = c 
            self.atoms.append(atom)
        # obtain bonds
        a, b, type = (data['_geom_bond_atom_site_label_1'], 
                      data['_geom_bond_atom_site_label_2'], 
                      data['_ccdc_geom_bond_type'])

        for label1, label2, t in zip(a,b,type):
            atm1 = self.get_atom_from_label(label1.strip())
            atm2 = self.get_atom_from_label(label2.strip())
            #TODO(check if atm2 crosses a periodic boundary to bond with atm1)
            #.cif file double counts bonds for some reason.. maybe symmetry related
            if (atm2.index not in atm1.neighbours) and (atm1.index not in atm2.neighbours):
                atm1.neighbours.append(atm2.index)
                atm2.neighbours.append(atm1.index)
                bond = Bond(atm1=atm1, atm2=atm2, 
                            order=CCDC_BOND_ORDERS[t.strip()])

                self.bonds.append(bond)

        # unwrap symmetry elements if they exist

    def get_atom_from_label(self, label):
        for atom in self.atoms:
            if atom.ciflabel == label:
                return atom
    

    def compute_angles(self):
        for atom in self.atoms:
            if len(atom.neighbours) < 2:
                continue
            angles = itertools.combinations(atom.neighbours, 2)
            for (lid, rid) in angles:
                left_atom = self.atoms[lid]
                right_atom = self.atoms[rid]
                abbond = self.get_bond(left_atom, atom)
                bcbond = self.get_bond(atom, right_atom)
                angle = Angle(abbond, bcbond)
                self.angles.append(angle)

    def get_bond(self, atom1, atom2):
        for bond in self.bonds:
            if set((atom1, atom2)) ==  set(bond.atoms):
                return bond
        return None

    def compute_dihedrals(self):
        done_bs=[]
        for atom_b in self.atoms:
            ib = atom_b.index
            ib_type = atom_b.ff_type_index
            angles = itertools.permutations(atom_b.neighbours, 2)
            done_bs.append(atom_b)
            for ia, ic in angles:
                atom_a = self.atoms[ia]
                atom_c = self.atoms[ic]
                ia_type = atom_a.ff_type_index
                ic_type = atom_c.ff_type_index
                # ignore cases where the c atom has already
                # been used as a b atom. Otherwise this will double-count
                # dihedral angles in the reverse order..
                if atom_c in done_bs:
                    continue
                c_neighbours = [i for i in atom_c.neighbours if ib != i and
                                ia != i]
                for id in c_neighbours:
                    atom_d = self.atoms[id]
                    id_type = atom_d.ff_type_index
                    
                    angle1 = self.get_angle(atom_a, atom_b, atom_c)
                    angle2 = self.get_angle(atom_b, atom_c, atom_d)
                    if angle1 is None:
                        angle1 = self.get_angle(atom_c, atom_b, atom_a)
                    if angle2 is None:
                        angle2 = self.get_angle(atom_d, atom_c, atom_b)
                    dihedral = Dihedral(angle1, angle2)
                    self.dihedrals.append(dihedral)
    
    def get_angle(self, atom_a, atom_b, atom_c):
        for angle in self.angles:
            if (atom_a, atom_b, atom_c) ==  angle.atoms:
                return angle
        return None

    def compute_improper_dihedrals(self):
        count = 0
        improper_type = {}

        for atom_b in self.atoms:
            if not atom_b.atomic_number in (6, 7, 8, 15, 33, 51, 83):
                continue
            if len(atom_b.neighbours) != 3:
                continue
            ib = atom_b.index
            # three improper torsion angles about each atom
            for idx,(ia,ic,id) in enumerate(itertools.permutations(atom_b.neighbours)):
                if idx == 3:
                    break
                atom_a, atom_c, atom_d = self.atoms[ia], self.atoms[ic], self.atoms[id]

                abbond = self.get_bond(atom_a, atom_b)
                bcbond = self.get_bond(atom_b, atom_c)
                bdbond = self.get_bond(atom_b, atom_d)
                improper = ImproperDihedral(abbond, bcbond, bdbond)
                self.impropers.append(improper)

class Bond(object):
    __ID = 0

    def __init__(self, atm1=None, atm2=None, order=1):
        self.index = self.__ID
        self.order = order
        self._atoms = (atm1, atm2)
        self.length = 0.
        self.ff_type_index = 0
        self.midpoint = np.array([0., 0., 0.])
        self.function = None
        self.parameters = None
        Bond.__ID += 1

    def compute_length(self, coord1, coord2):
        return np.linalg.norm(np.array(coord2) - np.array(coord1))

    def set_atoms(self, atm1, atm2):
        self._atoms = (atm1, atm2)

    def get_atoms(self):
        return self._atoms

    atoms = property(get_atoms, set_atoms)
    
    @property
    def indices(self):
        if not None in self.atoms:
            return (self.atoms[0].index, self.atoms[1].index)
        return (None, None)

    @property
    def elements(self):
        if not None in self.atoms:
            return (self.atoms[0].element, self.atoms[1].element)
        return (None, None)

class Angle(object):
    __ID = 0
    def __init__(self, abbond=None, bcbond=None):
        """Class to contain angles. Atoms are labelled according to the angle:
        a   c
         \ /
          b 
        """
        # atoms are obtained from the bonds.
        self._atoms = (None, None, None)
        if abbond is not None and bcbond is not None:
            self.bonds = (abbond, bcbond)
        else:
            self._bonds = (abbond, bcbond)
        self.ff_type_index = 0
        self.function = None
        self._angle = 0.
        self.index = self.__ID
        Angle.__ID += 1

    def set_bonds(self, bonds):
        """order is assumed (ab_bond, bc_bond)"""
        self._bonds = bonds
        atm1, atm2 = bonds[0].atoms
        atm3, atm4 = bonds[1].atoms

        self._atoms = list(self._atoms)
        if atm1 in (atm3, atm4):
            self._atoms[0] = atm2
            self._atoms[1] = atm1
            if atm1 == atm3:
                self._atoms[2] = atm4
            else:
                self._atoms[2] = atm3

        elif atm2 in (atm3, atm4):
            self._atoms[0] = atm1
            self._atoms[1] = atm2
            if atm2 == atm3:
                self._atoms[2] = atm4
            else:
                self._atoms[2] = atm3
        self._atoms = tuple(self._atoms)

    def get_bonds(self):
        return self._bonds

    bonds = property(get_bonds, set_bonds)

    @property
    def ab_bond(self):
        return self._bonds[0]
   
    @property
    def bc_bond(self):
        return self._bonds[1]
   
    @property
    def atoms(self):
        return self._atoms

    @property
    def a_atom(self):
        return self._atoms[0]

    @property
    def b_atom(self):
        return self._atoms[1]

    @property
    def c_atom(self):
        return self._atoms[2]

class Dihedral(object):
    """Class to store dihedral angles
    a
     \ 
      b -- c
            \ 
             d

    """
    __ID = 0
    def __init__(self, angle1=None, angle2=None):
        self._atoms = (None, None, None, None)
        self._bonds = (None, None, None)
        # angles of the form: angle_abc, angle_bcd
        self._angles = (angle1, angle2)
        if not None in (angle1, angle2):
            self.angles = (angle1, angle2)
        self.ff_type_index = 0
        self.index = self.__ID
        self.function = None
        Dihedral.__ID += 1

    def set_angles(self, angles):
        angle1, angle2 = angles
        bonds1 = angle1.bonds
        bonds2 = angle2.bonds

        if angle1.bc_bond != angle2.ab_bond:
            if angle1.bc_bond == angle2.bc_bond:
                angle2.bonds = tuple(reversed(bonds2))
            elif angle1.ab_bond == angle2.ab_bond:
                angle1.bonds = tuple(reversed(bonds1))
            elif angle1.ab_bond == angle2.bc_bond:
                angle1.bonds = tuple(reversed(bonds1))
                angle2.bonds = tuple(reversed(bonds2))
        self._angles = (angle1, angle2)

        assert angle1.bc_bond == angle2.ab_bond

        assert angle1.b_atom == angle2.a_atom

        assert angle1.c_atom == angle2.b_atom

        self._atoms = tuple([angle1.a_atom, angle1.b_atom,
                             angle2.b_atom, angle2.c_atom])
        self._bonds = tuple([angle1.ab_bond, angle1.bc_bond, angle2.bc_bond])

    def get_angles(self):
        return self._angles

    angles = property(get_angles, set_angles)

    @property
    def a_atom(self):
        return self._atoms[0]

    @property
    def b_atom(self):
        return self._atoms[1]

    @property
    def c_atom(self):
        return self._atoms[2]
    
    @property
    def d_atom(self):
        return self._atoms[3]
    
    @property
    def atoms(self):
        return self._atoms

    @property
    def ab_bond(self):
        return self._bonds[0]

    @property
    def bc_bond(self):
        return self._bonds[1]

    @property
    def cd_bond(self):
        return self._bonds[2]

    @property
    def bonds(self):
        return self._bonds

    @property
    def abc_angle(self):
        return self._angles[0]

    @property
    def bcd_angle(self):
        return self._angles[1]

class ImproperDihedral(object):
    """Class to store improper dihedral angles

    a
     \ 
      b -- c
      |
      d

    """
    __ID = 0
    def __init__(self, bond1=None, bond2=None, bond3=None):
        self._atoms = (None, None, None, None)
        self._bonds = (bond1, bond2, bond3)
        if not None in (bond1, bond2, bond3):
            self.bonds = (bond1, bond2, bond3)
        self.ff_type_index = 0
        self.function = None
        self.index = self.__ID
        ImproperDihedral.__ID += 1
    
    def set_bonds(self, bonds):
        self._angles = bonds
        bond1, bond2, bond3 = bonds
        self._atoms = [None, None, None, None]
        for a1 in bond1.atoms:
            for a2 in bond2.atoms:
                for a3 in bond3.atoms:
                    if a1 == a2 == a3:
                        self._atoms[1] = a1

        ab1, ab2 = bond1.atoms
        ab3, ab4 = bond2.atoms
        ab5, ab6 = bond3.atoms

        if ab1 == self._atoms[1]:
            self._atoms[0] = ab2
        else:
            self._atoms[0] = ab1

        if ab3 == self._atoms[1]:
            self._atoms[2] = ab4
        else:
            self._atoms[2] = ab3

        if ab5 == self._atoms[1]:
            self._atoms[3] = ab6
        else:
            self._atoms[3] = ab5

    def get_bonds(self):
        return self._bonds

    bonds = property(get_bonds, set_bonds)

    @property
    def a_atom(self):
        return self._atoms[0]

    @property
    def b_atom(self):
        return self._atoms[1]

    @property
    def c_atom(self):
        return self._atoms[2]
    
    @property
    def d_atom(self):
        return self._atoms[3]
    
    @property
    def atoms(self):
        return self._atoms

    @property
    def ab_bond(self):
        return self._bonds[0]

    @property
    def bc_bond(self):
        return self._bonds[1]

    @property
    def bd_bond(self):
        return self._bonds[2]

class Atom(object):
    __ID = 0
    def __init__(self, element="X", coordinates=np.zeros(3)):
        self.element = element
        self.index = self.__ID
        self.neighbours = []
        self.ciflabel = None
        self.force_field_type = None
        self.coordinates = coordinates
        self.charge = 0.
        self.ff_type_index = 0 # keeps track of the unique integer value assigned to the force field type
        Atom.__ID += 1

    def scaled_pos(self, inv_cell):
        return np.dot(self.coordinates[:3], inv_cell)

    def in_cell_scaled(self, inv_cell):
        return np.array([i%1 for i in self.scaled_pos(inv_cell)])

    def in_cell(self, cell, inv_cell):
        return np.dot(self.in_cell_scaled(inv_cell), cell)

    @property
    def mass(self):
        return MASS[self.element]

    @property
    def atomic_number(self):
        return ATOMIC_NUMBER.index(self.element)

class Cell(object):
    def __init__(self):
        self._cell = np.identity(3, dtype=np.float64)
        # cell parameters (a, b, c, alpha, beta, gamma)
        self._params = (1., 1., 1., 90., 90., 90.)
        self._inverse = None

    @property
    def volume(self):
        """Calculate cell volume a.bxc."""
        b_cross_c = cross(self.cell[1], self.cell[2])
        return dot(self.cell[0], b_cross_c)

    def get_cell(self):
        """Get the 3x3 vector cell representation."""
        return self._cell

    def set_cell(self, value):
        """Set cell and params from the cell representation."""
        # Class internally expects an array
        self._cell = np.array(value).reshape((3,3))
        self.__mkparam()
        self.__mklammps()
        self._inverse = np.linalg.inv(self.cell.T)

    # Property so that params are updated when cell is set
    cell = property(get_cell, set_cell)

    def get_params(self):
        """Get the six parameter cell representation as a tuple."""
        return tuple(self._params)

    def set_params(self, value):
        """Set cell and params from the cell parameters."""
        self._params = value
        self.__mkcell()
        self.__mklammps()
        self._inverse = np.linalg.inv(self.cell.T)

    params = property(get_params, set_params)

    def minimum_supercell(self, cutoff):
        """Calculate the smallest supercell with a half-cell width cutoff."""
        a_cross_b = cross(self.cell[0], self.cell[1])
        b_cross_c = cross(self.cell[1], self.cell[2])
        c_cross_a = cross(self.cell[2], self.cell[0])

        volume = dot(self.cell[0], b_cross_c)

        widths = [volume / np.linalg.norm(b_cross_c),
                  volume / np.linalg.norm(c_cross_a),
                  volume / np.linalg.norm(a_cross_b)]

        return tuple(int(ceil(2*cutoff/x)) for x in widths)

    @property
    def minimum_width(self):
        """The shortest perpendicular distance within the cell."""
        a_cross_b = cross(self.cell[0], self.cell[1])
        b_cross_c = cross(self.cell[1], self.cell[2])
        c_cross_a = cross(self.cell[2], self.cell[0])

        volume = dot(self.cell[0], b_cross_c)

        return volume / min(np.linalg.norm(b_cross_c), np.linalg.norm(c_cross_a), np.linalg.norm(a_cross_b))

    @property
    def inverse(self):
        """Inverted cell matrix for converting to fractional coordinates."""
        try:
            if self._inverse is None:
                self._inverse = np.linalg.inv(self.cell.T)
        except AttributeError:
            self._inverse = np.linalg.inv(self.cell.T)
        return self._inverse

    @property
    def crystal_system(self):
        """Return the IUCr designation for the crystal system."""
        #FIXME(tdaff): must be aligned with x to work
        if self.alpha == self.beta == self.gamma == 90:
            if self.a == self.b == self.c:
                return 'cubic'
            elif self.a == self.b or self.a == self.c or self.b == self.c:
                return 'tetragonal'
            else:
                return 'orthorhombic'
        elif self.alpha == self.beta == 90:
            if self.a == self.b and self.gamma == 120:
                return 'hexagonal'
            else:
                return 'monoclinic'
        elif self.alpha == self.gamma == 90:
            if self.a == self.c and self.beta == 120:
                return 'hexagonal'
            else:
                return 'monoclinic'
        elif self.beta == self.gamma == 90:
            if self.b == self.c and self.alpha == 120:
                return 'hexagonal'
            else:
                return 'monoclinic'
        elif self.a == self.b == self.c and self.alpha == self.beta == self.gamma:
            return 'trigonal'
        else:
            return 'triclinic'

    def __mkcell(self):
        """Update the cell representation to match the parameters."""
        a_mag, b_mag, c_mag = self.params[:3]
        alpha, beta, gamma = [x * DEG2RAD for x in self.params[3:]]
        a_vec = np.array([a_mag, 0.0, 0.0])
        b_vec = np.array([b_mag * np.cos(gamma), b_mag * np.sin(gamma), 0.0])
        c_x = c_mag * np.cos(beta)
        c_y = c_mag * (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
        c_vec = np.array([c_x, c_y, (c_mag**2 - c_x**2 - c_y**2)**0.5])
        self._cell = np.array([a_vec, b_vec, c_vec])

    def __mkparam(self):
        """Update the parameters to match the cell."""
        cell_a = sqrt(sum(x**2 for x in self.cell[0]))
        cell_b = sqrt(sum(x**2 for x in self.cell[1]))
        cell_c = sqrt(sum(x**2 for x in self.cell[2]))
        alpha = np.arccos(sum(self.cell[1, :] * self.cell[2, :]) /
                       (cell_b * cell_c)) * 180 / pi
        beta = np.arccos(sum(self.cell[0, :] * self.cell[2, :]) /
                      (cell_a * cell_c)) * 180 / pi
        gamma = np.arccos(sum(self.cell[0, :] * self.cell[1, :]) /
                       (cell_a * cell_b)) * 180 / pi
        self._params = (cell_a, cell_b, cell_c, alpha, beta, gamma)

    def __mklammps(self):
        a, b, c, alpha, beta, gamma = self._params
        lx = a
        xy = b*math.cos(gamma*DEG2RAD)
        xz = c*math.cos(beta*DEG2RAD)
        ly = math.sqrt(b**2 - xy**2)
        yz = (b*c*math.cos(alpha*DEG2RAD) - xy*xz)/ly
        lz = math.sqrt(c**2 - xz**2 - yz**2)
        self._lammps = (lx, ly, lz, xy, xz, yz)

    @property
    def lx(self):
        return self._lammps[0]
    @property
    def ly(self):
        return self._lammps[1]
    @property
    def lz(self):
        return self._lammps[2]
    @property
    def xy(self):
        return self._lammps[3]
    @property
    def xz(self):
        return self._lammps[4]
    @property
    def yz(self):
        return self._lammps[5]

    @property
    def a(self):
        """Magnitude of cell a vector."""
        return self.params[0]

    @property
    def b(self):
        """Magnitude of cell b vector."""
        return self.params[1]

    @property
    def c(self):
        """Magnitude of cell c vector."""
        return self.params[2]

    @property
    def alpha(self):
        """Cell angle alpha."""
        return self.params[3]

    @property
    def beta(self):
        """Cell angle beta."""
        return self.params[4]

    @property
    def gamma(self):
        """Cell angle gamma."""
        return self.params[5]

class CIF(object):

    def __init__(self, name="structure", file=None):
        self.name = name
        self._data = {}
        self._headings = {}
        self._element_labels = {}
        self.non_loops = ["data", "cell", "sym", "end"]
        self.block_order = ["data", "sym", "sym_loop", "cell", "atoms", "bonds"]
        if file is not None:
            self._readfile(file)

    def read(self, filename):
        self._readfile(filename)

    def _readfile(self, filename):
        filestream = open(filename, 'r')
        filelines = filestream.readlines()
        blocks = []
        loopcount = 0
        loopentries = {}
        loopread = False
        blockread = False
        self.block_order = []

        for line in filelines:
            #line=line.replace("\n", "")
            # why not strip here?
            line = line.strip()
            if line.startswith("data_"):
                self.name = line[5:]
                self.insert_block_order("data")
                self.add_data("data", data_=self.name)

            if loopread and line.startswith("_"):
                loopentries[loopcount].append(line)

            elif loopread and not line.startswith("_"):
                loopread = False
                blockread = True

            elif not loopread and line.startswith("_"):
                block = self.get_non_loop_block(line)
                self.insert_block_order(block)
                # hopefully all non-loop entries are just single value entries, 
                # otherwise this is invalid.
                try:
                    key, val = line.strip().split()
                except ValueError:
                    print(line.strip().split())
                    key, val = line.strip().split()[:2]
                if val.endswith("(0)"):
                    val = val[:-3]
                self.add_data(block, **{key.strip():self.general_label(val)})
            
            if blockread and (line.startswith("loop_") or line.startswith("_") or not line):
                blockread = False

            if line == "loop_":
                loopcount += 1
                loopentries[loopcount] = []
                loopread = True
                blockread = False
                self.insert_block_order(loopcount)

            if blockread:
                split_line = line.strip().split()
                # problem for symmetry line
                assert len(loopentries[loopcount]) == len(split_line)
                for key, val in zip(loopentries[loopcount], split_line):
                    self.add_data(loopcount, **{key:self.general_label(val)})

        filestream.close()

    def get_time(self):
        t = date.today()
        return t.strftime("%A %d %B %Y")

    def insert_block_order(self, name, index=None):
        """Adds a block to the cif file in a specified order, unless index is specified,
        will not override existing order"""
        if index is None and name in self.block_order:
            return
        elif index is None and name not in self.block_order:
            index = len(self.block_order)
        elif index is not None and name in self.block_order and index < len(self.block_order):
            old = self.block_order.index(name)
            self.block_order.pop(old)
        elif index is not None and name in self.block_order and index >= len(self.block_order):
            old = self.block_order.index(name)
            self.block_order.pop(old)
            index = len(self.block_order)
        self.block_order = self.block_order[:index] + [name] + \
                            self.block_order[index:]

    def add_data(self, block, **kwargs):
        self._headings.setdefault(block, [])
        for key, val in kwargs.items():
            try:
                self._data[key].append(val)
            except KeyError:
                self._headings[block].append(key)
                if block in self.non_loops:
                    self._data[key] = val
                else:
                    self._data[key] = [val]

    def get_element_label(self, el):
        self._element_labels.setdefault(el, 0)
        self._element_labels[el] += 1
        return el + str(self._element_labels[el])

    def __str__(self):
        line = ""
        for block in self.block_order:
            heads = self._headings[block]
            if block in self.non_loops: 
                vals = zip([CIF.label(i) for i in heads], [self._data[i] for i in heads])
            else:
                line += "loop_\n"+"\n".join([CIF.label(i) for i in heads])+"\n"
                vals = zip(*[self._data[i] for i in heads])
            for ll in vals:
                line += "".join(ll) + "\n"
        return line

    def get_non_loop_block(self, line):
        if line.startswith("_cell"):
            return "cell"
        elif line.startswith("_symmetry"):
            return "sym"
        elif line.startswith("_audit"):
            return "data"

    # terrible idea for formatting.. but oh well :)
    @staticmethod
    def atom_site_fract_x(x):
        return "%10.5f "%(x)
    @staticmethod
    def atom_site_fract_y(x):
        return "%10.5f "%(x)
    @staticmethod
    def atom_site_fract_z(x):
        return "%10.5f "%(x)
    @staticmethod
    def atom_site_label(x):
        return "%-7s "%(x)
    @staticmethod
    def atom_site_type_symbol(x):
        return "%-6s "%(x)
    @staticmethod
    def atom_site_description(x):
        return "%-5s "%(x)
    @staticmethod
    def geom_bond_atom_site_label_1(x):
        return "%-7s "%(x)
    @staticmethod
    def geom_bond_atom_site_label_2(x):
        return "%-7s "%(x)
    @staticmethod
    def geom_bond_distance(x):
        return "%7.3f "%(x)
    @staticmethod
    def geom_bond_site_symmetry_2(x):
        return "%-5s "%(x)
    @staticmethod
    def ccdc_geom_bond_type(x):
        return "%5s "%(x)
    @staticmethod
    def cell_length_a(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_length_b(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_length_c(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_angle_alpha(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_angle_beta(x):
        return "%-7.4f "%(x)
    @staticmethod
    def cell_angle_gamma(x):
        return "%-7.4f "%(x)
    @staticmethod
    def atom_site_fragment(x):
        return "%-4i "%(x)
    @staticmethod
    def atom_site_constraints(x):
        return "%-4i "%(x)

    @staticmethod
    def label(x):
        """special cases"""
        if x == "data_":
            return x
        elif x == "_symmetry_space_group_name_H_M":
            # replace H_M with H-M. 
            x = x[:28] + "-" + x[29:]
        return "%-34s"%(x)
    
    @staticmethod
    def general_label(x):
        return "%s     "%(x)
