#!/usr/bin/env python
from datetime import date
import numpy as np
import itertools
from atomic import MASS
from ccdc import CCDC_BOND_ORDERS
DEG2RAD=np.pi/180.
class Structure(object):

    def __init__(self, name):
        self.name = name
        self.cell = Cell()
        self.atoms = []
        self.guests = []
        self.bonds = []
        self.angles = {}
        self.unique_atom_types = {}
        self.unique_bond_types = {}
        self.unique_angle_types = {}
        
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

        x, y, z = data['_atom_site_fract_x'], data['_atom_site_fract_y'], data['_atom_site_fract_z']
        
        label, element, ff_param = data['_atom_site_label'], data['_atom_site_type_symbol'], data['_atom_site_description']
        index = 0
        for l,e,ff,fx,fy,fz in zip(label,element,ff_param,x,y,z):
            fcoord = np.array([float(j) for j in (fx, fy, fz)])
            atom = Atom(element=e.strip(), coordinates = np.dot(fcoord, self.cell.cell))
            atom.force_field_type = ff.strip()
            atom.ciflabel = l.strip() 
            self.atoms.append(atom)
        # obtain bonds
        a, b, type = data['_geom_bond_atom_site_label_1'], data['_geom_bond_atom_site_label_2'], data['_ccdc_geom_bond_type']

        for label1, label2, t in zip(a,b,type):
            atm1 = self.get_atom_from_label(label1.strip())
            atm2 = self.get_atom_from_label(label2.strip())
            #TODO(check if atm2 crosses a periodic boundary to bond with atm1)
            atm1.neighbours.append(atm2.index)
            atm2.neighbours.append(atm1.index)
            bond = Bond(atid1=atm1.index, atid2=atm2.index, 
                            order=CCDC_BOND_ORDERS[t.strip()])

            self.bonds.append(bond)

        # unwrap symmetry elements if they exist

    def get_atom_from_label(self, label):
        for atom in self.atoms:
            if atom.ciflabel == label:
                return atom
    
    def unique_atoms(self):
        # ff_type keeps track of the unique integer index
        ff_type = {}
        count = 0
        for atom in self.atoms:
            
            if atom.force_field_type is None:
                label = atom.element
            else:
                label = atom.force_field_type

            try:
                type = ff_type[label][0]
            except KeyError:
                count += 1
                type = count
                ff_type[type] = (count, atom.mass)
                self.unique_atom_types[type] = (atom.mass, label)

            atom.ff_type_index = type

    def unique_bonds(self):
        count = 0
        bb_type = {}
        for bond in self.bonds:
            idx1, idx2 = bond.indices
            atm1, atm2 = self.atoms[idx1], self.atoms[idx2]
            
            try:
                type = bb_type[(atm1.ff_type_index, atm2.ff_type_index, bond.bond_order)]
            except KeyError:
                count += 1
                type = count
                bb_type[(atm1.ff_type_index, atm2.ff_type_index, bond.bond_order)] = type

                self.unique_bond_types[type] = (bond.bond_order, atm1.force_field_type, 
                                                atm2.force_field_type)
                
            bond.ff_type_index = type

    def unique_angles(self):
        count = 0
        ang_type = {}
        for atom in self.atoms:
            angles = itertools.combinations(atom.neighbours, 2)
            for (lid, rid) in angles:
                left_atom = self.atoms[lid]
                right_atom = self.atoms[rid]

                try:
                    type = ang_type[(left_atom.ff_type_index, 
                                     atom.ff_type_index,
                                     right_atom.ff_type_index,
                                     len(atom.neighbours))]
                except KeyError:
                    count += 1
                    type = count
                    ang_type[(left_atom.ff_type_index, 
                              atom.ff_type_index,
                              right_atom.ff_type_index,
                              len(atom.neighbours))] = type
                    self.unique_angle_types[type] = (left_atom.ff_type_index,
                                                     atom.ff_type_index,
                                                     right_atom.ff_type_index)

                self.angles[(left_atom.index, atom.index, right_atom.index)] = type

            

class Bond(object):
    __ID = 0

    def __init__(self, atid1=0, atid2=0, order=1):
        self.index = self.__ID
        self.bond_order = order
        self._indices = (atid1, atid2)
        self._elements = (None, None)
        self.length = 0.
        self.ff_type_index = 0
        self.midpoint = np.array([0., 0., 0.])
        Bond.__ID += 1

    def compute_length(self, coord1, coord2):
        return np.linalg.norm(np.array(coord2) - np.array(coord1))

    def get_indices(self):
        return tuple(self._indices)

    def set_indices(self, id1, id2):
        self._indices = (id1, id2)

    indices = property(get_indices, set_indices)

    def get_elements(self):
        return tuple(self._elements)

    def set_elements(self, e1, e2):
        self._elements = (e1, e2)

    elements = property(get_elements, set_elements)


class Atom(object):
    __ID = 0
    def __init__(self, element="X", coordinates=np.zeros(3)):
        self.element = element
        self.index = self.__ID
        self.neighbours = []
        self.ciflabel = None
        self.force_field_type = None
        self.coordinates = coordinates 
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
            line=line.replace("\n", "")
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
                    key, val = line.split()
                except ValueError:
                    key, val = line.split()[:2]
                if val.endswith("(0)"):
                    val = val[:-3]
                self.add_data(block, **{key:self.general_label(val)})
            
            if blockread and (line.startswith("loop_") or line.startswith("_") or not line):
                blockread = False

            if line == "loop_":
                loopcount += 1
                loopentries[loopcount] = []
                loopread = True
                blockread = False
                self.insert_block_order(loopcount)

            if blockread:
                split_line = line.split()
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
