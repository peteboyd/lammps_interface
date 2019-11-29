"""
Force field methods.
"""
from .uff import UFF_DATA
from .uff4mof import UFF4MOF_DATA
from .dreiding import DREIDING_DATA
from .uff_nonbonded import UFF_DATA_nonbonded
from .BTW import BTW_angles, BTW_dihedrals, BTW_opbends, BTW_atoms, BTW_bonds, BTW_charges
from .Dubbeldam import Dub_atoms, Dub_bonds, Dub_angles, Dub_dihedrals, Dub_impropers
#from FMOFCu import FMOFCu_angles, FMOFCu_dihedrals, FMOFCu_opbends, FMOFCu_atoms, FMOFCu_bonds
from .MOFFF import MOFFF_angles, MOFFF_dihedrals, MOFFF_opbends, MOFFF_atoms, MOFFF_bonds
from .water_models import SPC_E_atoms, TIP3P_atoms, TIP4P_atoms, TIP5P_atoms
from .gas_models import EPM2_atoms, EPM2_angles
from .lammps_potentials import BondPotential, AnglePotential, DihedralPotential, ImproperPotential, PairPotential
from .atomic import METALS
from .atomic import organic, non_metals, noble_gases, metalloids, lanthanides, actinides, transition_metals
from .atomic import alkali, alkaline_earth, main_group, metals
import math
import numpy as np
from operator import mul
import itertools
import abc
import re
import sys
from .Molecules import *


DEG2RAD = math.pi / 180.
kBtokcal = 0.00198588


class ForceField(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def bond_term(self):
        """Computes the bond parameters"""

    @abc.abstractmethod
    def angle_term(self):
        """Computes the angle parameters"""

    @abc.abstractmethod
    def dihedral_term(self):
        """Computes the dihedral parameters"""

    @abc.abstractmethod
    def improper_term(self):
        """Computes the improper dihedral parameters"""

    def compute_force_field_terms(self):
        self.compute_atomic_pair_terms()
        self.compute_bond_terms()
        self.compute_angle_terms()
        self.compute_dihedral_terms()
        self.compute_improper_terms()

    def compute_atomic_pair_terms(self):
        charges = not np.allclose(0.0, [float(self.graph.nodes[i]['charge']) for i in list(self.graph.nodes)], atol=0.00001)
        for n, data in self.graph.nodes_iter2(data=True):
            self.pair_terms(n, data, self.cutoff, charges=charges)

    def compute_bond_terms(self):
        del_edges = []
        for n1, n2, data in self.graph.edges_iter2(data=True):

            if self.bond_term((n1, n2, data)) is None:
                del_edges.append((n1, n2))
        for (n1, n2) in del_edges:
            self.graph.remove_edge(n1, n2)

    def compute_angle_terms(self):
        for b, data in self.graph.nodes_iter2(data=True):
            # compute and store angle terms
            try:
                rem_ang = []
                ang_data = data['angles']
                for (a, c), val in ang_data.items():
                    if self.angle_term((a, b, c, val)) is None:
                        rem_ang.append((a,c))
                for i in rem_ang:
                    del(data['angles'][i])

            except KeyError:
                pass

    def compute_dihedral_terms(self):
        for b, c, data in self.graph.edges_iter2(data=True):
            try:
                rem_dihed = []
                dihed_data = data['dihedrals']
                for (a, d), val in dihed_data.items():
                    if self.dihedral_term((a,b,c,d, val)) is None:
                        rem_dihed.append((a,d))
                for i in rem_dihed:
                    del(data['dihedrals'][i])

            except KeyError:
                pass

    def compute_improper_terms(self):

        for b, data in self.graph.nodes_iter2(data=True):
            try:
                rem_imp = []
                imp_data = data['impropers']
                for (a, c, d), val in imp_data.items():
                    if self.improper_term((a,b,c,d, val)) is None:
                        rem_imp.append((a,c,d))
                for i in rem_imp:
                    del(data['impropers'][i])

            except KeyError:
                pass

class UserFF(ForceField):

    def __init__(self, graph):
        self.graph = graph
        self.unique_atom_types = {}
        self.unique_bond_types = {}
        self.unique_angle_types = {}
        self.unique_dihedral_types = {}
        self.unique_improper_types = {}
        self.unique_pair_types = {}

    def bond_term(self, bond):
        return 1
    def angle_term(self, angle):
        return 1
    def dihedral_term(self, dihedral):
        return 1
    def improper_term(self, improper):
        return 1

    def unique_atoms(self):
        # ff_type keeps track of the unique integer index
        print("Here are the unique atoms")
        ff_type = {}
        count = 0
        for atom in self.structure.atoms:
            if atom.force_field_type is None:
                label = atom.element
            else:
                label = atom.force_field_type

            try:
                type = ff_type[label]
            except KeyError:
                count += 1
                type = count
                ff_type[label] = type
                self.unique_atom_types[type] = atom

            atom.ff_type_index = type
            print(atom.ff_type_index)

        for key, atom in list(self.unique_atom_types.items()):
            print(str(key) + " : " + str(atom.index))

    def unique_bonds(self):
        print("Here are the unique bonds (Total = " +
                str(len(self.structure.bonds)) + ")")
        count = 0
        bb_type = {}
        for bond in self.structure.bonds:
            idx1, idx2 = bond.indices
            atm1, atm2 = self.structure.atoms[idx1], self.structure.atoms[idx2]

            self.bond_term(bond)
            try:
                type = bb_type[(atm1.ff_type_index,
                                atm2.ff_type_index,
                                bond.order)]
            except KeyError:
                try:
                    type = bb_type[(atm2.ff_type_index,
                                    atm1.ff_type_index,
                                    bond.order)]
                except KeyError:
                    count += 1
                    type = count
                    bb_type[(atm1.ff_type_index,
                             atm2.ff_type_index,
                             bond.order)] = type

                    self.unique_bond_types[type] = bond
            bond.ff_type_index = type
            print(bond.ff_type_index)

        for key, bond in list(self.unique_bond_types.items()):
            print(str(key) + " : " + str(bond.atoms[0].index)
                    + " - " + str(bond.atoms[1].index))


    def unique_angles(self):
        print("Here are the unique angles (Total = " +
                str(len(self.structure.angles)) + ")")
        ang_type = {}
        count = 0
        for angle in self.structure.angles:
            atom_a, atom_b, atom_c = angle.atoms
            type_a, type_b, type_c = (atom_a.ff_type_index,
                                      atom_b.ff_type_index,
                                      atom_c.ff_type_index)
            # compute and store angle terms
            self.angle_term(angle)

            try:
                type = ang_type[(type_a, type_b, type_c)]

            except KeyError:
                try:
                    type = ang_type[(type_c, type_b, type_a)]

                except KeyError:
                    count += 1
                    type = count
                    ang_type[(type_a, type_b, type_c)] = type
                    self.unique_angle_types[type] = angle
            angle.ff_type_index = type
            print(angle.ff_type_index)

        for key, angle in list(self.unique_angle_types.items()):
            print(str(key) + " : " + str(angle.atoms[0].index) + "-" +
                  str(angle.atoms[1].index) + "-" +
                  str(angle.atoms[2].index))
            print(str(key) + " : " + str(angle.atoms[0].force_field_type)
                    + "-" + str(angle.atoms[1].force_field_type) +
                    "-" + str(angle.atoms[2].force_field_type))


    def unique_dihedrals(self):
        print("Here are the unique dihedrals (Total = "
                + str(len(self.structure.dihedrals)) + ")")
        count = 0
        dihedral_type = {}
        for dihedral in self.structure.dihedrals:
            atom_a, atom_b, atom_c, atom_d = dihedral.atoms
            type_a, type_b, type_c, type_d = (atom_a.ff_type_index,
                                              atom_b.ff_type_index,
                                              atom_c.ff_type_index,
                                              atom_d.ff_type_index)
            M = len(atom_c.neighbours)*len(atom_b.neighbours)
            try:
                type = dihedral_type[(type_a, type_b, type_c, type_d, M)]
            except KeyError:
                try:
                    type = dihedral_type[(type_d, type_c, type_b, type_a, M)]
                except KeyError:
                    count += 1
                    type = count
                    dihedral_type[(type_a, type_b, type_c, type_d, M)] = type
                    #self.dihedral_term(dihedral)
                    self.unique_dihedral_types[type] = dihedral
            dihedral.ff_type_index = type
            print(dihedral.ff_type_index)

        for key, dihedral in list(self.unique_dihedral_types.items()):
            print(str(key) + " : " + str(dihedral.atoms[0].index) + "-" +
                    str(dihedral.atoms[1].index) + "-" + str(dihedral.atoms[2].index)
                    + "-" + str(dihedral.atoms[3].index))
            print(str(key) + " : " + str(dihedral.atoms[0].force_field_type)
                    + "-" + str(dihedral.atoms[1].force_field_type) + "-"
                    + str(dihedral.atoms[2].force_field_type) + "-" +
                    str(dihedral.atoms[3].force_field_type))


    def unique_impropers(self):
        """How many times to list the same set of atoms ???"""
        print("Here are the unique impropers (Total = " +
                str(len(self.structure.impropers)) + ")")
        count = 0
        improper_type = {}
        #i = 0
        #for improper in self.structure.impropers:
        #    i += 1
        #    print(str(i) + " : " + str(improper.atoms[0].force_field_type) + "-" + str(improper.atoms[1].force_field_type) +     "-" + str(improper.atoms[2].force_field_type) + "-" + str(improper.atoms[3].force_field_type)


        for improper in self.structure.impropers:
            print("Now keys are + " + str(improper_type.keys()))
            atom_a, atom_b, atom_c, atom_d = improper.atoms
            type_a, type_b, type_c, type_d = (atom_a.ff_type_index, atom_b.ff_type_index,
                                              atom_c.ff_type_index, atom_d.ff_type_index)
            d1 = (type_b, type_a, type_c, type_d)
            d2 = (type_b, type_a, type_d, type_c)
            d3 = (type_b, type_c, type_d, type_a)
            d4 = (type_b, type_c, type_a, type_d)
            d5 = (type_b, type_d, type_a, type_c)
            d6 = (type_b, type_d, type_c, type_a)

            if d1 in improper_type.keys():
                print("found d1" + str(d1))
                type = improper_type[d1]
            elif d2 in improper_type.keys():
                print("found d2")
                type = improper_type[d2]
            elif d3 in improper_type.keys():
                print("found d3")
                type = improper_type[d3]
            elif d4 in improper_type.keys():
                print("found d4")
                type = improper_type[d4]
            elif d5 in improper_type.keys():
                print("found d5")
                type = improper_type[d5]
            elif d6 in improper_type.keys():
                print("found d6")
                type = improper_type[d6]
            else:
                print("found else" + str(d1))
                count += 1
                type = count
                improper_type[d1] = type
                self.unique_improper_types[type] = improper

            improper.ff_type_index = type
            print(improper.ff_type_index)

        for key, improper in list(self.unique_improper_types.items()):
            print(str(key) + " : " + str(improper.atoms[0].force_field_type) +
                    "-" + str(improper.atoms[1].force_field_type) + "-" +
                    str(improper.atoms[2].force_field_type) + "-" +
                    str(improper.atoms[3].force_field_type))

    def van_der_waals_pairs(self):
        atom_types = self.unique_atom_types.keys()
        for type1, type2 in itertools.combinations_with_replacement(atom_types, 2):
            atm1 = self.unique_atom_types[type1]
            atm2 = self.unique_atom_types[type2]

            print(str(re.findall(r'^[a-zA-Z]*',atm1.force_field_type)[0]))
            print(str(re.findall(r'^[a-zA-Z]*',atm2.force_field_type)[0]))

            # if we are using non-UFF atom types, need to splice off the end descriptors (first non alphabetic char)
            eps1 = UFF_DATA_nonbonded[re.findall(r'^[a-zA-Z]*',atm1.force_field_type)[0]][3]
            eps2 = UFF_DATA_nonbonded[re.findall(r'^[a-zA-Z]*',atm2.force_field_type)[0]][3]

            # radius --> sigma = radius*2**(-1/6)
            sig1 = UFF_DATA_nonbonded[re.findall(r'^[a-zA-Z]*',atm1.force_field_type)[0]][2]*(2**(-1./6.))
            sig2 = UFF_DATA_nonbonded[re.findall(r'^[a-zA-Z]*',atm2.force_field_type)[0]][2]*(2**(-1./6.))

            # l-b mixing
            eps = math.sqrt(eps1*eps2)
            sig = (sig1 + sig2) / 2.
            self.unique_pair_types[(type1, type2)] = (eps, sig)

    def parse_user_input(self, filename):
        infile = open("user_input.txt","r")
        lines = infile.readlines()

        # type of interaction found: 1= bonds, 2 = angles, 3 = dihedrals, 4 = torsions
        parse_type = 0

        for line in lines:
            match = line.lower().strip()
            if match == "bonds":
                print("parsing bond")
                parse_type = 1
                continue
            elif match == "angles":
                print("parsing angle")
                parse_type = 2
                continue
            elif match == "dihedrals":
                print("parsing dihedral")
                parse_type = 3
                continue
            elif match == "impropers":
                print("parsing impropers")
                parse_type = 4
                continue

            data = line.split()
            print(data)
            if parse_type == 1:
                atms = [data[0], data[1]]
                bond_pair = [self.map_user_to_unique_atom(atms[0]),
                              self.map_user_to_unique_atom(atms[1])]
                bond_id = self.map_pair_unique_bond(bond_pair, atms)
                self.unique_bond_types[bond_id].function = data[2]
                self.unique_bond_types[bond_id].parameters = data[3:]

            elif parse_type == 2:
                atms = [data[0], data[1], data[2]]
                angle_triplet = [self.map_user_to_unique_atom(atms[0]),
                                 self.map_user_to_unique_atom(atms[1]),
                                 self.map_user_to_unique_atom(atms[2])]
                angle_id = self.map_triplet_unique_angle(angle_triplet, atms)
                self.unique_angle_types[angle_id].function = data[3]
                self.unique_angle_types[angle_id].parameters = data[4:]

            elif parse_type == 3:
                atms = [data[0], data[1], data[2], data[3]]
                dihedral_quadruplet = [self.map_user_to_unique_atom(atms[0]),
                                       self.map_user_to_unique_atom(atms[1]),
                                       self.map_user_to_unique_atom(atms[2]),
                                       self.map_user_to_unique_atom(atms[3])]
                dihedral_id = self.map_quadruplet_unique_dihedral(dihedral_quadruplet, atms)
                self.unique_dihedral_types[dihedral_id].function = data[4]
                self.unique_dihedral_types[dihedral_id].parameters = data[5:]

            elif parse_type == 4:
                atms = [data[0], data[1], data[2], data[3]]
                improper_quadruplet = [self.map_user_to_unique_atom(atms[0]),
                                       self.map_user_to_unique_atom(atms[1]),
                                       self.map_user_to_unique_atom(atms[2]),
                                       self.map_user_to_unique_atom(atms[3])]
                improper_id = self.map_quadruplet_unique_improper(improper_quadruplet, atms)
                self.unique_improper_types[improper_id].function = data[4]
                self.unique_improper_types[improper_id].parameters = data[5:]



    def write_missing_uniques(self, description):
        # Warn user about any unique bond, angle, etc. found that have not
        # been specified in user_input.txt
        pass

    def map_user_to_unique_atom(self, descriptor):
        for key, atom in list(self.unique_atom_types.items()):
            if descriptor == atom.force_field_type:
                return atom.ff_type_index

        raise ValueError('Error! An atom identifier ' + str(description) +
                ' in user_input.txt did not match any atom_site_description in your cif')

    def map_pair_unique_bond(self, pair, descriptor):
        for key, bond in list(self.unique_bond_types.items()):
            if (pair == [bond.atoms[0].ff_type_index, bond.atoms[1].ff_type_index]
                or pair == [bond.atoms[1].ff_type_index, bond.atoms[0].ff_type_index]):
                return key

        raise ValueError('Error! An bond identifier ' + str(descriptor) +
                ' in user_input.txt did not match any bonds in your cif')

    def map_triplet_unique_angle(self, triplet, descriptor):
        #print(triplet)
        #print(descriptor)
        for key, angle in list(self.unique_angle_types.items()):
            #print(str(key) + " : " + str([angle.atoms[2].ff_type_index, angle.atoms[1].ff_type_index, angle.atoms[0].ff_type_index]))
            if (triplet == [angle.atoms[0].ff_type_index,
                            angle.atoms[1].ff_type_index,
                            angle.atoms[2].ff_type_index] or
                triplet == [angle.atoms[2].ff_type_index,
                            angle.atoms[1].ff_type_index,
                            angle.atoms[0].ff_type_index]):
                return key

        raise ValueError('Error! An angle identifier ' + str(descriptor) +
                ' in user_input.txt did not match any angles in your cif')

    def map_quadruplet_unique_dihedral(self, quadruplet, descriptor):
        for key, dihedral in list(self.unique_dihedral_types.items()):
            if (quadruplet == [dihedral.atoms[0].ff_type_index,
                               dihedral.atoms[1].ff_type_index,
                               dihedral.atoms[2].ff_type_index,
                               dihedral.atoms[3].ff_type_index] or
                quadruplet == [dihedral.atoms[3].ff_type_index,
                               dihedral.atoms[2].ff_type_index,
                               dihedral.atoms[1].ff_type_index,
                               dihedral.atoms[0].ff_type_index]):
                return key

        raise ValueError('Error! A dihdral identifier ' + str(descriptor) +
                ' in user_input.txt did not match any dihedrals in your cif')

    def map_quadruplet_unique_improper(self, quadruplet, descriptor):
        for key, improper in list(self.unique_improper_types.items()):
            if (quadruplet == [improper.atoms[0].ff_type_index,
                               improper.atoms[1].ff_type_index,
                               improper.atoms[2].ff_type_index,
                               improper.atoms[3].ff_type_index] or
                quadruplet == [improper.atoms[3].ff_type_index,
                               improper.atoms[2].ff_type_index,
                               improper.atoms[1].ff_type_index,
                               improper.atoms[0].ff_type_index]):
                return key

        raise ValueError('Error! An improper identifier ' + str(descriptor) +
                ' in user_input.txt did not match any improper in your cif')

    def overwrite_force_field_terms(self):
        self.parse_user_input("blah")

    def compute_force_field_terms(self):
        self.unique_atoms()
        self.unique_bonds()
        self.unique_angles()
        self.unique_dihedrals()
        self.unique_impropers()

        self.parse_user_input("blah")
        self.van_der_waals_pairs()

class OverwriteFF(ForceField):
    """
    Prepare a nanoprous material FF from a given structure for a known
    FF type.

    Then overwrite any parameters that are supplied by user_input.txt

    Methods are duplicated from UserFF, can reduce redundancy of code
    later if desired
    """

    def __init__(self, struct, base_FF):
        # Assign the base ForceField
        if(baseFF == "UFF"):
            self = UFF(struct)
        elif(baseFF == "DREIDING"):
            self = Dreiding(struct)
        elif(baseFF == "CVFF"):
            print("CVFF not implemented yet...")
            sys.exit()
            pass
        elif(baseFF == "CHARMM"):
            print("CHARMM not implemented yet...")
            sys.exit()
            pass
        else:
            # etc. TODO worth adding in these additional FF types
            print("Invalid base FF requested\nExiting...")
            sys.exit()

        # Overwrite any parameters specified by user_input.txt
        parse_user_input("user_input.txt")

    def parse_user_input(self, filename):
        infile = open("user_input.txt","r")
        lines = infile.readlines()

        # type of interaction found: 1= bonds, 2 = angles, 3 = dihedrals, 4 = torsions
        parse_type = 0

        for line in lines:
            match = line.lower().strip()
            if match == "bonds":
                print("parsing bond")
                parse_type = 1
                continue
            elif match == "angles":
                print("parsing angle")
                parse_type = 2
                continue
            elif match == "dihedrals":
                print("parsing dihedral")
                parse_type = 3
                continue
            elif match == "impropers":
                print("parsing impropers")
                parse_type = 4
                continue

            data = line.split()
            print(data)
            if parse_type == 1:
                atms = [data[0], data[1]]
                bond_pair = [self.map_user_to_unique_atom(atms[0]),
                              self.map_user_to_unique_atom(atms[1])]
                bond_id = self.map_pair_unique_bond(bond_pair, atms)
                self.unique_bond_types[bond_id].function = data[2]
                self.unique_bond_types[bond_id].parameters = data[3:]

            elif parse_type == 2:
                atms = [data[0], data[1], data[2]]
                angle_triplet = [self.map_user_to_unique_atom(atms[0]),
                                 self.map_user_to_unique_atom(atms[1]),
                                 self.map_user_to_unique_atom(atms[2])]
                angle_id = self.map_triplet_unique_angle(angle_triplet, atms)
                self.unique_angle_types[angle_id].function = data[3]
                self.unique_angle_types[angle_id].parameters = data[4:]

            elif parse_type == 3:
                atms = [data[0], data[1], data[2], data[3]]
                dihedral_quadruplet = [self.map_user_to_unique_atom(atms[0]),
                                       self.map_user_to_unique_atom(atms[1]),
                                       self.map_user_to_unique_atom(atms[2]),
                                       self.map_user_to_unique_atom(atms[3])]
                dihedral_id = self.map_quadruplet_unique_dihedral(dihedral_quadruplet, atms)
                self.unique_dihedral_types[dihedral_id].function = data[4]
                self.unique_dihedral_types[dihedral_id].parameters = data[5:]

            elif parse_type == 4:
                atms = [data[0], data[1], data[2], data[3]]
                improper_quadruplet = [self.map_user_to_unique_atom(atms[0]),
                                       self.map_user_to_unique_atom(atms[1]),
                                       self.map_user_to_unique_atom(atms[2]),
                                       self.map_user_to_unique_atom(atms[3])]
                improper_id = self.map_quadruplet_unique_improper(improper_quadruplet, atms)
                self.unique_improper_types[improper_id].function = data[4]
                self.unique_improper_types[improper_id].parameters = data[5:]



    def write_missing_uniques(self, description):
        # Warn user about any unique bond, angle, etc. found that have not
        # been specified in user_input.txt
        pass



    def map_user_to_unique_atom(self, descriptor):
        for key, atom in list(self.unique_atom_types.items()):
            if descriptor == atom.force_field_type:
                return atom.ff_type_index

        raise ValueError('Error! An atom identifier ' + str(description) +
                ' in user_input.txt did not match any atom_site_description in your cif')

    def map_pair_unique_bond(self, pair, descriptor):
        for key, bond in list(self.unique_bond_types.items()):
            if (pair == [bond.atoms[0].ff_type_index, bond.atoms[1].ff_type_index]
                or pair == [bond.atoms[1].ff_type_index, bond.atoms[0].ff_type_index]):
                return key

        raise ValueError('Error! A bond identifier ' + str(descriptor) +
                ' in user_input.txt did not match any bonds in your cif')

    def map_triplet_unique_angle(self, triplet, descriptor):
        #print(triplet)
        #print(descriptor)
        for key, angle in list(self.unique_angle_types.items()):
            #print(str(key) + " : " + str([angle.atoms[2].ff_type_index, angle.atoms[1].ff_type_index, angle.atoms[0].ff_type_index]))
            if (triplet == [angle.atoms[0].ff_type_index,
                            angle.atoms[1].ff_type_index,
                            angle.atoms[2].ff_type_index] or
                triplet == [angle.atoms[2].ff_type_index,
                            angle.atoms[1].ff_type_index,
                            angle.atoms[0].ff_type_index]):
                return key

        raise ValueError('Error! An angle identifier ' + str(descriptor) +
                ' in user_input.txt did not match any angles in your cif')

    def map_quadruplet_unique_dihedral(self, quadruplet, descriptor):
        for key, dihedral in list(self.unique_dihedral_types.items()):
            if (quadruplet == [dihedral.atoms[0].ff_type_index,
                               dihedral.atoms[1].ff_type_index,
                               dihedral.atoms[2].ff_type_index,
                               dihedral.atoms[3].ff_type_index] or
                quadruplet == [dihedral.atoms[3].ff_type_index,
                               dihedral.atoms[2].ff_type_index,
                               dihedral.atoms[1].ff_type_index,
                               dihedral.atoms[0].ff_type_index]):
                return key

        raise ValueError('Error! A dihdral identifier ' + str(descriptor) +
                ' in user_input.txt did not match any dihedrals in your cif')

    def map_quadruplet_unique_improper(self, quadruplet, descriptor):
        for key, improper in list(self.unique_improper_types.items()):
            if (quadruplet == [improper.atoms[0].ff_type_index,
                               improper.atoms[1].ff_type_index,
                               improper.atoms[2].ff_type_index,
                               improper.atoms[3].ff_type_index] or
                quadruplet == [improper.atoms[3].ff_type_index,
                               improper.atoms[2].ff_type_index,
                               improper.atoms[1].ff_type_index,
                               improper.atoms[0].ff_type_index]):
                return key

        raise ValueError('Error! An improper identifier ' + str(descriptor) +
                ' in user_input.txt did not match any improper in your cif')


class BTW_FF(ForceField):
    def __init__(self,  **kwargs):
        self.pair_in_data = False
        self.keep_metal_geometry = False
        self.graph = None
        # override existing arguments with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        if (self.graph is not None):
            self.detect_ff_terms()
            self.compute_force_field_terms()

    def detect_ff_terms(self):
        """
        Assigning force field type of atoms
        """

        # for each atom determine the ff type if it is None
        BTW_organics = ["O", "C", "H"]
        mof_sbus = set(self.graph.inorganic_sbus.keys())
        BTW_sbus = set(["Cu Paddlewheel", "Zn4O", "Zr_UiO"])
        if not (mof_sbus <= BTW_sbus):
            print("The system cannot be simulated with BTW-FF!")
            sys.exit()
        elif ( len(mof_sbus)> 1):
            print("No exact charge for the IRMOF is available from BTW-FF. Average charges in BTW-FF is used.")
            chrg_flag="TFF_" # Transferable FF charges (average values)
        elif("Zn4O" in mof_sbus):
            #resp= input("What is the IRMOF number?[1/10]")
            # temp fix so I don't have to respond to screen prompt
            resp = "0"
            if self.graph.number_of_nodes() == 424:
                resp = "1"
            elif self.graph.number_of_nodes() == 664:
                resp = "10"

            if resp=="1":
                chrg_flag="Zn4O_"
            elif resp=="10":
                chrg_flag="IRMOF10_"
            else:
                #print("No exact charge for the IRMOF is available from BTW-FF. Average charges in BTW-FF is used.")
                #chrg_flag="TFF_"
                print("Cannot parameterize this MOF with BTW-FF.")
                sys.exit()
        else:
            sbu_type = next(iter(mof_sbus))
            chrg_flag=sbu_type+"_"

        #Assigning force field type of atoms
        for node, atom in self.graph.nodes_iter2(data=True):
            # check if element not in one of the SBUS
            if atom['element'] == "Cu":
                try:
                    if atom['special_flag'] == 'Cu_pdw':
                        atom['force_field_type']="185"
                        chrg_key = chrg_flag+atom['force_field_type']

                        atom['charge']=BTW_charges[chrg_key]
                    else:
                        print("ERROR: Cu %i is not assigned to a Cu Paddlewheel! exiting"%(node))
                        sys.exit()
                except KeyError:
                    print("ERROR: Cu %i is not assigned to a Cu Paddlewheel! exiting"%(node))
                    sys.exit()

            elif atom['element'] == "Zn":
                try:
                    if atom['special_flag'] == 'Zn4O':
                        atom['force_field_type']="172"
                        chrg_key = chrg_flag+atom['force_field_type']

                        atom['charge']=BTW_charges[chrg_key]

                    else:
                        print("ERROR: Zn %i is not assigned to a Zn4O! exiting"%(node))
                        sys.exit()
                except KeyError:
                    print("ERROR: Zn %i is not assigned to a Zn4O! exiting"%(node))
                    sys.exit()


            elif atom['element'] == "Zr":
                try:
                    if atom['special_flag'] == 'Zr_UiO':
                        atom['force_field_type']="192"
                        chrg_key = chrg_flag+atom['force_field_type']

                        atom['charge']=BTW_charges[chrg_key]

                    else:
                        print("ERROR: Zr %i is not assigned to a Zr_UiO! exiting"%(node))
                        sys.exit()
                except KeyError:
                    print("ERROR: Zr %i is not assigned to a Zr_UiO! exiting"%(node))
                    sys.exit()


            if atom['force_field_type'] is None:
                type_assigned = False
                neighbours = [self.graph.nodes[i] for i in self.graph.neighbors(node)]
                neighbour_elements = [a['element'] for a in neighbours]
                special = False
                if 'special_flag' in atom:
                    special = True
                if (atom['element'] == "O") and special:
                    # Zn4O cases
                    if atom['special_flag'] == "O_z_Zn4O":
                        atom['force_field_type'] = "171"
                        chrg_key = chrg_flag+atom['force_field_type']
                        atom['charge']=BTW_charges[chrg_key]
                    elif atom['special_flag'] == "O_c_Zn4O":
                        atom['force_field_type'] = "170"
                        chrg_key = chrg_flag+atom['force_field_type']
                        atom['charge']=BTW_charges[chrg_key]
                    # Zr_UiO cases
                    elif atom['special_flag'] == "O_z_Zr_UiO":
                        atom['force_field_type'] = "171"
                        chrg_key = chrg_flag+atom['force_field_type']
                        atom['charge']=BTW_charges[chrg_key]
                    elif atom['special_flag'] == "O_h_Zr_UiO":
                        atom['force_field_type'] = "75"
                        chrg_key = chrg_flag+atom['force_field_type']
                        atom['charge']=BTW_charges[chrg_key]
                    elif atom['special_flag'] == "O_c_Zr_UiO":
                        atom['force_field_type'] = "170"
                        chrg_key = chrg_flag+atom['force_field_type']
                        atom['charge']=BTW_charges[chrg_key]
                    # Cu Paddlewheel case
                    elif (atom['special_flag'] == "O1_Cu_pdw") or (atom['special_flag'] == "O2_Cu_pdw"):
                        atom['force_field_type'] = "170"
                        chrg_key = chrg_flag+atom['force_field_type']
                        atom['charge']=BTW_charges[chrg_key]
                    else:
                        print("Oxygen number %i type cannot be detected!"%node)
                        sys.exit()
                elif (atom['element'] == "C") and special:
                    # Zn4O case
                    if atom['special_flag'] == "C_Zn4O":
                        atom['force_field_type'] = "913" # C-acid
                        chrg_key = chrg_flag+atom['force_field_type']
                        atom['charge']=BTW_charges[chrg_key]
                    # Zr_UiO case
                    elif atom['special_flag'] == "C_Zr_UiO":
                        atom['force_field_type'] = "913" # C-acid
                        chrg_key = chrg_flag+atom['force_field_type']
                        atom['charge']=BTW_charges[chrg_key]
                    # Cu Paddlewheel case
                    elif atom['special_flag'] == "C_Cu_pdw":
                        atom['force_field_type'] = "913" # C-acid
                        chrg_key = chrg_flag+atom['force_field_type']
                        atom['charge']=BTW_charges[chrg_key]
                    else:
                        print("Carbon number %i type cannot be detected!"%node)
                        sys.exit()

                elif (atom['element'] == "H") and special:
                    # only UiO case
                    if atom['special_flag'] == "H_o_Zr_UiO":
                        atom['force_field_type'] = "21"
                        chrg_key = chrg_flag+atom['force_field_type']
                        atom['charge']=BTW_charges[chrg_key]
                    else:
                        print("Hydrogen number %i type cannot be detected!"%node)
                        sys.exit()

                # currently no oxygens assigned types outside of metal SBUs
                elif (atom['element'] == "O") and not special:
                    print("Oxygen number %i type cannot be detected!"%node)
                    sys.exit()

                elif (atom['element'] == "C") and not special:
                    # all organic SBUs have the same types..
                    if set(neighbour_elements) == set(["C","H"]):
                        atom['force_field_type'] = "912" # C- benzene we should be careful that in this case C in ligand has also bond with H, but not in the FF
                        chrg_key = chrg_flag+atom['force_field_type']
                        atom['charge']=BTW_charges[chrg_key]
                    elif set(neighbour_elements) == set(["C"]):
                        # check if carbon adjacent to metal SBU (signified by the key 'special_flag')
                        if any(['special_flag' in at for at in neighbours]):
                            atom['force_field_type'] = "902"
                        else:
                            atom['force_field_type'] = "903"
                        chrg_key = chrg_flag+atom['force_field_type']

                        atom['charge']=BTW_charges[chrg_key]
                    else:
                        print("Carbon number %i type cannot be detected!"%node)
                        sys.exit()

                elif (atom['element'] == "H") and not special:
                    if set(neighbour_elements)<=set(["C"]):
                        atom['force_field_type'] = "915"
                        chrg_key = chrg_flag+atom['force_field_type']

                        atom['charge']=BTW_charges[chrg_key]
                    else:
                        print("Hydrogen number %i type cannot be detected!"%node)
                        sys.exit()

        # Assigning force field type of bonds
        for a, b, bond in self.graph.edges_iter2(data=True):
            a_atom = self.graph.nodes[a]
            b_atom = self.graph.nodes[b]
            atom_a_fflabel, atom_b_fflabel = a_atom['force_field_type'], b_atom['force_field_type']
            bond_fflabel1 = atom_a_fflabel+"_"+atom_b_fflabel
            bond_fflabel2 = atom_b_fflabel+"_"+atom_a_fflabel
            if bond_fflabel1 in BTW_bonds:
                bond['force_field_type']=bond_fflabel1
            elif bond_fflabel2 in BTW_bonds:
                bond['force_field_type']=bond_fflabel2
            else:
                print ("BTW-FF cannot be used for the system!\nNo parameter found for bond %s"%(bond_fflabel1))
                sys.exit()

        #Assigning force field type of angles
        missing_labels=[]
        for b , data in self.graph.nodes_iter2(data=True):
            try:
                missing_angles=[]
                ang_data = data['angles']
                for (a, c), val in ang_data.items():
                    a_atom = self.graph.nodes[a]
                    b_atom = data
                    c_atom = self.graph.nodes[c]
                    atom_a_fflabel = a_atom['force_field_type']
                    atom_b_fflabel = b_atom['force_field_type']
                    atom_c_fflabel = c_atom['force_field_type']
                    angle_fflabel1=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel
                    angle_fflabel2=atom_c_fflabel+"_"+atom_b_fflabel+"_"+atom_a_fflabel
                    if (angle_fflabel1=="170_185_170"):
                        val['force_field_type']=angle_fflabel1
                    elif angle_fflabel1 in BTW_angles:
                        val['force_field_type']=angle_fflabel1
                    elif angle_fflabel2 in BTW_angles:
                        val['force_field_type']=angle_fflabel2
                    else:
                        missing_angles.append((a,c))
                        missing_labels.append(angle_fflabel1)
                for key in missing_angles:
                    del ang_data[key]
            except KeyError:
                pass

        for ff_label in set(missing_labels):
            print ("%s angle is deleted since the angle was not parametrized in BTW-FF!"%(ff_label))

        #Assigning force field type of dihedrals
        missing_labels=[]
        for b, c, data in self.graph.edges_iter2(data=True):
            try:
                missing_dihedral=[]
                dihed_data = data['dihedrals']
                for (a, d), val in dihed_data.items():
                    a_atom = self.graph.nodes[a]
                    b_atom = self.graph.nodes[b]
                    c_atom = self.graph.nodes[c]
                    d_atom = self.graph.nodes[d]
                    atom_a_fflabel = a_atom['force_field_type']
                    atom_b_fflabel = b_atom['force_field_type']
                    atom_c_fflabel = c_atom['force_field_type']
                    atom_d_fflabel = d_atom['force_field_type']
                    dihedral_fflabel1=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel
                    dihedral_fflabel2=atom_d_fflabel+"_"+atom_c_fflabel+"_"+atom_b_fflabel+"_"+atom_a_fflabel

                    if dihedral_fflabel1 in BTW_dihedrals:
                        val['force_field_type']=dihedral_fflabel1
                    elif dihedral_fflabel2 in BTW_dihedrals:
                        val['force_field_type']=dihedral_fflabel2
                    else:
                        missing_dihedral.append((a,d))
                        missing_labels.append(dihedral_fflabel1)
                for key in missing_dihedral:
                    del dihed_data[key]
            except KeyError:
                pass
        for ff_label in set(missing_labels):
            print ("%s dihedral is deleted since the dihedral was not parametrized in BTW-FF!"%(ff_label))

        #Assigning force field type of impropers
        missing_labels=[]
        for b, data in self.graph.nodes_iter2(data=True):
            try:
                missing_improper=[]
                imp_data = data['impropers']
                for (a, c, d), val in imp_data.items():
                    a_atom = self.graph.nodes[a]
                    b_atom = self.graph.nodes[b]
                    c_atom = self.graph.nodes[c]
                    d_atom = self.graph.nodes[d]
                    atom_a_fflabel = a_atom['force_field_type']
                    atom_b_fflabel = b_atom['force_field_type']
                    atom_c_fflabel = c_atom['force_field_type']
                    atom_d_fflabel = d_atom['force_field_type']
                    improper_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel
                    if improper_fflabel in BTW_opbends:
                        val['force_field_type']=improper_fflabel
                    else:
                        missing_improper.append((a,c,d))
                        missing_labels.append(improper_fflabel)
                for key in missing_improper:
                    del imp_data[key]
            except KeyError:
                pass

        for ff_label in set(missing_labels):
            print ("%s improper is deleted since the improper was not parametrized in BTW-FF!"%(ff_label))

    def bond_term(self, edge):
        """class2 bond: 4-order polynomial """
        n1, n2, data = edge
        Ks =  BTW_bonds[data['force_field_type']][0]
        l0 =  BTW_bonds[data['force_field_type']][1]
        ### All the factors are conversion to kcal/mol from the units in the paper ###
        K2= 71.94*Ks
        K3= -2.55*K2
        K4= 3.793125*K2
        data['potential'] = BondPotential.Class2()
        data['potential'].K2 = K2
        data['potential'].K3 = K3
        data['potential'].K4 = K4
        data['potential'].R0 = l0
        return 1

    def angle_term(self, angle):
        """class2 angle

        NOTE: We ignored the 5and6 order terms of polynomial since the functional is not implemented in LAMMPS!!
        """
        a, b, c, data = angle

        if (data['force_field_type']=="170_185_170"):   ### in the case of square planar coordination of Cu-paddle-wheel, fourier angle must be used
            data['potential'] = AnglePotential.CosinePeriodic()
            data['potential'].C = 126.64 # conversion from the K value in MOF-FF to the C value for LAMMPS in cosine/periodic
            data['potential'].B = 1
            data['potential'].n = 4
            return 1
        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        ab_bond = self.graph[a][b]
        bc_bond = self.graph[b][c]
        atom_a_fflabel = a_data['force_field_type']
        atom_b_fflabel = b_data['force_field_type']
        atom_c_fflabel = c_data['force_field_type']
        ang_ff_tmp = atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel

        Ktheta = BTW_angles[data['force_field_type']][0]
        theta0 = BTW_angles[data['force_field_type']][1]
        ### BondAngle ###
        baN1 = BTW_angles[data['force_field_type']][4]
        baN2 = BTW_angles[data['force_field_type']][5]
        ### BondBond ###
        bbM  = BTW_angles[data['force_field_type']][6]

        if not (ang_ff_tmp == data['force_field_type']):  # switching the force constants in the case of assigning swaped angle_ff_label
            buf1 = atom_a_fflabel
            atom_a_fflabel = atom_c_fflabel
            atom_c_fflabel = buf1
            buf2 = baN1
            baN1=baN2
            baN2=buf2
        ### assingning the equilibrium distance of each bond from FF
        bond1_label = atom_a_fflabel+"_"+atom_b_fflabel
        bond2_label = atom_b_fflabel+"_"+atom_c_fflabel
        if (bond1_label) in BTW_bonds:
            r1 = BTW_bonds[bond1_label][1]
        else:
            bond1_label = atom_b_fflabel+"_"+atom_a_fflabel
            r1 = BTW_bonds[bond1_label][1]

        if (bond2_label) in BTW_bonds:
            r2 = BTW_bonds[bond2_label][1]
        else:
            bond2_label = atom_c_fflabel+"_"+atom_b_fflabel
            r2 = BTW_bonds[bond2_label][1]
        ### Unit conversion ###
        bbM = bbM *71.94
        baN1 = 2.51118 * baN1 / (DEG2RAD)
        baN2 = 2.51118 * baN2/ (DEG2RAD)
        # 0.021914 is 143.82/2 * (pi/180)**2
        K2 = 0.021914*Ktheta/(DEG2RAD**2)
        K3 = -0.014*K2/(DEG2RAD**1)
        K4 = 5.6e-5*K2/(DEG2RAD**2)

        data['potential'] = AnglePotential.Class2()
        data['potential'].theta0 = theta0
        data['potential'].K2 = K2
        data['potential'].K3 = K3
        data['potential'].K4 = K4
        data['potential'].ba.N1 = baN1
        data['potential'].ba.N2 = baN2
        data['potential'].ba.r1 = r1
        data['potential'].ba.r2 = r2
        return 1


    def dihedral_term(self, dihedral):
        """ fourier diherdral """
        a,b,c,d, data = dihedral

        kt1 = 0.5 * BTW_dihedrals[data['force_field_type']][0]
        kt2 = 0.5 * BTW_dihedrals[data['force_field_type']][3]
        kt3 = 0.5 * BTW_dihedrals[data['force_field_type']][6]
        kt4 = 0.5 * BTW_dihedrals[data['force_field_type']][9]
        n1 = BTW_dihedrals[data['force_field_type']][2]
        n2 = BTW_dihedrals[data['force_field_type']][5]
        n3 = BTW_dihedrals[data['force_field_type']][8]
        n4 = BTW_dihedrals[data['force_field_type']][11]
        d1 = BTW_dihedrals[data['force_field_type']][1]
        d2 = BTW_dihedrals[data['force_field_type']][4]
        d3 = BTW_dihedrals[data['force_field_type']][7]
        d4 = BTW_dihedrals[data['force_field_type']][10]

        ki = [kt1,kt2,kt3,kt4]
        ni = [n1,n2,n3,n4]
        di = [d1,d2,d3,d4]

        data['potential'] = DihedralPotential.Fourier()
        data['potential'].Ki = ki
        data['potential'].ni = ni
        data['potential'].di = di
        return 1

    def improper_term(self, improper):
        """class2 improper"""
        a,b,c,d, data = improper
        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        d_data = self.graph.nodes[d]
        atom_a_fflabel=a_data['force_field_type']
        atom_b_fflabel=b_data['force_field_type']
        atom_c_fflabel=c_data['force_field_type']
        atom_d_fflabel=d_data['force_field_type']
        # 0.021914 is 143.82/2 * (pi/180)**-2
        Kopb = BTW_opbends[data['force_field_type']][0]/(DEG2RAD**2)*0.02191418
        c0 =  BTW_opbends[data['force_field_type']][1]
        #Angle-Angle term
        # units should be energy/distance
        M1 = BTW_opbends[data['force_field_type']][2]/(DEG2RAD**2)*0.02191418*(-1)/3.  # Dividing by three to get rid of overcounting of angle-angle terms
        M2 = BTW_opbends[data['force_field_type']][3]/(DEG2RAD**2)*0.02191418*(-1)/3.  # Dividing by three to get rid of overcounting of angle-angle terms
        M3 = BTW_opbends[data['force_field_type']][4]/(DEG2RAD**2)*0.02191418*(-1)/3.  # Dividing by three to get rid of overcounting of angle-angle terms

        ang1_ff_label = atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel
        ang2_ff_label = atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_d_fflabel
        ang3_ff_label = atom_c_fflabel+"_"+atom_b_fflabel+"_"+atom_d_fflabel
        if (ang1_ff_label) in BTW_angles:
            Theta1 =  BTW_angles[ang1_ff_label][1]
        else:
            ang1_ff_label = atom_c_fflabel+"_"+atom_b_fflabel+"_"+atom_a_fflabel
            Theta1 =  BTW_angles[ang1_ff_label][1]
        if (ang2_ff_label) in BTW_angles:
            Theta2 =  BTW_angles[ang2_ff_label][1]
        else:
            ang2_ff_label = atom_d_fflabel+"_"+atom_b_fflabel+"_"+atom_a_fflabel
            Theta2 =  BTW_angles[ang2_ff_label][1]
        if (ang3_ff_label) in BTW_angles:
            Theta3 =  BTW_angles[ang3_ff_label][1]
        else:
            ang3_ff_label = atom_d_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel
            Theta3 =  BTW_angles[ang3_ff_label][1]

        data['potential'] =  ImproperPotential.Class2() #does not work now!
        data['potential'].K = Kopb
        data['potential'].chi0 = c0
        data['potential'].aa.M1 = M1
        data['potential'].aa.M2 = M2
        data['potential'].aa.M3 = M3
        data['potential'].aa.theta1 = Theta1
        data['potential'].aa.theta2 = Theta2
        data['potential'].aa.theta3 = Theta3
        return 1

    def pair_terms( self, node , data, cutoff, **kwargs):
        """
        Buckingham equation in MM3 type is used!
        """
        eps = BTW_atoms[data['force_field_type']][4]
        sig = BTW_atoms[data['force_field_type']][3]

        data['pair_potential']=PairPotential.BuckCoulLong()
        data['pair_potential'].cutoff= cutoff
        data['pair_potential'].eps = eps
        data['pair_potential'].sig = sig


    def special_commands(self):
        st = ["%-15s %s"%("pair_modify", "tail yes"),
              "%-15s %s"%("special_bonds", "lj/coul 0.0 0.0 1"),
              "%-15s %.2f"%('dielectric', 1.5)]
        return st


class MOF_FF(ForceField):

    def __init__(self, **kwargs):
        self.pair_in_data = False
        self.keep_metal_geometry = False
        self.graph = None
        # override existing arguments with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        if (self.graph is not None):
            self.detect_ff_terms()
            self.compute_force_field_terms()
    def detect_ff_terms(self):
        """ MOF-FF contains force field descriptions for three different
        inorganic SBUs:
        Cu-paddle-wheel
        Zn cluster --> IRMOF series
        Zr cluster --> UiO series
        """

        # for each atom determine the ff type if it is None
        MOF_FF_organics = ["O", "C", "H"]
        # change to cluster detection.
        mof_sbus = set(self.graph.inorganic_sbus.keys())
        MOF_FF_sbus = set(["Cu Paddlewheel", "Zn4O", "Zr_UiO"])
        if not (mof_sbus <= MOF_FF_sbus):
            print("The system cannot be simulated with MOF-FF!")
            sys.exit()

        #Assigning force field type of atoms
        for node, atom in self.graph.nodes_iter2(data=True):
            # check if element not in one of the SBUS
            if atom['element'] == "Cu":
                try:
                    if atom['special_flag'] == 'Cu_pdw':
                        atom['force_field_type'] = "165"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    else:
                        print("ERROR: Cu %i is not assigned to a Cu Paddlewheel! exiting"%(node))
                        sys.exit()
                except KeyError:
                    print("ERROR: Cu %i is not assigned to a Cu Paddlewheel! exiting"%(node))
                    sys.exit()

            elif atom['element'] == "Zn":
                try:
                    if atom['special_flag'] == 'Zn4O':
                        atom['force_field_type'] = "166"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]

                    else:
                        print("ERROR: Zn %i is not assigned to a Zn4O! exiting"%(node))
                        sys.exit()
                except KeyError:
                    print("ERROR: Zn %i is not assigned to a Zn4O! exiting"%(node))
                    sys.exit()


            elif atom['element'] == "Zr":
                try:
                    if atom['special_flag'] == 'Zr_UiO':
                        atom['force_field_type'] = "101"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]

                    else:
                        print("ERROR: Zr %i is not assigned to a Zr_UiO! exiting"%(node))
                        sys.exit()
                except KeyError:
                    print("ERROR: Zr %i is not assigned to a Zr_UiO! exiting"%(node))
                    sys.exit()


            if atom['force_field_type'] is None:
                type_assigned = False
                neighbours = [self.graph.nodes[i] for i in self.graph.neighbors(node)]
                neighbour_elements = [a['element'] for a in neighbours]
                special = False
                if 'special_flag' in atom:
                    special = True
                if (atom['element'] == "O") and special:
                    # Zn4O cases
                    if atom['special_flag'] == "O_z_Zn4O":
                        atom['force_field_type'] = "165"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    elif atom['special_flag'] == "O_c_Zn4O":
                        atom['force_field_type'] = "167"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    # Zr_UiO cases
                    elif atom['special_flag'] == "O_z_Zr_UiO":
                        atom['force_field_type'] = "102"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    elif atom['special_flag'] == "O_h_Zr_UiO":
                        atom['force_field_type'] = "103"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    elif atom['special_flag'] == "O_c_Zr_UiO":
                        atom['force_field_type'] = "106"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    # Cu Paddlewheel case
                    elif (atom['special_flag'] == "O1_Cu_pdw") or (atom['special_flag'] == "O2_Cu_pdw"):
                        atom['force_field_type'] = "167"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    else:
                        print("Oxygen number %i type cannot be detected!"%node)
                        sys.exit()
                elif (atom['element'] == "C") and special:
                    # Zn4O case
                    if atom['special_flag'] == "C_Zn4O":
                        atom['force_field_type'] = "168"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    # Zr_UiO case
                    elif atom['special_flag'] == "C_Zr_UiO":
                        atom['force_field_type'] = "104"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    # Cu Paddlewheel case
                    elif atom['special_flag'] == "C_Cu_pdw":
                        atom['force_field_type'] = "168"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    else:
                        print("Carbon number %i type cannot be detected!"%node)
                        sys.exit()

                elif (atom['element'] == "H") and special:
                    # only UiO case
                    if atom['special_flag'] == "H_o_Zr_UiO":
                        atom['force_field_type'] = "105"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    else:
                        print("Hydrogen number %i type cannot be detected!"%node)
                        sys.exit()

                # currently no oxygens assigned types outside of metal SBUs
                elif (atom['element'] == "O") and not special:
                    print("Oxygen number %i type cannot be detected!"%node)
                    sys.exit()

                elif (atom['element'] == "C") and not special:
                    # all organic SBUs have the same types..
                    if set(neighbour_elements) == set(["C","H"]):
                        atom['force_field_type'] = "2"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    elif set(neighbour_elements) == set(["C"]):
                        atom['force_field_type'] = "2"
                        # check if Zn4O, UiO, or Cu pdw
                        if mof_sbus == set(['Zn4O']):
                            atom['charge']=  0.18   #special charge for C_ph - C_carb
                        elif mof_sbus == set(['Zr_UiO']):
                            atom['charge']=  0.042   #special charge for C_ph - C_carb
                        elif mof_sbus == set(['Cu Paddlewheel']):
                            atom['charge']=  0.15   #special charge for C_ph - C_carb
                    else:
                        print("Carbon number %i type cannot be detected!"%node)
                        sys.exit()

                elif (atom['element'] == "H") and not special:
                    if set(neighbour_elements)<=set(["C"]):
                        atom['force_field_type'] = "5"
                        atom['charge']=MOFFF_atoms[atom['force_field_type']][6]
                    else:
                        print("Hydrogen number %i type cannot be detected!"%node)
                        sys.exit()

#                atom['charge']=0
        # THE REST OF THIS SHOULD BE IN SEPARATE FUNCTIONS AS PER OTHER FF's DESCRIBED HERE
        # TODO(Mohammad): make this easier to read.
        #Assigning force field type of bonds
        for a, b, bond in self.graph.edges_iter2(data=True):
            a_atom = self.graph.nodes[a]
            b_atom = self.graph.nodes[b]
            atom_a_fflabel, atom_b_fflabel = a_atom['force_field_type'], b_atom['force_field_type']
            bond1_fflabel=atom_a_fflabel+"_"+atom_b_fflabel
            bond2_fflabel=atom_b_fflabel+"_"+atom_a_fflabel
            if bond1_fflabel in MOFFF_bonds:
                bond['force_field_type']=bond1_fflabel
            elif bond2_fflabel in MOFFF_bonds:
                bond['force_field_type']=bond2_fflabel
            else:
                print ("%s bond does not exist in FF!"%(bond1_fflabel))
                exit()
        #Assigning force field type of angles
        missing_labels=[]
        for b , data in self.graph.nodes_iter2(data=True):
            try:
                missing_angles=[]
                ang_data = data['angles']
                for (a, c), val in ang_data.items():
                    a_atom = self.graph.nodes[a]
                    b_atom = data
                    c_atom = self.graph.nodes[c]
                    atom_a_fflabel = a_atom['force_field_type']
                    atom_b_fflabel = b_atom['force_field_type']
                    atom_c_fflabel = c_atom['force_field_type']
                    angle1_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel
                    angle2_fflabel=atom_c_fflabel+"_"+atom_b_fflabel+"_"+atom_a_fflabel
                    if (angle1_fflabel=="167_165_167"):
                        val['force_field_type']=angle1_fflabel
                    elif (angle1_fflabel=="103_101_106") or (angle1_fflabel=="106_101_103"):
                        val['force_field_type']="103_101_106"
                    elif (angle1_fflabel=="106_101_106"):
                        val['force_field_type']=angle1_fflabel
                    elif angle1_fflabel in MOFFF_angles:
                        val['force_field_type']=angle1_fflabel
                    elif angle2_fflabel in MOFFF_angles:
                        val['force_field_type']=angle2_fflabel
                    else:
                        missing_angles.append((a,c))
                        missing_labels.append(angle1_fflabel)
                for key in missing_angles:
                    del ang_data[key]
            except KeyError:
                pass

        for ff_label in set(missing_labels):
            print ("%s angle does not exist in FF!"%(ff_label))
        #Assigning force field type of dihedrals
        missing_labels=[]
        for b, c, data in self.graph.edges_iter2(data=True):
            try:
                missing_dihedral=[]
                dihed_data = data['dihedrals']
                for (a, d), val in dihed_data.items():
                    a_atom = self.graph.nodes[a]
                    b_atom = self.graph.nodes[b]
                    c_atom = self.graph.nodes[c]
                    d_atom = self.graph.nodes[d]
                    atom_a_fflabel = a_atom['force_field_type']
                    atom_b_fflabel = b_atom['force_field_type']
                    atom_c_fflabel = c_atom['force_field_type']
                    atom_d_fflabel = d_atom['force_field_type']
                    dihedral1_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel
                    dihedral2_fflabel=atom_d_fflabel+"_"+atom_c_fflabel+"_"+atom_b_fflabel+"_"+atom_a_fflabel

                    if dihedral1_fflabel in MOFFF_dihedrals:
                        val['force_field_type']=dihedral1_fflabel
                    elif dihedral2_fflabel in MOFFF_dihedrals:
                        val['force_field_type']=dihedral2_fflabel
                    else:
                        missing_dihedral.append((a,d))
                        missing_labels.append(dihedral1_fflabel)
                for key in missing_dihedral:
                    del dihed_data[key]
            except KeyError:
                pass
        for ff_label in set(missing_labels):
            print ("%s dihedral does not exist in FF!"%(ff_label))

        #Assigning force field type of impropers
        missing_labels=[]
        for b, data in self.graph.nodes_iter2(data=True):
            try:
                missing_improper=[]
                imp_data = data['impropers']
                for (a, c, d), val in imp_data.items():
                    a_atom = self.graph.nodes[a]
                    b_atom = self.graph.nodes[b]
                    c_atom = self.graph.nodes[c]
                    d_atom = self.graph.nodes[d]
                    atom_a_fflabel = a_atom['force_field_type']
                    atom_b_fflabel = b_atom['force_field_type']
                    atom_c_fflabel = c_atom['force_field_type']
                    atom_d_fflabel = d_atom['force_field_type']
                    improper_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel
                    if improper_fflabel in MOFFF_opbends:
                        val['force_field_type']=improper_fflabel
                    else:
                        missing_improper.append((a,c,d))
                        missing_labels.append(improper_fflabel)
                for key in missing_improper:
                    del imp_data[key]
            except KeyError:
                pass

        for ff_label in set(missing_labels):
            print ("%s improper does not exist in FF!"%(ff_label))

    def bond_term(self, edge):
        """class2 bond
        Es=71.94*Ks*(l-l0)^2[1-2.55(l-l0)+(7/12)*2.55*(l-l0)^2]
        (Allinger et. al. J.Am.Chem.Soc., Vol. 111, No. 23, 1989)

        """
        n1, n2, data = edge
        Ks =  MOFFF_bonds[data['force_field_type']][0]
        l0 =  MOFFF_bonds[data['force_field_type']][1]
        D =   MOFFF_bonds[data['force_field_type']][2] # the value should be in kcal/mol

        if (D!=0):  # in case of coordination bond, MOF-FF used morse potential
           alpha = np.sqrt(Ks*2*71.94/(2.0*D))
           data['potential'] = BondPotential.Morse()
           data['potential'].D = D
           data['potential'].alpha = alpha
           data['potential'].R0 = l0
           return 1

        K2= 71.94*Ks   # mdyne to kcal *(1/2)
        K3= -2.55*K2
        K4= 3.793125*K2
        data['potential'] = BondPotential.Class2()
        data['potential'].K2 = K2
        data['potential'].K3 = K3
        data['potential'].K4 = K4
        data['potential'].R0 = l0
        return 1

    def angle_term(self, angle):
        """class2 angle

        Be careful that the 5and6 order terms are vanished here since they are not implemented in LAMMPS!!
        Etheta = 0.021914*Ktheta*(theta-theta0)^2[1-0.014(theta-theta0)+5.6(10^-5)*(theta-theta0)^2-7.0*(10^-7)*(theta-theta0)^3+9.0*(10^-10)*(theta-theta0)^4]
        (Allinger et. al. J.Am.Chem.Soc., Vol. 111, No. 23, 1989)
        """
        a, b, c, data = angle
        if (data['force_field_type']=="167_165_167"):   ### in the case of square planar coordination of Cu-paddle-wheel, fourier angle must be used
            data['potential'] = AnglePotential.CosinePeriodic()
            data['potential'].C = 100 #  Need to be parameterized!
            data['potential'].B = 1
            data['potential'].n = 4
            return 1
        elif (data['force_field_type']=="106_101_106"):   ### in the case of square planar coordination of Cu-paddle-wheel, fourier angle must be used
            data['potential'] = AnglePotential.CosinePeriodic()
            data['potential'].C = 0 #  Need to be parameterized!
            data['potential'].B = 1
            data['potential'].n = 4
            return 1
        elif (data['force_field_type']=="103_101_106"):   ### in the case of square planar coordination of Cu-paddle-wheel, fourier angle must be used
            data['potential'] = AnglePotential.CosinePeriodic()
            data['potential'].C = 0 #  Need to be parameterized!
            data['potential'].B = 1
            data['potential'].n = 4
            return 1

        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        ab_bond = self.graph[a][b]
        bc_bond = self.graph[b][c]
        atom_a_fflabel = a_data['force_field_type']
        atom_b_fflabel = b_data['force_field_type']
        atom_c_fflabel = c_data['force_field_type']
        ang_ff_tmp = atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel

        Ktheta = MOFFF_angles[data['force_field_type']][0]
        theta0 = MOFFF_angles[data['force_field_type']][1]
        ### BondAngle ###
        baN1 = MOFFF_angles[data['force_field_type']][4]
        baN2 = MOFFF_angles[data['force_field_type']][5]
        ### BondBond ###
        bbM  = MOFFF_angles[data['force_field_type']][6]

        if not (ang_ff_tmp == data['force_field_type']):  # switching the force constants in the case of assigning swaped angle_ff_label
            buf1 = atom_a_fflabel
            atom_a_fflabel = atom_c_fflabel
            atom_c_fflabel = buf1
            buf2 = baN1
            baN1=baN2
            baN2=buf2
        ### assingning the equilibrium distance of each bond from FF
        bond1_label = atom_a_fflabel+"_"+atom_b_fflabel
        bond2_label = atom_b_fflabel+"_"+atom_c_fflabel
        if (bond1_label) in MOFFF_bonds:
            r1 = MOFFF_bonds[bond1_label][1]
        else:
            bond1_label = atom_b_fflabel+"_"+atom_a_fflabel
            r1 = MOFFF_bonds[bond1_label][1]

        if (bond2_label) in MOFFF_bonds:
            r2 = MOFFF_bonds[bond2_label][1]
        else:
            bond2_label = atom_c_fflabel+"_"+atom_b_fflabel
            r2 = MOFFF_bonds[bond2_label][1]
        ### Unit conversion ###
        bbM = bbM *71.94 # (TODO) maybe is wrong!
        baN1 = 2.51118 * baN1 / (DEG2RAD)
        baN2 = 2.51118 * baN2/ (DEG2RAD)
        K2 = 0.021914*Ktheta/(DEG2RAD**2)
        K3 = -0.014*K2/(DEG2RAD**1)
        K4 = 5.6e-5*K2/(DEG2RAD**2)


        data['potential'] = AnglePotential.Class2()
        data['potential'].theta0 = theta0
        data['potential'].K2 = K2
        data['potential'].K3 = K3
        data['potential'].K4 = K4
        data['potential'].bb.M = 0.0 #bbM
        data['potential'].bb.r1 = r1
        data['potential'].bb.r2 = r2
        data['potential'].ba.N1 = 0.0 #baN1
        data['potential'].ba.N2 = 0.0 # baN2
        data['potential'].ba.r1 = r1
        data['potential'].ba.r2 = r2
        return 1

    def dihedral_term(self, dihedral):
        """fourier diherdral

        Ew = (V1/2)(1 + cos w) + (V2/2)(1 - cos 2*w)+(V3/2)(1 + cos 3*w)+(V4/2)(1 + cos 4*w)
        (Allinger et. al. J.Am.Chem.Soc., Vol. 111, No. 23, 1989)

        """
        a,b,c,d, data = dihedral

        kt1 = 0.5 * MOFFF_dihedrals[data['force_field_type']][0]
        kt2 = 0.5 * MOFFF_dihedrals[data['force_field_type']][3]
        kt3 = 0.5 * MOFFF_dihedrals[data['force_field_type']][6]
        kt4 = 0.5 * MOFFF_dihedrals[data['force_field_type']][9]
        n1 = MOFFF_dihedrals[data['force_field_type']][2]
        n2 = MOFFF_dihedrals[data['force_field_type']][5]
        n3 = MOFFF_dihedrals[data['force_field_type']][8]
        n4 = MOFFF_dihedrals[data['force_field_type']][11]
        d1 = -1.0 * MOFFF_dihedrals[data['force_field_type']][1]
        d2 = -1.0 * MOFFF_dihedrals[data['force_field_type']][4]
        d3 = -1.0 * MOFFF_dihedrals[data['force_field_type']][7]
        d4 = -1.0 * MOFFF_dihedrals[data['force_field_type']][10]

        ki = [kt1,kt2,kt3,kt4]
        ni = [n1,n2,n3,n4]
        di = [d1,d2,d3,d4]

        data['potential'] = DihedralPotential.Fourier()
        data['potential'].Ki = ki
        data['potential'].ni = ni
        data['potential'].di = di
        return 1

    def improper_term(self, improper):
        """Harmonic improper"""
        a,b,c,d, data = improper
        Kopb = MOFFF_opbends[data['force_field_type']][0]/(DEG2RAD**2)*0.02191418
        c0 =  MOFFF_opbends[data['force_field_type']][1]
        data['potential'] = ImproperPotential.Harmonic()
        data['potential'].K = Kopb
        data['potential'].chi0 = c0
        return 1

    def pair_terms(self, node, data, cutoff, **kwargs):
        """
        Buckingham equation in MM3 type is used!

        Also, Table for short range coulombic interactions
        """
        eps = MOFFF_atoms[data['force_field_type']][4]
        sig = MOFFF_atoms[data['force_field_type']][3]

        data['pair_potential'] = PairPotential.Buck()
        data['pair_potential'].cutoff = cutoff
        data['pair_potential'].eps = eps
        data['pair_potential'].sig = sig

        data['tabulated_potential'] = True
        data['table_function'] = self.pair_coul_term
        data['table_potential'] = PairPotential.Table()

    def pair_coul_term(self, node1, node2, data):
        """MOF-FF uses a damped gaussian for close-range interactions.
        This is added in a tabulated form.

        k = 332.063711 kcal*angstroms/e^2

        E_ij = k*qi*qj*erf(r_ij/sig_ij)/r_ij

        --- From Wolfram Alpha ---
        F_ij = - (K*qi*qj) * (2/(pi^(.5) * sig_ij)) * [ e^(-r_ij^2/sig_ij^2) / r_ij - erf(r_ij/sig_ij)/r_ij^2 ]
        ---

        N is set to 1000 by default

        """
        str = ""
        # kcal/mol energy units assumed...
        K = 332.063711

        n = 5000
        rlow=0.01
        R = np.linspace(rlow, self.cutoff+2., n)
        ff1 = node1['force_field_type']
        ff2 = node2['force_field_type']

        qi = node1['charge']
        qj = node2['charge']
        sigi = MOFFF_atoms[ff1][7]
        sigj = MOFFF_atoms[ff2][7]
        sigij = math.sqrt(sigi**2+sigj**2)
        E_coeff = K*qi*qj
        str += "# damped coulomb potential for %s - %s\n"%(ff1, ff2)
        str += "GAUSS_%s_%s\n"%(ff1, ff2)
        str += "N %i R %.2f %.f\n\n"%(n, rlow, self.cutoff+2.)
        data['table_potential'].style = 'linear'
        data['table_potential'].N = n
        data['table_potential'].keyword = 'ewald'
        data['table_potential'].cutoff = self.cutoff
        data['table_potential'].entry = "GAUSS_%s_%s"%(ff1, ff2)
        data['table_potential'].N = n

        for i, r in enumerate(R):
            rsq = r**2
            e = E_coeff*math.erf(r/sigij)/r
            f = -E_coeff  * (math.exp(-(rsq)/(sigij**2))/ r* 2/(math.sqrt(math.pi) * sigij) - math.erf(r/sigij)/(rsq))
            str += "%i %.3f %f %f\n"%(i+1, r, e, f)
        return(str)

    def special_commands(self):
        st = ["%-15s %s"%("pair_modify", "tail yes"),
              "%-15s %s"%("special_bonds", "lj 0 0 1 coul 1 1 1 #!!! Note: Gaussian charges have to be used!"),
              "%-15s %.1f"%("dielectric", 1.0)]
        return st

class FMOFCu(ForceField):
    def __init__(self, **kwargs):
        self.pair_in_data = False
        self.keep_metal_geometry = False
        self.graph = None
        # override existing arguments with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        if (self.graph is not None):
            self.detect_ff_terms()
            self.compute_force_field_terms()

    def insert_graph(self, graph):
        self.graph = graph
        self.detect_ff_terms()
        self.compute_force_field_terms()

    def detect_ff_terms(self):
        # for each atom determine the ff type if it is None
        FMOFCu_organics = [ "O", "C","H","F" ]
        FMOFCu_metals = ["Cu"]
        for node, atom in self.graph.nodes_iter2(data=True):
            flag_coordination=False
            if atom['force_field_type'] is None:
                type_assigned=False
                neighbours = [self.graph.nodes[i] for i in self.graph.neighbors(node)]
                neighbour_elements = [a['element'] for a in neighbours]
                if atom['element'] in FMOFCu_organics:
                    if (atom['element'] == "O"):
                        if("H" in neighbour_elements): #O-H
                            atom['force_field_type']="75"
                            atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]
                        elif ("C" in neighbour_elements): # Carboxylate
                            for i in self.graph.neighbors(node):
                                if (self.graph.nodes[i]['element']=="C"):
                                    neighboursofneighbour = self.graph.neighbors(i)
                                    neighboursofneighbour_elements=[self.graph.nodes[at]['element'] for at in neighboursofneighbour]

                            for atom1 in neighboursofneighbour:
                                if (self.graph.nodes[atom1]['element']=="O"):
                                    bondeds=[self.graph.nodes[j] for j in self.graph.neighbors(atom1)]
                                    bondeds_elements=[at['element'] for at in bondeds]
                                    if("H" in bondeds_elements):
                                        flag_coordination=True

                            if (flag_coordination):
                                atom['force_field_type']="180"
                                atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]
                            else:
                                atom['force_field_type']="170"
                                atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]

                        else:
                            print("Oxygen number : %i could not be recognized!"%atom.index)
                            sys.exit()

                    elif (atom['element'] == "H"):
                        if ("O" in neighbour_elements):
                            atom['force_field_type']="24"
                            atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]
                        elif("C" in neighbour_elements):
                            atom['force_field_type']="915"
                            atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]
                        else:
                            print("Hydrogen number : %i could not be recognized!"%atom.index)

                    elif (atom['element'] == "F"):
                        if (set(neighbour_elements) <= set(["C"])):
                            atom['force_field_type']="911"
                            atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]
                        else:
                            print("Flourine number : %i could not be recognized!"%atom.index)

                    elif( atom['element']=="C"):
                        if ("O" in neighbour_elements):
                            atom['force_field_type']="913" # C-acid
                            atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]
                        elif ("H" in neighbour_elements):
                            atom['force_field_type']="912" # C- benzene we should be careful that in this case C in ligand has also bond with H, but not in the FF
                            atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]
                        elif ("F" in neighbour_elements):
                            atom['force_field_type']="101" # C- benzene we should be careful that in this case C in ligand has also bond with H, but not in the FF
                            atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]
                        elif (set(neighbour_elements)<=set(["C"])):
                            for i in self.graph.neighbors(node):
                                neighboursofneighbour = self.graph.neighbors(i)
                                neighboursofneighbour_elements=[self.graph.nodes[at]['element'] for at in neighboursofneighbour]
                                if ("O" in neighboursofneighbour_elements):
                                    atom['force_field_type']="902"
                                    atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]
                                    type_assigned=True

                            if (type_assigned==False) and (atom['hybridization']=="aromatic"):
                                atom['force_field_type']="903"
                                atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]
                            elif (type_assigned==False) and (atom['hybridization']=="sp3"):
                                atom['force_field_type']="901"
                                atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]
                            elif (type_assigned==False):
                                print("Carbon number : %i could not be recognized! erorr1 %s "%(atom.index, atom.hybridization))

                        else:
                            print("Carbon number : %i could not be recognized! error2"%atom.index)

                elif atom['element'] in FMOFCu_metals:
                    if (atom['element'] == "Cu"):
                        atom['force_field_type']="185"
                        atom['charge']=FMOFCu_atoms[atom['force_field_type']][6]
                else:
                        print('Error!! Cannot detect atom types. Atom type does not exist in FMOFCu-FF!')

               # atom.charge=0
            else:
                print('FFtype is already assigned!')

        """ """  """  """ """
        Assigning force field type of bonds
        """ """  """  """ """
        for a, b, bond in self.graph.edges_iter2(data=True):
            a_atom = self.graph.nodes[a]
            b_atom = self.graph.nodes[b]
            atom_a_fflabel, atom_b_fflabel = a_atom['force_field_type'], b_atom['force_field_type']
            bond1_fflabel=atom_a_fflabel+"_"+atom_b_fflabel
            bond2_fflabel=atom_b_fflabel+"_"+atom_a_fflabel
            if bond1_fflabel in FMOFCu_bonds:
                bond['force_field_type']=bond1_fflabel
            elif bond2_fflabel in FMOFCu_bonds:
                bond['force_field_type']=bond2_fflabel
            else:
                print ("%s bond does not exist in FF!"%(bond1_fflabel))
                exit()

        """ """  """  """ """
        Assigning force field type of angles
        """ """  """  """ """
        missing_labels=[]
        for b , data in self.graph.nodes_iter2(data=True):
            # compute and store angle terms
            try:
                missing_angles=[]
                ang_data = data['angles']
                for (a, c), val in ang_data.items():
                    a_atom = self.graph.nodes[a]
                    b_atom = data
                    c_atom = self.graph.nodes[c]
                    atom_a_fflabel = a_atom['force_field_type']
                    atom_b_fflabel = b_atom['force_field_type']
                    atom_c_fflabel = c_atom['force_field_type']
                    angle1_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel
                    angle2_fflabel=atom_c_fflabel+"_"+atom_b_fflabel+"_"+atom_a_fflabel
                    if (angle1_fflabel=="170_185_170"):
                        val['force_field_type']=angle1_fflabel
#                    elif (angle1_fflabel=="103_101_106") or (angle1_fflabel=="106_101_103"):
#                        val['force_field_type']="103_101_106"
#                    elif (angle1_fflabel=="106_101_106"):
#                        val['force_field_type']=angle1_fflabel
                    elif angle1_fflabel in FMOFCu_angles:
                        val['force_field_type']=angle1_fflabel
                    elif angle2_fflabel in FMOFCu_angles:
                        val['force_field_type']=angle2_fflabel
                    else:
                        print("1")
                        missing_angles.append((a,c))
                        missing_labels.append(angle1_fflabel)
                for key in missing_angles:
                    del ang_data[key]
            except KeyError:
                pass

        for ff_label in set(missing_labels):
            print ("%s angle does not exist in FF!"%(ff_label))

        """ """  """  """ """
        Assigning force field type of dihedrals
        """ """  """  """ """
        missing_labels=[]
        for b, c, data in self.graph.edges_iter2(data=True):
            try:
                missing_dihedral=[]
                dihed_data = data['dihedrals']
                for (a, d), val in dihed_data.items():
                    a_atom = self.graph.nodes[a]
                    b_atom = self.graph.nodes[b]
                    c_atom = self.graph.nodes[c]
                    d_atom = self.graph.nodes[d]
                    atom_a_fflabel = a_atom['force_field_type']
                    atom_b_fflabel = b_atom['force_field_type']
                    atom_c_fflabel = c_atom['force_field_type']
                    atom_d_fflabel = d_atom['force_field_type']
                    dihedral1_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel
                    dihedral2_fflabel=atom_d_fflabel+"_"+atom_c_fflabel+"_"+atom_b_fflabel+"_"+atom_a_fflabel

                    if dihedral1_fflabel in FMOFCu_dihedrals:
                        val['force_field_type']=dihedral1_fflabel
                    elif dihedral2_fflabel in FMOFCu_dihedrals:
                        val['force_field_type']=dihedral2_fflabel
                    else:
                        missing_dihedral.append((a,d))
                        missing_labels.append(dihedral1_fflabel)
                for key in missing_dihedral:
                    del dihed_data[key]
            except KeyError:
                pass
        for ff_label in set(missing_labels):
            print ("%s dihedral does not exist in FF!"%(ff_label))

        """ """  """  """ """
        Assigning force field type of impropers
        """ """  """  """ """
        missing_labels=[]
        for b, data in self.graph.nodes_iter2(data=True):
            try:
                missing_improper=[]
                imp_data = data['impropers']
                for (a, c, d), val in imp_data.items():
                    a_atom = self.graph.nodes[a]
                    b_atom = self.graph.nodes[b]
                    c_atom = self.graph.nodes[c]
                    d_atom = self.graph.nodes[d]
                    atom_a_fflabel = a_atom['force_field_type']
                    atom_b_fflabel = b_atom['force_field_type']
                    atom_c_fflabel = c_atom['force_field_type']
                    atom_d_fflabel = d_atom['force_field_type']
                    improper_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel
                    if improper_fflabel in FMOFCu_opbends:
                        val['force_field_type']=improper_fflabel
                    else:
                        missing_improper.append((a,c,d))
                        missing_labels.append(improper_fflabel)
                for key in missing_improper:
                    del imp_data[key]
            except KeyError:
                pass

        for ff_label in set(missing_labels):
            print ("%s improper does not exist in FF!"%(ff_label))

    def bond_term(self, edge):
        """class2 bond"""
        """
        Es=71.94*Ks*(l-l0)^2[1-2.55(l-l0)+(7/12)*2.55*(l-l0)^2]
        (Allinger et. al. J.Am.Chem.Soc., Vol. 111, No. 23, 1989)
        """
        n1, n2, data = edge
        Ks =  FMOFCu_bonds[data['force_field_type']][0]
        l0 =  FMOFCu_bonds[data['force_field_type']][1]
        K2= 71.94*Ks   # mdyne to kcal *(1/2)
        K3= -2.55*K2
        K4= 3.793125*K2
        data['potential'] = BondPotential.Class2()
        data['potential'].K2 = K2
        data['potential'].K3 = K3
        data['potential'].K4 = K4
        data['potential'].R0 = l0
        return 1

    def angle_term(self, angle):
        """class2 angle"""
        """
        Be careful that the 5and6 order terms are vanished here since they are not implemented in LAMMPS!!
        Etheta = 0.021914*Ktheta*(theta-theta0)^2[1-0.014(theta-theta0)+5.6(10^-5)*(theta-theta0)^2-7.0*(10^-7)*(theta-theta0)^3+9.0*(10^-10)*(theta-theta0)^4]
        (Allinger et. al. J.Am.Chem.Soc., Vol. 111, No. 23, 1989)
        """
        a, b, c, data = angle

        if (data['force_field_type']=="170_185_170"):   ### in the case of square planar coordination of Cu-paddle-wheel, fourier angle must be used
            data['potential'] = AnglePotential.CosinePeriodic()
            data['potential'].C = 100 #  Need to be parameterized!
            data['potential'].B = 1
            data['potential'].n = 4
            return 1

        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        ab_bond = self.graph[a][b]
        bc_bond = self.graph[b][c]
        atom_a_fflabel = a_data['force_field_type']
        atom_b_fflabel = b_data['force_field_type']
        atom_c_fflabel = c_data['force_field_type']
        ang_ff_tmp = atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel

        Ktheta = FMOFCu_angles[data['force_field_type']][0]
        theta0 = FMOFCu_angles[data['force_field_type']][1]
        ### BondAngle ###
        baN1 = FMOFCu_angles[data['force_field_type']][4]
        baN2 = FMOFCu_angles[data['force_field_type']][5]
        ### BondBond ###
        bbM  = FMOFCu_angles[data['force_field_type']][6]

        if not (ang_ff_tmp == data['force_field_type']):  # switching the force constants in the case of assigning swaped angle_ff_label
            buf1 = atom_a_fflabel
            atom_a_fflabel = atom_c_fflabel
            atom_c_fflabel = buf1
            buf2 = baN1
            baN1=baN2
            baN2=buf2
        ### assingning the equilibrium distance of each bond from FF
        bond1_label = atom_a_fflabel+"_"+atom_b_fflabel
        bond2_label = atom_b_fflabel+"_"+atom_c_fflabel
        if (bond1_label) in FMOFCu_bonds:
            r1 = FMOFCu_bonds[bond1_label][1]
        else:
            bond1_label = atom_b_fflabel+"_"+atom_a_fflabel
            r1 = FMOFCu_bonds[bond1_label][1]

        if (bond2_label) in FMOFCu_bonds:
            r2 = FMOFCu_bonds[bond2_label][1]
        else:
            bond2_label = atom_c_fflabel+"_"+atom_b_fflabel
            r2 = FMOFCu_bonds[bond2_label][1]
        ### Unit conversion ###
        bbM = bbM *71.94 # (TODO) maybe is wrong!
        baN1 = 2.51118 * baN1 / (DEG2RAD)
        baN2 = 2.51118 * baN2/ (DEG2RAD)
        K2 = 0.021914*Ktheta/(DEG2RAD**2)
        K3 = -0.014*K2/(DEG2RAD**1)
        K4 = 5.6e-5*K2/(DEG2RAD**2)


        data['potential'] = AnglePotential.Class2()
        data['potential'].theta0 = theta0
        data['potential'].K2 = K2
        data['potential'].K3 = K3
        data['potential'].K4 = K4
        data['potential'].ba.N1 = baN1
        data['potential'].ba.N2 = baN2
        data['potential'].ba.r1 = r1
        data['potential'].ba.r2 = r2
        return 1

    def dihedral_term(self, dihedral):
        """fourier diherdral"""
        """
        Ew = (V1/2)(1 + cos w) + (V2/2)(1 - cos 2*w)+(V3/2)(1 + cos 3*w)+(V4/2)(1 + cos 4*w)
        (Allinger et. al. J.Am.Chem.Soc., Vol. 111, No. 23, 1989)
        """
        a,b,c,d, data = dihedral

        kt1 = 0.5 * FMOFCu_dihedrals[data['force_field_type']][0]
        kt2 = 0.5 * FMOFCu_dihedrals[data['force_field_type']][3]
        kt3 = 0.5 * FMOFCu_dihedrals[data['force_field_type']][6]
        kt4 = 0.5 * FMOFCu_dihedrals[data['force_field_type']][9]
        n1 = FMOFCu_dihedrals[data['force_field_type']][2]
        n2 = FMOFCu_dihedrals[data['force_field_type']][5]
        n3 = FMOFCu_dihedrals[data['force_field_type']][8]
        n4 = FMOFCu_dihedrals[data['force_field_type']][11]
        d1 = -1.0 * FMOFCu_dihedrals[data['force_field_type']][1]
        d2 = -1.0 * FMOFCu_dihedrals[data['force_field_type']][4]
        d3 = -1.0 * FMOFCu_dihedrals[data['force_field_type']][7]
        d4 = -1.0 * FMOFCu_dihedrals[data['force_field_type']][10]

        ki = [kt1,kt2,kt3,kt4]
        ni = [n1,n2,n3,n4]
        di = [d1,d2,d3,d4]

        data['potential'] = DihedralPotential.Fourier()
        data['potential'].Ki = ki
        data['potential'].ni = ni
        data['potential'].di = di
        return 1


    def improper_term(self, improper):
        """class2 diherdral"""
        a,b,c,d, data = improper
        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        d_data = self.graph.nodes[d]
        atom_a_fflabel=a_data['force_field_type']
        atom_b_fflabel=b_data['force_field_type']
        atom_c_fflabel=c_data['force_field_type']
        atom_d_fflabel=d_data['force_field_type']
        Kopb = FMOFCu_opbends[data['force_field_type']][0]/(DEG2RAD**2)*0.02191418
        c0 =  FMOFCu_opbends[data['force_field_type']][1]
        """
        Angle-Angle term
        """
        M1 = FMOFCu_opbends[data['force_field_type']][2]/(DEG2RAD**2)*0.02191418*(-1)/3.  # Three times counting one angle-angle interaction
        M2 = FMOFCu_opbends[data['force_field_type']][3]/(DEG2RAD**2)*0.02191418*(-1)/3.  # Three times counting one angle-angle interaction
        M3 = FMOFCu_opbends[data['force_field_type']][4]/(DEG2RAD**2)*0.02191418*(-1)/3.  # Three times counting one angle-angle interaction
        ang1_ff_label = atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel
        ang2_ff_label = atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_d_fflabel
        ang3_ff_label = atom_c_fflabel+"_"+atom_b_fflabel+"_"+atom_d_fflabel
        if (ang1_ff_label) in FMOFCu_angles:
            Theta1 =  FMOFCu_angles[ang1_ff_label][1]
        else:
            ang1_ff_label = atom_c_fflabel+"_"+atom_b_fflabel+"_"+atom_a_fflabel
            Theta1 =  FMOFCu_angles[ang1_ff_label][1]
        if (ang2_ff_label) in FMOFCu_angles:
            Theta2 =  FMOFCu_angles[ang2_ff_label][1]
        else:
            ang2_ff_label = atom_d_fflabel+"_"+atom_b_fflabel+"_"+atom_a_fflabel
            Theta2 =  FMOFCu_angles[ang2_ff_label][1]
        if (ang3_ff_label) in FMOFCu_angles:
            Theta3 =  FMOFCu_angles[ang3_ff_label][1]
        else:
            ang3_ff_label = atom_d_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel
            Theta3 =  FMOFCu_angles[ang3_ff_label][1]

        data['potential'] =  ImproperPotential.Class2() #does not work now!
        data['potential'].K = Kopb
        data['potential'].chi0 = c0
        data['potential'].aa.M1 = M1
        data['potential'].aa.M2 = M2
        data['potential'].aa.M3 = M3
        data['potential'].aa.theta1 = Theta1
        data['potential'].aa.theta2 = Theta2
        data['potential'].aa.theta3 = Theta3
        return 1

    def pair_terms( self, node , data, cutoff, **kwargs):
        """
        Buckingham equation in MM3 type is used!
        """
        eps = FMOFCu_atoms[data['force_field_type']][4]
        sig = FMOFCu_atoms[data['force_field_type']][3]

        data['pair_potential']=PairPotential.BuckCoulLong()
        data['pair_potential'].cutoff= cutoff
        data['pair_potential'].eps = eps
        data['pair_potential'].sig = sig


    def special_commands(self):
        st = ["%-15s %s"%("pair_modify", "tail yes"),
              "%-15s %s"%("special_bonds", "lj/coul 0.0 0.0 1"),
              "%-15s %.2f"%("dielectric", 1.50)]
        return st


class UFF(ForceField):
    """Parameterize the periodic material with the UFF parameters.
    NB: I have come across important information regarding the
    implementation of UFF from the author of MCCCS TOWHEE.
    It can be found here: (as of 05/11/2015)
    http://towhee.sourceforge.net/forcefields/uff.html

    The ammendments mentioned that document are included here
    """

    def __init__(self,  **kwargs):
        self.pair_in_data = True
        self.keep_metal_geometry = False
        self.graph = None
        # override existing arguments with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        if (self.graph is not None):
            self.detect_ff_terms()
            self.compute_force_field_terms()

    def pair_terms(self, node, data, cutoff, charges=True):
        """Add L-J term to atom"""
        if(charges):
            data['pair_potential'] = PairPotential.LjCutCoulLong()
        else:
            data['pair_potential'] = PairPotential.LjCut()
        data['pair_potential'].eps = UFF_DATA[data['force_field_type']][3]
        data['pair_potential'].sig = UFF_DATA[data['force_field_type']][2]*(2**(-1./6.))
        data['pair_potential'].cutoff = cutoff

    def bond_term(self, edge):
        """Harmonic assumed"""
        n1, n2, data = edge
        n1_data, n2_data = self.graph.nodes[n1], self.graph.nodes[n2]
        fflabel1, fflabel2 = n1_data['force_field_type'], n2_data['force_field_type']
        r_1 = UFF_DATA[fflabel1][0]
        r_2 = UFF_DATA[fflabel2][0]
        chi_1 = UFF_DATA[fflabel1][8]
        chi_2 = UFF_DATA[fflabel2][8]

        rbo = -0.1332*(r_1 + r_2)*math.log(data['order'])
        ren = r_1*r_2*(((math.sqrt(chi_1) - math.sqrt(chi_2))**2))/(chi_1*r_1 + chi_2*r_2)
        r0 = (r_1 + r_2 + rbo - ren)
        # The values for K in the UFF paper were set such that in the final
        # harmonic function, they would be divided by '2' to satisfy the
        # form K/2(R-Req)**2
        # in Lammps, the value for K is already assumed to be divided by '2'
        K = 664.12*(UFF_DATA[fflabel1][5]*UFF_DATA[fflabel2][5])/(r0**3) / 2.
        if (self.keep_metal_geometry) and (n1_data['atomic_number'] in METALS
            or n2_data['atomic_number'] in METALS):
            r0 = data['length']
        data['potential'] = BondPotential.Harmonic()
        data['potential'].K = K
        data['potential'].R0 = r0
        return 1

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
        a, b, c, data = angle
        angle_type = self.uff_angle_type(b)
        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        ab_bond = self.graph[a][b]
        bc_bond = self.graph[b][c]

        auff, buff, cuff = a_data['force_field_type'], b_data['force_field_type'], c_data['force_field_type']

        theta0 = UFF_DATA[buff][1]
        # just check if the central node is a metal, then apply a rigid angle term.
        # NB: Functional form may change dynamics, but at this point we will not
        # concern ourselves if the force constants are big.
        if (self.keep_metal_geometry) and (b_data['atomic_number'] in METALS):
            theta0 = self.graph.compute_angle_between(a, b, c)
            # should put this angle in the general - non-linear case
            # unless the angle is 0 or 180 deg - then linear case.
            # here the K value will be scaled by the number of neighbors
            angle_type = "None"
            # note, coefficient might be too strong here, if the metal
            # is octahedral, for example, kappa = ka/16
            if np.allclose(theta0, 180.0, atol=0.1):
                angle_type = 'linear'

        cosT0 = np.cos(theta0*DEG2RAD)
        sinT0 = np.sin(theta0*DEG2RAD)
        c2 = 1.0 / (4.0*sinT0*sinT0)
        c1 = -4.0 * c2 * cosT0
        c0 = c2 * (2.0*cosT0*cosT0 + 1.0)

        za = UFF_DATA[auff][5]
        zc = UFF_DATA[cuff][5]

        r_ab = ab_bond['potential'].R0
        r_bc = bc_bond['potential'].R0
        r_ac = math.sqrt(r_ab*r_ab + r_bc*r_bc - 2.*r_ab*r_bc*cosT0)

        beta = 664.12/r_ab/r_bc
        ka = beta*(za*zc /(r_ac**5.))*r_ab*r_bc
        ka *= (3.*r_ab*r_bc*(1. - cosT0*cosT0) - r_ac*r_ac*cosT0)

        if angle_type in sf or (angle_type == 'tetrahedral' and int(theta0) == 90):
            if angle_type == 'linear':
                kappa = ka
                c0 = -1.
                B  = 1
                c1 = 1.
            # the description of the actual parameters for 'n' are not obvious
            # for the tetrahedral special case from the UFF paper or the write up in TOWHEE.
            # The values were found in the TOWHEE source code (eg. Bi3+3).
            if angle_type == 'tetrahedral':
                kappa = ka/4.
                c0 = 1.
                B  = -1
                c1 = 2.

            if angle_type == 'trigonal-planar':
                kappa = ka/9.
                c0 = -1.
                B  = -1
                c1 = 3.

            if angle_type == 'square-planar' or angle_type == 'octahedral':
                kappa = ka/16.
                c0 = -1.
                B  = 1
                c1 = 4.

            #data['potential'] = AnglePotential.FourierSimple()
            data['potential'] = AnglePotential.CosinePeriodic()
            #data['potential'].K = kappa
            data['potential'].C = kappa*(c1**2)
            #data['potential'].c = c0
            data['potential'].B = B
            data['potential'].n = c1
        # general-nonlinear
        else:

            #TODO: a bunch of special cases which require molecular recognition here..
            # water, for example has it's own theta0 angle.

            kappa = ka
            data['potential'] = AnglePotential.Fourier()
            data['potential'].K = kappa
            data['potential'].C0 = c0
            data['potential'].C1 = c1
            data['potential'].C2 = c2
        return 1

    def uff_angle_type(self, b):
        name = self.graph.nodes[b]['force_field_type']
        try:
            coord_type = name[2]
        except IndexError:
            # eg, H_, F_
            return 'linear'
        if coord_type == "1":
            return 'linear'
        elif coord_type in ["R", "2"]:
            return 'trigonal-planar'
        elif coord_type == "3":
            return 'tetrahedral'
        elif coord_type == "4":
            return 'square-planar'
        elif coord_type == "5":
            return 'trigonal-bipyrimidal'
        elif coord_type == "6":
            return 'octahedral'
        elif coord_type == "8":
            return 'cubic-antiprism'
        else:
            print("ERROR: Cannot find coordination type for %s"%name)
            sys.exit()

    def dihedral_term(self, dihedral):
        """Use a small cosine Fourier expansion

        E_phi = 1/2*V_phi * [1 - cos(n*phi0)*cos(n*phi)]


        this is available in Lammps in the form of a harmonic potential
        E = K * [1 + d*cos(n*phi)]

        NB: the d term must be negated to recover the UFF potential.
        """
        a,b,c,d, data = dihedral
        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        d_data = self.graph.nodes[d]

        torsiontype = self.graph[b][c]['order']

        coord_bc = (self.graph.degree(b), self.graph.degree(c))
        bc = (b_data['force_field_type'], c_data['force_field_type'])
        M = mul(*coord_bc)
        V = 0
        n = 0
        mixed_case = ((b_data['hybridization'] == 'sp2' or b_data['hybridization'] == 'aromatic') and
                      c_data['hybridization'] == 'sp3') or \
                (b_data['hybridization'] == 'sp3' and
                (c_data['hybridization'] == 'sp2' or c_data['hybridization'] == 'aromatic'))
        all_sp2 = ((b_data['hybridization'] == 'sp2' or b_data['hybridization'] == 'aromatic') and
                   c_data['hybridization'] == 'sp2' or c_data['hybridization'] == 'aromatic')
        all_sp3 = (b_data['hybridization'] == 'sp3' and
                   c_data['hybridization'] == 'sp3')

        phi0 = 0
        if (b_data['atomic_number'] in METALS or c_data['atomic_number'] in METALS):
            return None
        if all_sp3:
            phi0 = 60.0
            n = 3
            vi = UFF_DATA[b_data['force_field_type']][6]
            vj = UFF_DATA[c_data['force_field_type']][6]

            if b_data['atomic_number'] == 8:
                vi = 2.
                n = 2
                phi0 = 90.
            elif b_data['atomic_number'] in (16, 34, 52, 84):
                vi = 6.8
                n = 2
                phi0 = 90.0
            if c_data['atomic_number'] == 8:
                vj = 2.
                n = 2
                phi0 = 90.0

            elif c_data['atomic_number'] in (16, 34, 52, 84):
                vj = 6.8
                n = 2
                phi0 = 90.0

            V = (vi*vj)**0.5

        elif all_sp2:
            ui = UFF_DATA[b_data['force_field_type']][7]
            uj = UFF_DATA[c_data['force_field_type']][7]
            phi0 = 180.0
            n = 2
            V = 5.0 * (ui*uj)**0.5 * (1. + 4.18*math.log(torsiontype))

        elif mixed_case:
            phi0 = 180.0
            n = 3
            V = 2.

            if c_data['hybridization'] == 'sp3':
                if c_data['atomic_number'] in (8, 16, 34, 52):
                    n = 2
                    phi0 = 90.
            elif b_data['hybridization'] == 'sp3':
                if b_data['atomic_number'] in (8, 16, 34, 52):
                    n = 2
                    phi0 = 90.0
            # special case group 6 elements
            if n==2:
                ui = UFF_DATA[b_data['force_field_type']][7]
                uj = UFF_DATA[c_data['force_field_type']][7]
                V = 5.0 * (ui*uj)**0.5 * (1. + 4.18*math.log(torsiontype))

        V /= float(M)
        nphi0 = n*phi0

        if abs(math.sin(nphi0*DEG2RAD)) > 1.0e-3:
            print("WARNING!!! nphi0 = %r" % nphi0)

        if (self.keep_metal_geometry) and (b_data['atomic_number'] in METALS or
            c_data['atomic_number'] in METALS):
            # must use different potential with minimum at the computed dihedral
            # angle.
            #nphi0 = n*self.graph.compute_dihedral_between(a, b, c, d)
            #data['potential'] = DihedralPotential.Charmm()
            #data['potential'].K = 0.5
            #data['potential'].d = 180 + nphi0
            #data['potential'].n = n
            # NO torsional terms in UFF for non-main group elements
            return None
        if V==0.:
            return None
        data['potential'] = DihedralPotential.Harmonic()
        data['potential'].K = 0.5*V
        data['potential'].d = -math.cos(nphi0*DEG2RAD)
        data['potential'].n = n
        return 1

    def improper_term(self, improper):
        """
        The improper function can be described with a fourier function

        E = K*[C_0 + C_1*cos(w) + C_2*cos(2*w)]

        NB: not sure if keep metal geometry is important here.
        """
        a, b, c, d, data = improper
        b_data = self.graph.nodes[b]
        a_ff = self.graph.nodes[a]['force_field_type']
        c_ff = self.graph.nodes[c]['force_field_type']
        d_ff = self.graph.nodes[d]['force_field_type']
        if not b_data['atomic_number'] in (6, 7, 8, 15, 33, 51, 83):
            return None
        if b_data['force_field_type'] in ('N_3', 'N_2', 'N_R', 'O_2', 'O_R'):
            c0 = 1.0
            c1 = -1.0
            c2 = 0.0
            koop = 6.0
        elif b_data['force_field_type'] in ('P_3+3', 'As3+3', 'Sb3+3', 'Bi3+3'):
            if b_data['force_field_type'] == 'P_3+3':
                phi = 84.4339 * DEG2RAD
            elif b_data['force_field_type'] == 'As3+3':
                phi = 86.9735 * DEG2RAD
            elif b_data['force_field_type'] == 'Sb3+3':
                phi = 87.7047 * DEG2RAD
            else:
                phi = 90.0 * DEG2RAD
            c1 = -4.0 * math.cos(phi)
            c2 = 1.0
            c0 = -1.0*c1*math.cos(phi) + c2*math.cos(2.0*phi)
            koop = 22.0
        elif b_data['force_field_type'] in ('C_2', 'C_R'):
            c0 = 1.0
            c1 = -1.0
            c2 = 0.0
            koop = 6.0
            if 'O_2' in (a_ff, c_ff, d_ff):
                # check to make sure an aldehyde (i.e. not carboxylate bonded to metal)
                if a_ff == "O_2" and self.graph.degree(a) == 1:
                    koop = 50.0
                elif c_ff == "O_2" and self.graph.degree(c) == 1:
                    koop = 50.0
                elif d_ff == "O_2" and self.graph.degree(d) == 1:
                    koop = 50.0
        else:
            return None

        koop /= 3.

        data['potential'] = ImproperPotential.Fourier()
        data['potential'].K = koop
        data['potential'].C0 = c0
        data['potential'].C1 = c1
        data['potential'].C2 = c2
        return 1

    def special_commands(self):
        st = ["%-15s %s %s"%("pair_modify", "tail yes", "mix arithmetic"),
              "%-15s %s"%("special_bonds", "lj/coul 0.0 0.0 1.0"),
              "%-15s %.1f"%('dielectric', 1.0)
              ]
        return st

    def detect_ff_terms(self):
        # for each atom determine the ff type if it is None
        organics = ["C", "N", "O", "S"]
        halides = ["F", "Cl", "Br", "I"]
        sqpl = ["He", "Ne", "Ar", "Ni", "Kr", "Pd", "Xe", "Pt", "Au", "Rn"]

        for node, data in self.graph.nodes_iter2(data=True):
            if data['force_field_type'] is None:
                if data['element'] in organics:
                    if data['hybridization'] == "sp3":
                        data['force_field_type'] = "%s_3"%data['element']
                        if data['element'] == "O" and self.graph.degree(node) >= 2:
                            neigh_elem = set([self.graph.nodes[i]['element'] for i in self.graph.neighbors(node)])
                            if neigh_elem <= metals and self.graph.degree(node) == 2:
                                data['force_field_type'] = "O_2"
                            elif neigh_elem <= metals and self.graph.degree(node) == 3:
                                data['force_field_type'] = "O_3"
                            # zeolites
                            if neigh_elem <= set(["Si", "Al"]):
                                data['force_field_type'] = "O_3_z"
                        elif data['element'] == "S":
                            # default sp3 hybridized sulphur set to S_3+6
                            data['force_field_type'] = "S_3+6"

                    elif data['hybridization'] == "aromatic":
                        data['force_field_type'] = "%s_R"%data['element']
                        # fix to make the angle 120
                        #if data['element'] == "O":
                        #    data['force_field_type'] = "O_2"
                    elif data['hybridization'] == "sp2":
                        data['force_field_type'] = "%s_2"%data['element']
                    elif data['hybridization'] == "sp":
                        data['force_field_type'] = "%s_1"%data['element']
                elif data['element'] == "H":
                    data['force_field_type'] = "H_"
                elif data['element'] in halides:
                    data['force_field_type'] = data['element']
                    if data['element'] == "F":
                        data['force_field_type'] += "_"
                    elif data['element'] == "I":
                        data['force_field_type'] += "_"
                elif data['element'] == "Li":
                    data['force_field_type'] = data['element']
                else:
                    ffs = list(UFF_DATA.keys())
                    valency = self.graph.degree(node)
                    # temp fix for some real geometrical analysis
                    if (valency == 4) and (data['element'] not in sqpl):
                        valency = 3
                    for j in ffs:
                        if data['element'] == j[:2].strip("_"):
                            try:
                                if valency == int(j[2]):
                                    data['force_field_type'] = j
                            except IndexError:
                                # no valency for this atom
                                data['force_field_type'] = j

            if data['force_field_type'] is None:
                assigned = False
                # find the first entry that corresponds to this element and print a warning
                for j in list(UFF_DATA.keys()):
                    if data['element'] == j[:2].strip("_"):
                        data['force_field_type'] = j
                        neigh = self.graph.degree(node)
                        print("WARNING: Atom %i element "%data['index'] +
                                "%s has %i neighbors, "%(data['element'], neigh)+
                                "but was assigned %s as a force field type!"%(j))
                        assigned = True

                if not assigned:
                    print("ERROR: could not find the proper force field type for atom %i"%(data['index'])+
                            " with element: '%s'"%(data['element']))
                    sys.exit()

class Dreiding(ForceField):

    def __init__(self, graph=None,  h_bonding=False, **kwargs):
        self.pair_in_data = True
        self.h_bonding = h_bonding
        self.bondtype = 'harmonic'
        self.keep_metal_geometry = False
        # override existing arguments with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        if (graph is not None):
            self.graph = graph
            self.detect_ff_terms()
            self.compute_force_field_terms()

    def bond_term(self, edge):
        """The DREIDING Force Field contains two possible bond terms, harmonic and Morse.
        The authors recommend using harmonic as a default, and Morse potentials for more
        'refined' calculations.
        Here we will assume a harmonic term by default, then the user can chose to switch
        to Morse if they so choose. (change type argument to 'morse')

        E = 0.5 * K * (R - Req)^2


        E = D * [exp{-(alpha*R - Req)} - 1]^2



        """
        n1, n2, data = edge

        n1_data, n2_data = self.graph.nodes[n1], self.graph.nodes[n2]
        fflabel1, fflabel2 = n1_data['force_field_type'], n2_data['force_field_type']
        R1 = DREIDING_DATA[fflabel1][0]
        R2 = DREIDING_DATA[fflabel2][0]
        order = data['order']
        K = order*700.
        D = order*70.
        Re = R1 + R2 - 0.01

        if (self.keep_metal_geometry) and (n1_data['atomic_number'] in METALS
            or n2_data['atomic_number'] in METALS):

            if self.bondtype.lower() == "harmonic":
                data['potential'] = BondPotential.Harmonic()
                data['potential'].K = K/2.
                data['potential'].R0 = data['length']

            elif self.bondtype.lower() == "morse":
                alpha = order * np.sqrt(K/2./D)
                data['potential'] = BondPotential.Morse()
                data['potential'].D = D
                data['potential'].alpha = alpha
                data['potential'].R = data['length']
            return 1

        if self.bondtype.lower() == 'harmonic':
            data['potential'] = BondPotential.Harmonic()
            data['potential'].K = K/2.
            data['potential'].R0 = Re

        elif self.bondtype.lower() == 'morse':
            alpha = order * np.sqrt(K/2./D)
            data['potential'] = BondPotential.Morse()
            data['potential'].D = D
            data['potential'].alpha = alpha
            data['potential'].R0 = Re

        else:
            print("ERROR: Cannot recognize bond potential for Dreiding: %s"%self.bondtype)
            print("Please chose between 'morse' or 'harmonic'")
            sys.exit()
        return 1

    def angle_term(self, angle):
        """
        Harmonic cosine angle

        E = 0.5*C*[cos(theta) - cos(theta0)]^2

        This is available in LAMMPS as the cosine/squared angle style
        (NB. the prefactor includes the usual 1/2 term.)

        if theta0 == 180, use

        E = K*(1 + cos(theta))

        This is available in LAMMPS as the cosine angle style

        """
        a, b, c, data = angle
        K = 100.0
        a_data, b_data, c_data = self.graph.nodes[a], self.graph.nodes[b], self.graph.nodes[c]
        btype = b_data['force_field_type']
        theta0 = DREIDING_DATA[btype][1]

        if (self.keep_metal_geometry) and (b_data['atomic_number'] in METALS):
            theta0 = self.graph.compute_angle_between(a, b, c)
            data['potential'] = AnglePotential.CosineSquared()
            K = 0.5*K/(np.sin(theta0*DEG2RAD))**2
            data['potential'].K = K
            data['potential'].theta0 = theta0
            return 1

        if (theta0 == 180.):
            data['potential'] = AnglePotential.Cosine()
            data['potential'].K = K / 2.
            # Dreiding suggests setting K = 100.0, but the resulting maxima in the cosine function
            # wind up as 200.0 due to the functional form. I have divided by two in case this is an
            # error in the intended Dreiding force field.
            #data['potential'].K = K

        elif (theta0 == 90.0):
            # this is for the Cu paddlehweel. This function has minima at 90 and 180 deg
            # and is an addition made by Peter Boyd.
            # NB: the slope around the minima is similar compared with the CosineSquared function,
            #     however, the energetic barrier is much lower (50 kcal vs ~ 12). There
            #     may be some unphysical effects of this parameterization
            data['potential'] = AnglePotential.CosinePeriodic()
            n = 4 # makes four minima in the angle range 0 - 360
            data['potential'].C = K #/n**2
            data['potential'].B = 1.
            data['potential'].n = n

        else:
            #data['potential'] = AnglePotential.Harmonic()
            data['potential'] = AnglePotential.CosineSquared()
            K = 0.5*K/(np.sin(theta0*DEG2RAD))**2
            data['potential'].K = K
            data['potential'].theta0 = theta0
        return 1

    def dihedral_term(self, dihedral):
        """

        The DREIDING dihedral is of the form

        E = 0.5*V*[1 - cos(n*(phi - phi0))]

        LAMMPS has a similar potential 'charmm' which is described as

        E = K*[1 + cos(n*phi - d)]

        In this case the 'd' term must be multiplied by 'n' before
        inputting to lammps. In addition a +180 degrees out-of-phase
        shift must be added to 'd' to ensure that the potential behaves
        the same as the DREIDING article intended.
        """

        monovalent = ["Na", "F_", "Cl", "Br", "I_", "K"]
        a,b,c,d, data = dihedral
        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        d_data = self.graph.nodes[d]

        btype = b_data['force_field_type']
        ctype = c_data['force_field_type']

        order = self.graph[b][c]['order']
        a_hyb = a_data['hybridization']
        b_hyb = b_data['hybridization']
        c_hyb = c_data['hybridization']
        d_hyb = d_data['hybridization']

        # special cases associated with oxygen column, listed here
        oxygen_sp3 = ["O_3", "S_3", "Se3", "Te3"]
        non_oxygen_sp2 = ["C_R", "N_R", "B_2", "C_2", "N_2"]

        sp2 = ["aromatic", "sp2"]
        # default is to include the full 1-4 non-bonded interactions.
        # but this breaks Lammps unless extra work-arounds are in place.
        # the weighting is added via a special_bonds keyword
        #w = 1.0
        w = 0.0
        #for ring in d_data['rings']:
        #    if a in ring and len(ring) == 6:
        #        w = 0.5
        #    if a in ring and len(ring) < 6:
        #        w = 0.0 # otherwise 1-2 and 1-3 non-bonded interactions would be counted.

        # g)
        if (b_hyb == 'sp' or c_hyb == 'sp') or (btype in monovalent or ctype in monovalent) or \
                (b_data['atomic_number'] in METALS or c_data['atomic_number'] in METALS):
            V = 0.0
            n = 2
            phi0 = 180.0
            return None
        # a)
        elif((b_hyb == "sp3")and(c_hyb == "sp3")):
            V = 2.0
            n = 3
            phi0 = 180.0
            # h) special case..
            if((btype in oxygen_sp3) and (ctype in oxygen_sp3)):
                V = 2.0
                n = 2
                phi0 = 90.0

        # b)
        elif(((b_hyb in sp2) and (c_hyb == "sp3"))or(
            (b_hyb == "sp3") and (c_hyb in sp2))):
            V = 1.0
            n = 6
            phi0 = 0.0
            # i) special case..
            if(((btype in oxygen_sp3)and(ctype in non_oxygen_sp2)) or
                    (ctype in oxygen_sp3)and(btype in non_oxygen_sp2)):
                V = 2.0
                n = 2
                phi0 = 180.0
            # j) special case..

            if(((b_hyb in sp2) and (a_hyb not in sp2))or(
                (c_hyb in sp2) and (d_hyb not in sp2))):
                V = 2.0
                n = 3
                phi0 = 180.0

        # c)
        elif((b_hyb in sp2) and (c_hyb in sp2) and (order == 2.)):
            V = 45.0
            n = 2
            phi0 = 180.0

        # d)
        elif((b_hyb in sp2) and (c_hyb in sp2) and (order >= 1.5)):
            V = 25.0
            n = 2
            phi0 = 180.0

        # e)
        elif((b_hyb in sp2) and (c_hyb in sp2) and (order == 1.0)):
            V = 5.0
            #V = 25.0 # temp fix aromatic...
            n = 2
            phi0 = 180.0
            # f) just check if neighbours are aromatic, then apply the exception
            # NB: this may fail for phenyl esters if the oxygen atoms are not
            # labelled as "R" (i.e. will fail if they are O_2 or O_3)
            if(b_hyb == "aromatic" and c_hyb == "aromatic"):
                b_arom = True
                for cycle in b_data['rings']:
                    # Need to make sure this isn't part of the same ring.
                    if c in cycle:
                        b_arom = False
                        print("WARNING: two resonant atoms "+
                              "%s and %s"%(b_data['ciflabel'], c_data['ciflabel'])+
                              "in the same ring have a bond order of 1.0! "
                              "This will likely yield unphysical characteristics"+
                              " of your system.")


                c_arom = True
                for cycle in c_data['rings']:
                    # Need to make sure this isn't part of the same ring.
                    if b in cycle:
                        c_arom = False
                        print("WARNING: two resonant atoms "+
                              "%s and %s"%(b_data['ciflabel'], c_data['ciflabel'])+
                              "in the same ring have a bond order of 1.0! "
                              "This will likely yield unphysical characteristics"+
                              " of your system.")
                if (b_arom and c_arom):
                    V *= 2.0

        # divide V by the number of dihedral angles
        # to compute across this a-b bond
        b_neigh = self.graph.degree(b) - 1
        c_neigh = self.graph.degree(c) - 1
        norm = float(b_neigh * c_neigh)
        V /= norm
        d = n*phi0 + 180
        data['potential'] = DihedralPotential.Charmm()
        data['potential'].K = V/2.
        data['potential'].n = n
        data['potential'].d = d
        data['potential'].w = w
        return 1

    def improper_term(self, improper):
        """Dreiding improper term.
        ::

                  b                        J
                 /                        /
                /                        /
          c----a     , DREIDING =  K----I
                \                        \ 
                 \                        \ 
                  d                        L
        
        For all non-planar configurations, DREIDING uses::

            E = 0.5*C*(cos(phi) - cos(phi0))^2

        For systems with planar equilibrium geometries, phi0 = 0 ::

            E = K*[1 - cos(phi)].

        This is available in LAMMPS as the 'umbrella' improper potential.
        """
        a, b, c, d, data = improper

        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        d_data = self.graph.nodes[d]

        btype = b_data['force_field_type']
        # special case: ignore N column
        sp3_N = ["N_3", "P_3", "As3", "Sb3"]
        K = 40.0
        if b_data['hybridization'] == 'sp2' or b_data['hybridization'] == 'aromatic':
            K /= 3.
        if btype in sp3_N:
            return None
        omega0 = DREIDING_DATA[btype][4]
        data['potential'] = ImproperPotential.Umbrella()

        data['potential'].K = K
        data['potential'].omega0 = omega0
        return 1

    def pair_terms(self, node, data, cutoff, nbpot='LJ', hbpot='morse', charges=True):
        """ DREIDING can adopt the exponential-6 or
        Ex6 = A*exp{-C*R} - B*R^{-6}

        the Lennard-Jones type interactions.
        Elj = A*R^{-12} - B*R^{-6}

        This will eventually be user-defined

        """
        eps = DREIDING_DATA[data['force_field_type']][3]
        R = DREIDING_DATA[data['force_field_type']][2]
        sig = R*(2**(-1./6.))

        if nbpot == "LJ":
            if(charges):
                data['pair_potential'] = PairPotential.LjCutCoulLong()
            else:
                data['pair_potential'] = PairPotential.LjCut()
            #data['pair_potential'] = PairPotential.LjCharmmCoulLong()
            data['pair_potential'].eps = eps
            data['pair_potential'].sig = sig
            #data['pair_potential'].eps14 = eps
            #data['pair_potential'].sig14 = sig

        else:
            S = DREIDING_DATA[data['force_field_type']][5]

            A = eps*(6./(S - 6.))*np.exp(S)
            rho = R
            C = eps*(S/(S - 6.)*R**6)

            data['pair_potential'] = PairPotential.BuckLongCoulLong()
            data['pair_potential'].A = A
            data['pair_potential'].C = C
            data['pair_potential'].rho = rho

        data['pair_potential'].cutoff = cutoff
        if data['h_bond_donor']:
            for n in self.graph.neighbors(node):
                if self.graph.nodes[n]['force_field_type'] == "H__HB":
                    data['h_bond_function'] = self.hbond_pot(node, hbpot, n)
                    break

    def hbond_pot(self, node, nbpot, hnode):
        """
        DREIDING can describe hbonded donor and acceptors
        using a lj function or a morse potential

        the morse potential is apparently better, so it
        will be default here

        DREIDING III h-bonding terms
        specified in 10.1021/ja8100227.

        Table S3 of the SI of 10.1021/ja8100227 is poorly documented,
        I have parameterized, to the best of my ability, what was intended
        in that paper. This message posted on the lammps-users
        message board http://lammps.sandia.gov/threads/msg36158.html
        was helpful
        N_3H == tertiary amine
        N_3P == primary amine
        N_3HP == protonated primary amine

        nb. need connectivity information - this is accessed by self.graph
        """

        if (nbpot == 'morse'):
            pass
        elif (nbpot == 'lj'):
            # not yet implemented
            pass
        def hbond_pair(node2, graph, flipped=False):
            potential = PairPotential.HbondDreidingMorse()
            data = graph.nodes[node]
            data2 = graph.nodes[node2]
            if(flipped):
                potential.donor = 'j'
                data1 = data2.copy()
                data2 = data.copy()
                node1 = node2
                node2 = node
            else:
                data1 = data
                node1 = node
            ff1 = data1['force_field_type']
            ff2 = data2['force_field_type']
            # generic HB
            D0 = 9.5
            R0 = 2.75
            ineigh = [graph.nodes[q]['element'] for q in graph.neighbors(node1)]
            jneigh = [graph.nodes[q]['element'] for q in graph.neighbors(node2)]
            if(ff1 == "N_3") or (ff1 == "N_R" and ineigh.count("H") == 2):
                # tertiary amine
                if ((ineigh.count("H") < 3) and (len(ineigh) == 4)or
                        ineigh.count("H")<2 and (len(ineigh) == 3)):
                    if(ff2 == "Cl_"):
                        D0 = 3.23
                        R0 = 3.575
                    elif(ff2 == "O_3"):
                        D0 = 1.31
                        R0 = 3.41
                    elif(ff2 == "O_2"):
                        D0 = 1.25
                        R0 = 3.405
                    elif(ff2 == "N_3"):
                        if((jneigh.count("H") > 0)):
                            D0 = 0.93
                            R0 = 3.47
                        else:
                            D0 = 0.1870
                            R0 = 3.90
                # primary amine
                elif((ineigh.count("H") == 2) and (len(ineigh) == 3)):
                    if(ff2 == "Cl_"):
                        D0 = 10.00
                        R0 = 2.9795
                    elif(ff2 == "O_3"):
                        D0 = 2.21
                        R0 = 3.12
                    elif(ff2 == "O_2" or ff2 == "O_R"): # Added the O_R if statement
                        #D0 = 8.38
                        #R0 = 2.77
                        R0 = 3.00
                    elif(ff2 == "N_3"):
                        if((jneigh.count("H") > 0)):
                            D0 = 8.45
                            R0 = 2.84
                        else:
                            D0 = 5.0
                            R0 = 2.765
                # protonated primary amine
                elif((ineigh.count("H") == 3) and (len(ineigh) >= 3)):
                    if(ff2 == "Cl_"):
                        D0 = 7.6
                        R0 = 3.275
                    elif(ff2 == "O_3"):
                        D0 = 1.22
                        R0 = 3.2
                    elif(ff2 == "O_2"):
                        D0 = 8.56
                        R0 = 2.635
                    elif(ff2 == "N_3"):
                        if((jneigh.count("H") > 0)):
                            D0 = 10.14
                            R0 = 2.6
                        else:
                            D0 = 0.8
                            R0 = 3.22
            elif(ff1 == "N_R"):
                if(ff2 == "Cl_"):
                    D0 = 5.6
                    R0 = 3.265
                elif(ff2 == "O_3"):
                    D0 = 1.38
                    R0 = 3.17
                elif(ff2 == "N_R"):
                    R0 = 2.72 # fix for adenine w-c
                elif(ff2 == "O_2"):
                    D0 = 3.88
                    R0 = 2.9
                elif(ff2 == "N_3"):
                    if((jneigh.count("H") > 0)):
                        D0 = 2.44
                        R0 = 3.15
                    else:
                        D0 = 0.43
                        R0 = 3.4
            elif(ff1 == "O_3"):
                if(ff2 == "O_2"):
                    D0 = 1.33
                    R0 = 3.15
                elif(ff2 == "N_3"):
                    if((jneigh.count("H") > 0)):
                        D0 = 1.97
                        R0 = 3.12
                    else:
                        D0 = 1.25
                        R0 = 3.15
            potential.htype = graph.nodes[hnode]['ff_type_index']
            potential.D0 = D0
            potential.alpha = 10.0/ 2. / R0
            potential.R0 = R0
            potential.n = 2
            # one can edit these values for bookkeeping.
            potential.Rin = 9.0
            potential.Rout = 11.0
            potential.a_cut = 30.0
            return potential

        return hbond_pair

    def special_commands(self):
        if self.pair_in_data:
            st = ["%-15s %s %s"%("pair_modify", "tail yes", "mix arithmetic")]
        else:
            st = ["%-15s %s"%("pair_modify", "tail yes")]
        st += ["%-15s %.1f"%('dielectric', 1.0),
               "%-15s %s"%("special_bonds", "dreiding") # equivalent to 'special_bonds lj 0.0 0.0 1.0'
               ]
        return st

    def detect_ff_terms(self):
        # for each atom determine the ff type if it is None
        organics = ["C", "N", "O", "S"]
        halides = ["F", "Cl", "Br", "I"]
        electro_neg_atoms = ["N", "O", "F"]
        for node, data in self.graph.nodes_iter2(data=True):
            if data['force_field_type'] is None or self.h_bonding:
                if data['element'] in organics:
                    if data['hybridization'] == "sp3":
                        data['force_field_type'] = "%s_3"%data['element']
                    elif data['hybridization'] == "aromatic":
                        data['force_field_type'] = "%s_R"%data['element']
                    elif data['hybridization'] == "sp2":
                        data['force_field_type'] = "%s_2"%data['element']
                    elif data['hybridization'] == "sp":
                        data['force_field_type'] = "%s_1"%data['element']
                    else:
                        data['force_field_type'] = "%s_3"%data['element']

                elif data['element'] == "H":
                    data['force_field_type'] = "H_"
                    if self.h_bonding:
                        for n in self.graph.neighbors(node):
                            if self.graph.nodes[n]['element'] in electro_neg_atoms:
                                self.graph.nodes[n]['h_bond_donor'] = True
                                data['force_field_type'] = "H__HB"

                elif data['element'] in halides:
                    data['force_field_type'] = data['element']
                    if data['element'] == "F":
                        data['force_field_type'] += "_"
                    elif data['element'] == "I":
                        data['force_field_type'] += "_"
                else:
                    ffs = list(DREIDING_DATA.keys())
                    for j in ffs:
                        if data['element'] == j[:2].strip("_"):
                            data['force_field_type'] = j
            elif data['force_field_type'] not in DREIDING_DATA.keys():
                print('Error: %s is not a force field type in DREIDING.'%(data['force_field_type']))
                sys.exit()

            if data['force_field_type'] is None:
                print("ERROR: could not find the proper force field type for atom %i"%(data['index'])+
                        " with element: '%s'"%(data['element']))
                sys.exit()

class UFF4MOF(ForceField):
    """Parameterize the periodic material with the UFF4MOF parameters.
    """

    def __init__(self, **kwargs):
        self.pair_in_data = True
        self.keep_metal_geometry = False
        self.graph = None
        # override existing arguments with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        if (self.graph is not None):
            self.detect_ff_terms()
            self.compute_force_field_terms()

    def pair_terms(self, node, data, cutoff, charges=True):
        """Add L-J term to atom"""
        if(charges):
            data['pair_potential'] = PairPotential.LjCutCoulLong()
        else:
            data['pair_potential'] = PairPotential.LjCut()
        data['pair_potential'].eps = UFF4MOF_DATA[data['force_field_type']][3]
        data['pair_potential'].sig = UFF4MOF_DATA[data['force_field_type']][2]*(2**(-1./6.))
        data['pair_potential'].cutoff = cutoff
        return 1

    def bond_term(self, edge):
        """Harmonic assumed"""
        n1, n2, data = edge
        n1_data, n2_data = self.graph.nodes[n1], self.graph.nodes[n2]
        fflabel1, fflabel2 = n1_data['force_field_type'], n2_data['force_field_type']
        r_1 = UFF4MOF_DATA[fflabel1][0]
        r_2 = UFF4MOF_DATA[fflabel2][0]
        chi_1 = UFF4MOF_DATA[fflabel1][8]
        chi_2 = UFF4MOF_DATA[fflabel2][8]

        rbo = -0.1332*(r_1 + r_2)*math.log(data['order'])
        ren = r_1*r_2*(((math.sqrt(chi_1) - math.sqrt(chi_2))**2))/(chi_1*r_1 + chi_2*r_2)
        r0 = (r_1 + r_2 + rbo - ren)
        # The values for K in the UFF paper were set such that in the final
        # harmonic function, they would be divided by '2' to satisfy the
        # form K/2(R-Req)**2
        # in Lammps, the value for K is already assumed to be divided by '2'
        K = 664.12*(UFF4MOF_DATA[fflabel1][5]*UFF4MOF_DATA[fflabel2][5])/(r0**3) / 2.
        if (self.keep_metal_geometry) and (n1_data['atomic_number'] in METALS
            or n2_data['atomic_number'] in METALS):
            r0 = data['length']
        data['potential'] = BondPotential.Harmonic()
        data['potential'].K = K
        data['potential'].R0 = r0
        return 1

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
        a, b, c, data = angle
        angle_type = self.uff_angle_type(b)

        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        ab_bond = self.graph[a][b]
        bc_bond = self.graph[b][c]

        auff, buff, cuff = a_data['force_field_type'], b_data['force_field_type'], c_data['force_field_type']

        theta0 = UFF4MOF_DATA[buff][1]
        # just check if the central node is a metal, then apply a rigid angle term.
        # NB: Functional form may change dynamics, but at this point we will not
        # concern ourselves if the force constants are big.
        if (self.keep_metal_geometry) and (b_data['atomic_number'] in METALS):
            theta0 = self.graph.compute_angle_between(a, b, c)
            angle_type = "None"

        cosT0 = math.cos(theta0*DEG2RAD)
        sinT0 = math.sin(theta0*DEG2RAD)

        c2 = 1.0 / (4.0*sinT0*sinT0)
        c1 = -4.0 * c2 * cosT0
        c0 = c2 * (2.0*cosT0*cosT0 + 1.0)

        za = UFF4MOF_DATA[auff][5]
        zc = UFF4MOF_DATA[cuff][5]

        r_ab = ab_bond['potential'].R0
        r_bc = bc_bond['potential'].R0
        r_ac = math.sqrt(r_ab*r_ab + r_bc*r_bc - 2.*r_ab*r_bc*cosT0)

        beta = 664.12/r_ab/r_bc
        ka = beta*(za*zc /(r_ac**5.))*r_ab*r_bc
        ka *= (3.*r_ab*r_bc*(1. - cosT0*cosT0) - r_ac*r_ac*cosT0)
        #if ("special_flag" in b_data.keys()) and b_data["special_flag"] == "Cu_pdw":
        #    angle_type = "None"
        #    print(self.graph.compute_angle_between(a, b, c))
        #    # try the fourier expansion instead of the small term.

        if angle_type in sf or (angle_type == 'tetrahedral' and int(theta0) == 90):
            if angle_type == 'linear':
                kappa = ka
                c0 = -1.
                B = 1
                c1 = 1.
            # the description of the actual parameters for 'n' are not obvious
            # for the tetrahedral special case from the UFF paper or the write up in TOWHEE.
            # The values were found in the TOWHEE source code (eg. Bi3+3).
            if angle_type == 'tetrahedral':
                kappa = ka/4.
                c0 = -1.
                B = -1
                c1 = 2.

            if angle_type == 'trigonal-planar':
                kappa = ka/9.
                c0 = -1.
                B = -1
                c1 = 3.

            if angle_type == 'square-planar' or angle_type == 'octahedral':
                kappa = ka/16.
                c0 = -1.
                B = 1
                c1 = 4.

            #data['potential'] = AnglePotential.FourierSimple()
            data['potential'] = AnglePotential.CosinePeriodic()
            #data['potential'].K = kappa
            # this is done for you in the source code of LAMMPS k[i] = c_one/(n_one*n_one);
            data['potential'].C = kappa*(c1**2)
            #data['potential'].c = c0
            data['potential'].B = B
            data['potential'].n = c1
            return 1
        # general-nonlinear
        else:

            #TODO: a bunch of special cases which require molecular recognition here..
            # water, for example has it's own theta0 angle.

            kappa = ka
            data['potential'] = AnglePotential.Fourier()
            data['potential'].K = kappa
            data['potential'].C0 = c0
            data['potential'].C1 = c1
            data['potential'].C2 = c2
            return 1

    def uff_angle_type(self, b):
        name = self.graph.nodes[b]['force_field_type']
        try:
            coord_type = name[2]
        except IndexError:
            # eg, H_, F_
            return 'linear'
        if coord_type == "1":
            return 'linear'
        elif coord_type in ["R", "2"]:
            return 'trigonal-planar'
        elif coord_type == "3":
            return 'tetrahedral'
        elif coord_type == "4":
            return 'square-planar'
        elif coord_type == "5":
            return 'trigonal-bipyrimidal'
        elif coord_type == "6":
            return 'octahedral'
        elif coord_type == "8":
            return 'cubic-antiprism'
        else:
            print("ERROR: Cannot find coordination type for %s"%name)
            sys.exit()

    def dihedral_term(self, dihedral):
        """Use a small cosine Fourier expansion

        E_phi = 1/2*V_phi * [1 - cos(n*phi0)*cos(n*phi)]


        this is available in Lammps in the form of a harmonic potential
        E = K * [1 + d*cos(n*phi)]

        NB: the d term must be negated to recover the UFF potential.
        """
        a,b,c,d, data = dihedral
        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        d_data = self.graph.nodes[d]

        torsiontype = self.graph[b][c]['order']

        coord_bc = (self.graph.degree(b), self.graph.degree(c))
        bc = (b_data['force_field_type'], c_data['force_field_type'])
        M = mul(*coord_bc)
        V = 0
        n = 0
        mixed_case = ((b_data['hybridization'] == 'sp2' or b_data['hybridization'] == 'aromatic') and
                      c_data['hybridization'] == 'sp3') or \
                (b_data['hybridization'] == 'sp3' and
                (c_data['hybridization'] == 'sp2' or c_data['hybridization'] == 'aromatic'))
        all_sp2 = ((b_data['hybridization'] == 'sp2' or b_data['hybridization'] == 'aromatic') and
                   c_data['hybridization'] == 'sp2' or c_data['hybridization'] == 'aromatic')
        all_sp3 = (b_data['hybridization'] == 'sp3' and
                   c_data['hybridization'] == 'sp3')

        phi0 = 0
        if (b_data['atomic_number'] in METALS or c_data['atomic_number'] in METALS):
            return None
        if all_sp3:
            phi0 = 60.0
            n = 3
            vi = UFF4MOF_DATA[b_data['force_field_type']][6]
            vj = UFF4MOF_DATA[c_data['force_field_type']][6]

            if b_data['atomic_number'] == 8:
                vi = 2.
                n = 2
                phi0 = 90.
            elif b_data['atomic_number'] in (16, 34, 52, 84):
                vi = 6.8
                n = 2
                phi0 = 90.0
            if c_data['atomic_number'] == 8:
                vj = 2.
                n = 2
                phi0 = 90.0

            elif c_data['atomic_number'] in (16, 34, 52, 84):
                vj = 6.8
                n = 2
                phi0 = 90.0

            V = (vi*vj)**0.5

        elif all_sp2:
            ui = UFF4MOF_DATA[b_data['force_field_type']][7]
            uj = UFF4MOF_DATA[c_data['force_field_type']][7]
            phi0 = 180.0
            n = 2
            V = 5.0 * (ui*uj)**0.5 * (1. + 4.18*math.log(torsiontype))

        elif mixed_case:
            phi0 = 180.0
            n = 3
            V = 2.

            if c_data['hybridization'] == 'sp3':
                if c_data['atomic_number'] in (8, 16, 34, 52):
                    n = 2
                    phi0 = 90.
            elif b_data['hybridization'] == 'sp3':
                if b_data['atomic_number'] in (8, 16, 34, 52):
                    n = 2
                    phi0 = 90.0
            # special case group 6 elements
            if n==2:
                ui = UFF4MOF_DATA[b_data['force_field_type']][7]
                uj = UFF4MOF_DATA[c_data['force_field_type']][7]
                V = 5.0 * (ui*uj)**0.5 * (1. + 4.18*math.log(torsiontype))

        V /= float(M)
        nphi0 = n*phi0

        if abs(math.sin(nphi0*DEG2RAD)) > 1.0e-3:
            print("WARNING!!! nphi0 = %r" % nphi0)

        if (self.keep_metal_geometry) and (b_data['atomic_number'] in METALS or
            c_data['atomic_number'] in METALS):
            # must use different potential with minimum at the computed dihedral
            # angle.
            nphi0 = n*self.graph.compute_dihedral_between(a, b, c, d)
            data['potential'] = DihedralPotential.Charmm()
            data['potential'].K = 0.5
            data['potential'].d = 180 + nphi0
            data['potential'].n = n
            return 1
        if V==0.:
            return None
        data['potential'] = DihedralPotential.Harmonic()
        data['potential'].K = 0.5*V
        data['potential'].d = -math.cos(nphi0*DEG2RAD)
        data['potential'].n = n
        return 1

    def improper_term(self, improper):
        """Improper term described by a Fourier function

        E = K*[C_0 + C_1*cos(w) + C_2*cos(2*w)]
        """
        a, b, c, d, data = improper
        b_data = self.graph.nodes[b]
        a_ff = self.graph.nodes[a]['force_field_type']
        c_ff = self.graph.nodes[c]['force_field_type']
        d_ff = self.graph.nodes[d]['force_field_type']
        if not b_data['atomic_number'] in (6, 7, 8, 15, 33, 51, 83):
            return None
        if b_data['force_field_type'] in ('N_3', 'N_2', 'N_R', 'O_2', 'O_R'):
            c0 = 1.0
            c1 = -1.0
            c2 = 0.0
            koop = 6.0
        elif b_data['force_field_type'] in ('P_3+3', 'As3+3', 'Sb3+3', 'Bi3+3'):
            if b_data['force_field_type'] == 'P_3+3':
                phi = 84.4339 * DEG2RAD
            elif b_data['force_field_type'] == 'As3+3':
                phi = 86.9735 * DEG2RAD
            elif b_data['force_field_type'] == 'Sb3+3':
                phi = 87.7047 * DEG2RAD
            else:
                phi = 90.0 * DEG2RAD
            c1 = -4.0 * math.cos(phi)
            c2 = 1.0
            c0 = -1.0*c1*math.cos(phi) + c2*math.cos(2.0*phi)
            koop = 22.0
        elif b_data['force_field_type'] in ('C_2', 'C_R'):
            c0 = 1.0
            c1 = -1.0
            c2 = 0.0
            koop = 6.0
            if 'O_2' in (a_ff, c_ff, d_ff):
                # check to make sure an aldehyde (i.e. not carboxylate bonded to metal)
                if a_ff == "O_2" and self.graph.degree(a) == 1:
                    koop = 50.0
                elif c_ff == "O_2" and self.graph.degree(c) == 1:
                    koop = 50.0
                elif d_ff == "O_2" and self.graph.degree(d) == 1:
                    koop = 50.0
        else:
            return None

        koop /= 3.

        data['potential'] = ImproperPotential.Fourier()
        data['potential'].K = koop
        data['potential'].C0 = c0
        data['potential'].C1 = c1
        data['potential'].C2 = c2
        return 1

    def special_commands(self):
        st = ["%-15s %s %s"%("pair_modify", "tail yes", "mix arithmetic"),
              "%-15s %s"%("special_bonds", "lj/coul 0.0 0.0 1.0"),
              "%-15s %.1f"%('dielectric', 1.0)
              ]
        return st

    def detect_ff_terms(self):
        """All new terms are associated with inorganic clusters, and the number of cases are extensive.
        Implementation of all of the metal types would require a bit of effort, but should be done
        in the near future.

        """
        # for each atom determine the ff type if it is None
        organics = ["C", "N", "O", "S"]
        halides = ["F", "Cl", "Br", "I"]
        for node, data in self.graph.nodes_iter2(data=True):
            special = 'special_flag' in data
            if data['force_field_type'] is None:
                if special:
                    # Zn4O case TODO(pboyd): generalize these cases...
                    if data['special_flag'] == "O_z_Zn4O":
                        data['force_field_type'] = "O_3_f"
                    elif data['special_flag'] == "Zn4O":
                        data['force_field_type'] = "Zn3f2"
                        # change the bond orders to 0.5 as per the paper
                        for n in self.graph.neighbors(node):
                            self.graph[node][n]['order'] = 0.5
                            # woops! this is correct only for the M3O type SBUs
                            #if self.graph.nodes[n]['special_flag'] == "O_z_Zn4O":
                            #    self.graph[node][n]['order'] = 1.0
                            #else:
                            #    self.graph[node][n]['order'] = 0.5
                    elif data['special_flag'] == "C_Zn4O":
                        data['force_field_type'] = "C_R"
                        for n in self.graph.neighbors(node):
                            if self.graph.nodes[n]['element'] == "O":
                                self.graph[node][n]['order'] = 1.5
                            elif self.graph.nodes[n]['element'] == "C":
                                self.graph[node][n]['order'] = 1
                    elif data['special_flag'] == "O_c_Zn4O":
                        data['force_field_type'] = 'O_2'

                    # Copper Paddlewheel TODO(pboyd): generalize these cases...
                    elif data['special_flag'] == "O1_Cu_pdw" or data['special_flag'] == "O2_Cu_pdw":
                        data['force_field_type'] = 'O_2'
                    elif data['special_flag'] == "Cu_pdw":
                        data['force_field_type'] = 'Cu4+2'
                        for n in self.graph.neighbors(node):
                            if self.graph.nodes[n]['element'] == "Cu":
                                self.graph[node][n]['order'] = 0.25
                            else:
                                self.graph[node][n]['order'] = 0.5
                    elif data['special_flag'] == "C_Cu_pdw":
                        data['force_field_type'] = 'C_R'
                        for n in self.graph.neighbors(node):
                            if self.graph.nodes[n]['element'] == "O":
                                self.graph[node][n]['order'] = 1.5
                            elif self.graph.nodes[n]['element'] == "C":
                                self.graph[node][n]['order'] = 1

                    # Zn Paddlewheel TODO(pboyd): generalize these cases...
                    elif data['special_flag'] == "O1_Zn_pdw" or data['special_flag'] == "O2_Zn_pdw":
                        data['force_field_type'] = 'O_2'
                    elif data['special_flag'] == "Zn_pdw":
                        data['force_field_type'] = 'Zn4+2'
                        for n in self.graph.neighbors(node):
                            if self.graph.nodes[n]['element'] == "Zn":
                                self.graph[node][n]['order'] = 0.25
                            else:
                                self.graph[node][n]['order'] = 0.5
                    elif data['special_flag'] == "C_Zn_pdw":
                        data['force_field_type'] = 'C_R'
                        for n in self.graph.neighbors(node):
                            if self.graph.nodes[n]['element'] == "O":
                                self.graph[node][n]['order'] = 1.5
                            elif self.graph.nodes[n]['element'] == "C":
                                self.graph[node][n]['order'] = 1

                    # Al Pillar TODO(pboyd): generalize these cases...
                    elif data['special_flag'] == "O_c_Al_pillar":
                        data['force_field_type'] = 'O_2'
                    elif data['special_flag'] == "O_z_Al_pillar":
                        data['force_field_type'] = 'O_2'
                    elif data['special_flag'] == "H_Al_pillar":
                        data['force_field_type'] = 'H_'
                    elif data['special_flag'] == "Al_pillar":
                        data['force_field_type'] = 'Al6+3'
                    elif data['special_flag'] == "C_Al_pillar":
                        data['force_field_type'] = 'C_R'
                        for n in self.graph.neighbors(node):
                            if self.graph.nodes[n]['element'] == "O":
                                self.graph[node][n]['order'] = 1.5
                            elif self.graph.nodes[n]['element'] == "C":
                                self.graph[node][n]['order'] = 1

                    # V Pillar TODO(pboyd): generalize these cases...
                    elif data['special_flag'] == "O_c_V_pillar":
                        data['force_field_type'] = 'O_2'
                    elif data['special_flag'] == "O_z_V_pillar":
                        data['force_field_type'] = 'O_2'
                    elif data['special_flag'] == "V_pillar":
                        data['force_field_type'] = 'V6+3'
                    elif data['special_flag'] == "C_V_pillar":
                        data['force_field_type'] = 'C_R'
                        for n in self.graph.neighbors(node):
                            if self.graph.nodes[n]['element'] == "O":
                                self.graph[node][n]['order'] = 1.5
                            elif self.graph.nodes[n]['element'] == "C":
                                self.graph[node][n]['order'] = 1

                elif data['element'] in organics:
                    neigh_elem = set([self.graph.nodes[i]['element'] for i in self.graph.neighbors(node)])
                    if data['hybridization'] == "sp3":
                        data['force_field_type'] = "%s_3"%data['element']

                    elif data['hybridization'] == "aromatic":
                        data['force_field_type'] = "%s_R"%data['element']
                    elif data['hybridization'] == "sp2":
                        data['force_field_type'] = "%s_2"%data['element']
                    elif data['hybridization'] == "sp":
                        data['force_field_type'] = "%s_1"%data['element']
                if data['element'] == "O" and self.graph.degree(node) == 2:
                    if neigh_elem <= metals:
                        data['force_field_type'] = "O_2"
                    if neigh_elem <= set(["Si", "Al"]):
                        data['force_field_type'] = "O_3_z"
                    if neigh_elem <= metals | set(["C"]):
                        data['force_field_type'] = "O_2"
                elif data['element'] == "O" and self.graph.degree(node) == 3:
                    if (neigh_elem <= metals):
                        data['force_field_type'] = "O_2_z"
                        # temp fix for UiO-series MOFs
                        if neigh_elem == set(["Zr"]):
                            data['force_field_type'] = "O_3_f"
                    else:
                        data['force_field_type'] = "O_2"
                elif data['element'] == "O" and self.graph.degree(node) == 4:
                    if (neigh_elem <= metals | set(["H"])):
                        data['force_field_type'] = "O_3_f"

                elif data['element'] == "H":
                    data['force_field_type'] = "H_"
                elif data['element'] in halides:
                    data['force_field_type'] = data['element']
                    if data['element'] == "F":
                        data['force_field_type'] += "_"
                    elif data['element'] == "I":
                        data['force_field_type'] += "_"
                elif data['element'] in metals:
                    if len(data['element']) == 1:
                        fftype = data['element'] + "_"
                    else:
                        fftype = data['element']

                    # get coordination number and angles between atoms
                    if(self.graph.degree(node) == 2):
                        fftype += "1f1"
                    elif(self.graph.degree(node) == 3):
                        fftype += "2f2"
                    elif(self.graph.degree(node) == 4):
                        # tetrahedral or square planar
                        if self.graph.coplanar(node):
                            fftype += "4f2"
                        # Could implement a tetrahedrality index here, but that would be overkill.
                        else:
                            fftype += "3f2"
                    elif(self.graph.degree(node) == 5):
                        # assume paddlewheels........
                        fftype += "4+2"
                    elif(self.graph.degree(node) == 6):
                        fftype += "6f3"
                    elif(self.graph.degree(node) == 8):
                        fftype += "8f4"
                    try:
                        UFF4MOF_DATA[fftype]
                        data['force_field_type'] = fftype
                    # couldn't find the force field type!
                    except KeyError:
                        try:
                            fftype = fftype.replace("f", "+")
                            UFF4MOF_DATA[fftype]
                        except KeyError:
                            pass
                    for n in self.graph.neighbors(node):
                        if self.graph.nodes[n]['element'] in metals:
                            self.graph[node][n]['order'] = 0.25
                        elif self.graph.nodes[n]['element'] == "O":
                            self.graph[node][n]['order'] = 0.5
                        # else: bond order stays = 1

                # WARNING, the following else statement will do unknown things to the system.
                if data['force_field_type'] is None:
                    ffs = list(UFF4MOF_DATA.keys())
                    for j in ffs:
                        if data['element'] == j[:2].strip("_"):
                            print("WARNING: Could not find an appropriate UFF4MOF type for %s. Assigning %s"%(
                                  data['element'], j))
                            data['force_field_type'] = j
            if data['force_field_type'] is None:
                print("ERROR: could not find the proper force field type for atom %i"%(data['index'])+
                        " with element: '%s'"%(data['element']))
                sys.exit()


class Dubbeldam(ForceField):

    def __init__(self, graph=None, **kwargs):
        self.pair_in_data = True
        # override existing arguments with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        if (graph is not None):
            self.graph = graph
            self.detect_ff_terms()
            self.compute_force_field_terms()

    def bond_term(self, edge):
        """
        Harmonic term

        E = 0.5 * K * (R - Req)^2

        """
        n1, n2, data = edge
        type1 = self.graph.nodes[n1]['force_field_type']
        type2 = self.graph.nodes[n2]['force_field_type']
        string = "_".join([type1, type2])
        if type1 == "Zn" or type2 == "Zn":
            #data['potential'] = BondPotential.Harmonic()
            #data['potential'].K = 0.0
            #data['potential'].R0 = 2.4
            return None

        if string not in Dub_bonds.keys():
            string = "_".join([type2, type1])
        if string not in Dub_bonds.keys():
            print("ERROR: Could not find the bond parameters for the bond between %s"%string)
            sys.exit()

        data['potential'] = BondPotential.Harmonic()
        data['potential'].K = Dub_bonds[string][0]*kBtokcal/2.
        data['potential'].R0 = Dub_bonds[string][1]
        return 1

    def angle_term(self, angle):
        """

        """
        a, b, c, data = angle
        K = 100.0
        a_data, b_data, c_data = self.graph.nodes[a], self.graph.nodes[b], self.graph.nodes[c]
        atype = a_data['force_field_type']
        btype = b_data['force_field_type']
        ctype = c_data['force_field_type']

        string = "_".join([atype, btype, ctype])

        if atype == "Zn" or btype == "Zn" or ctype == "Zn":
            return None

        if string not in Dub_angles.keys():
            string = "_".join([ctype, btype, atype])
        if string not in Dub_angles.keys():
            print("ERROR: Could not find the angle parameters for the atom types %s"%string)
            sys.exit()

        data['potential'] = AnglePotential.Harmonic()
        # check to make sure to divide by DEG2RAD**2
        data['potential'].K = Dub_angles[string][0]*kBtokcal/2. / DEG2RAD**2
        data['potential'].theta0 = Dub_angles[string][1]
        return 1

    def dihedral_term(self, dihedral):
        """
        The Dihedral potential has the form

        U(torsion) = k * (1 + cos(m*theta - theta0))

        in LAMMPS this potential can be accessed by the dihedral_style charmm

        E = K * [ 1 + d*cos(n*theta - d) ]
        """

        a,b,c,d, data = dihedral
        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        d_data = self.graph.nodes[d]

        atype = a_data['force_field_type']
        btype = b_data['force_field_type']
        ctype = c_data['force_field_type']
        dtype = d_data['force_field_type']
        if atype == "Zn" or btype == "Zn" or ctype == "Zn" or dtype == "Zn":
            return None

        string = "_".join([atype, btype, ctype, dtype])
        if string not in Dub_dihedrals.keys():
            string = "_".join([dtype, ctype, btype, atype])
        if string not in Dub_dihedrals.keys():
            print("ERROR: Could not find the torsion parameters for the atom types %s"%string)
            sys.exit()
        w = 0.0
        data['potential'] = DihedralPotential.Charmm()
        data['potential'].K = Dub_dihedrals[string][0]*kBtokcal
        data['potential'].n = Dub_dihedrals[string][2]
        data['potential'].d = Dub_dihedrals[string][1]
        data['potential'].w = w
        return 1

    def improper_term(self, improper):
        """
        The Improper potential has the form

        U(torsion) = k * (1 + cos(m*theta - theta0))

        in LAMMPS this potential can be accessed by the improper_style cvff

        E = K * [ 1 + d*cos(n*theta) ]

        # NO out of phase shift! to get a minima at 180 set d to -1.
        # all of Dubbeldam's terms are for 180 degrees out-of-plane movement so
        # this is fine.

        """
        a, b, c, d, data = improper

        a_data = self.graph.nodes[a]
        b_data = self.graph.nodes[b]
        c_data = self.graph.nodes[c]
        d_data = self.graph.nodes[d]

        atype = a_data['force_field_type']
        btype = b_data['force_field_type']
        ctype = c_data['force_field_type']
        dtype = d_data['force_field_type']

        if atype == "Zn" or btype == "Zn" or ctype == "Zn" or dtype == "Zn":

            return None

        string = "_".join([atype, btype, ctype, dtype])
        if string not in Dub_impropers.keys():
            string = "_".join([atype, btype, dtype, ctype])
        if string not in Dub_impropers.keys():
            string = "_".join([ctype, btype, atype, dtype])
        if string not in Dub_impropers.keys():
            string = "_".join([ctype, btype, dtype, atype])
        if string not in Dub_impropers.keys():
            string = "_".join([dtype, btype, atype, ctype])
        if string not in Dub_impropers.keys():
            string = "_".join([dtype, btype, ctype, atype])
        if string not in Dub_impropers.keys():
            print("ERROR: Could not find the improper torsion parameters for the atom types %s"%string)
            sys.exit()
        data['potential'] = ImproperPotential.Cvff()

        # I have 3 impropers, he only has 1. Divide by 3 to get average?
        data['potential'].K = Dub_impropers[string][0]*kBtokcal / self.graph.degree(b)
        data['potential'].d = -1
        data['potential'].n = Dub_impropers[string][2]
        return 1

    def pair_terms(self, node, data, cutoff, **kwargs):
        """

        """
        data['pair_potential'] = PairPotential.LjCutCoulLong()
        data['pair_potential'].eps = Dub_atoms[data['force_field_type']][0]*kBtokcal
        data['pair_potential'].sig = Dub_atoms[data['force_field_type']][1]
        data['pair_potential'].cutoff = cutoff
        data['charge'] = Dub_atoms[data['force_field_type']][2]

    def special_commands(self):
        st = ["%-15s %s"%("pair_modify", "tail yes"),
              "%-15s %s"%("special_bonds", "lj/coul 0.0 0.0 1.0"),
              "%-15s %.1f"%('dielectric', 1.0)]
        return st

    def detect_ff_terms(self):
        """ Instead of the painful experience of coding a huge set of 'if' statements over
        extended bonding neighbours to identify IRMOF-1, 10 and 16, all of the force field types
        will be detected from maximum cliques.

        This means that failing to find the SBUs will result in a bad parameterization.

        """
        for node, data in self.graph.nodes_iter2(data=True):
            special = 'special_flag' in data
            if not special:
                print("ERROR: Some atoms were not detected as part of an SBU." +
                        " This is a requirement for successful parameterization of the "+
                        "Dubbeldam forcefield for the IRMOFs.")
                sys.exit()
            if data['special_flag'] == "O_z_Zn4O":
                data['force_field_type'] = "Oa"

            elif data['special_flag'] == "Zn4O":
                data['force_field_type'] = "Zn"

            elif data['special_flag'] == "C_Zn4O":
                data['force_field_type'] = "Ca"

            elif data['special_flag'] == "O_c_Zn4O":
                data['force_field_type'] = "Ob"

            else:
                # NB this works only because the special flags for the organic
                # SBUs are set to be the force field types assigned by Dubeldam!
                # if the 'special_flags' are changed for the aromatic molecules in mof_sbus.py then this
                # will break!!
                data['force_field_type'] = data['special_flag']

            if data['force_field_type'] is None:
                print("ERROR: could not find the proper force field type for atom %i"%(data['index'])+
                        " with element: '%s'"%(data['element']))
                sys.exit()


class SPC_E(ForceField):
    def __init__(self, graph=None, **kwargs):
        self.pair_in_data = True
        # override existing arguments with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        if (graph is not None):
            self.graph = graph
            self.detect_ff_terms()
            self.compute_force_field_terms()

    def bond_term(self, edge):
        """Harmonic term

        E = 0.5 * K * (R - Req)^2

        just a placeholder in LAMMPS.
        The bond is rigid and fixed by SHAKE in LAMMPS
        therefore an unambiguous extra flag must be
        added here to ensure the potential is not
        grouped with identical (however unlikely) potentials
        which are not rigid.

        """
        n1, n2, data = edge

        data['potential'] = BondPotential.Harmonic()
        data['potential'].K = 350.0
        data['potential'].R0 = 1.0
        data['potential'].special_flag = "shake"
        return 1

    def angle_term(self, angle):
        """Harmonic angle term.

        E = 0.5 * K * (theta - theta0)^2

        just a placeholder in LAMMPS.
        The angle is rigid and fixed by SHAKE in LAMMPS
        therefore an unambiguous extra flag must be
        added here to ensure the potential is not
        grouped with identical (however unlikely) potentials
        which are not rigid.

        """
        a, b, c, data = angle
        K = 100.0
        a_data, b_data, c_data = self.graph.nodes[a], self.graph.nodes[b], self.graph.nodes[c]
        atype = a_data['force_field_type']
        btype = b_data['force_field_type']
        ctype = c_data['force_field_type']

        assert (b_data['element'] == "O")

        data['potential'] = AnglePotential.Harmonic()
        data['potential'].K = K/2.
        data['potential'].theta0 = 109.47
        # extra flag?
        data['potential'].special_flag = "shake"
        return 1

    def dihedral_term(self, dihedral):
        """
        No dihedral potential in SPC/E water model.

        """
        return None

    def improper_term(self, improper):
        """
        No improper potential in SPC/E water model.

        """
        return None

    def pair_terms(self, node, data, cutoff, **kwargs):
        """
        Lennard - Jones potential for OW and HW.

        cutoff should be set to 9 angstroms, but this may
        be unrealistic.
        Also, no long range treatment of coulombic
        term! Otherwise
        this isn't technically the SPC/E model but
        Ewald should be used for periodic materials.

        """
        data['pair_potential'] = PairPotential.LjCutCoulLong()
        data['pair_potential'].eps = SPC_E_atoms[data['force_field_type']][2]
        data['pair_potential'].sig = SPC_E_atoms[data['force_field_type']][1]
        data['pair_potential'].cutoff = cutoff

    def special_commands(self):
        st = [
              "%-15s %s"%("pair_modify", "tail yes")
             ]
        return st

    def detect_ff_terms(self):
        """Water consists of O and H, not too difficult.

        """
        for node, data in self.graph.nodes_iter2(data=True):
            if data['element'] == "O":
                fftype = "OW"
            elif data['element'] == "H":
                fftype = "HW"
            else:
                print("ERROR: could not find the proper force field type for atom %i"%(data['index'])+
                        " with element: '%s'"%(data['element']))
                sys.exit()
            data['force_field_type'] = fftype
            data['mass'] = SPC_E_atoms[fftype][0]
            data['charge'] = SPC_E_atoms[fftype][3]

class TIP3P(ForceField):
    def __init__(self, graph=None, **kwargs):
        self.pair_in_data = True
        # override existing arguments with kwargs
        # TIP3P can have flexible OH bonds and angles,
        # set this to 'True' to enable a flexible
        # model.
        self.flexible = False
        for key, value in kwargs.items():
            setattr(self, key, value)

        if (graph is not None):
            self.graph = graph
            self.detect_ff_terms()
            self.compute_force_field_terms()

    def bond_term(self, edge):
        """Harmonic term

        E = 0.5 * K * (R - Req)^2

        just a placeholder in LAMMPS.
        The bond is rigid and fixed by SHAKE in LAMMPS
        therefore an unambiguous extra flag must be
        added here to ensure the potential is not
        grouped with identical (however unlikely) potentials
        which are not rigid.

        """
        n1, n2, data = edge

        data['potential'] = BondPotential.Harmonic()
        data['potential'].K = 450.0
        data['potential'].R0 = 0.9572
        if not self.flexible:
            data['potential'].special_flag = "shake"
        return 1

    def angle_term(self, angle):
        """Harmonic angle term.

        E = 0.5 * K * (theta - theta0)^2

        just a placeholder in LAMMPS.
        The angle is rigid and fixed by SHAKE in LAMMPS
        therefore an unambiguous extra flag must be
        added here to ensure the potential is not
        grouped with identical (however unlikely) potentials
        which are not rigid.

        """
        a, b, c, data = angle
        a_data, b_data, c_data = self.graph.nodes[a], self.graph.nodes[b], self.graph.nodes[c]
        atype = a_data['force_field_type']
        btype = b_data['force_field_type']
        ctype = c_data['force_field_type']

        assert (b_data['element'] == "O")

        data['potential'] = AnglePotential.Harmonic()
        data['potential'].K = 55.0
        data['potential'].theta0 = 104.52
        if not self.flexible:
            data['potential'].special_flag = "shake"
        return 1

    def dihedral_term(self, dihedral):
        """
        No dihedral potential in TIP3P water model.

        """
        return None

    def improper_term(self, improper):
        """
        No improper potential in TIP3P water model.

        """
        return None

    def pair_terms(self, node, data, cutoff, **kwargs):
        """
        Lennard - Jones potential for OW and HW.

        """
        data['pair_potential'] = PairPotential.LjCutCoulLong()
        data['pair_potential'].eps = TIP3P_atoms[data['force_field_type']][2]
        data['pair_potential'].sig = TIP3P_atoms[data['force_field_type']][1]
        data['pair_potential'].cutoff = cutoff

    def special_commands(self):
        st = [
              "%-15s %s"%("pair_modify", "tail yes")
             ]
        return st

    def detect_ff_terms(self):
        """Water consists of O and H, not too difficult.

        """
        for node, data in self.graph.nodes_iter2(data=True):
            if data['element'] == "O":
                fftype = "OW"
            elif data['element'] == "H":
                fftype = "HW"
            else:
                print("ERROR: could not find the proper force field type for atom %i"%(data['index'])+
                        " with element: '%s'"%(data['element']))
                sys.exit()
            data['force_field_type'] = fftype
            data['mass'] = TIP3P_atoms[fftype][0]
            data['charge'] = TIP3P_atoms[fftype][3]

class TIP4P(ForceField, TIP4P_Water):
    def __init__(self, graph=None, **kwargs):
        self.pair_in_data = True
        self.lammps_implicit = False
        for key, value in kwargs.items():
            setattr(self, key, value)

        if (graph is not None):
            self.graph = graph
            if not self.lammps_implicit:
                self.graph.rigid = True
            self.detect_ff_terms()
            self.compute_force_field_terms()

    def bond_term(self, edge):
        """Harmonic term

        E = 0.5 * K * (R - Req)^2

        just a placeholder in LAMMPS.
        The bond is rigid and fixed by SHAKE in LAMMPS
        therefore an unambiguous extra flag must be
        added here to ensure the potential is not
        grouped with identical (however unlikely) potentials
        which are not rigid.

        """
        n1, n2, data = edge
        n1data, n2data = self.graph.nodes[n1], self.graph.nodes[n2]
        n1fftype, n2fftype = n1data['force_field_type'], n2data['force_field_type']

        data['potential'] = BondPotential.Harmonic()
        if not self.lammps_implicit:
            data['potential'].K = 4500000.0
            data['potential'].special_flag = "rigid"
        else:
            data['potential'].K = 450.0
            data['potential'].special_flag = "shake"
        if set([n1fftype, n2fftype]) == set(["HW", "OW"]):
            data['potential'].R0 = self.ROH
        elif set([n1fftype, n2fftype]) == set(["X", "OW"]):
            data['potential'].R0 = self.Rdum
        return 1

    def angle_term(self, angle):
        """Harmonic angle term.

        E = 0.5 * K * (theta - theta0)^2

        just a placeholder in LAMMPS.
        The angle is rigid and fixed by SHAKE in LAMMPS
        therefore an unambiguous extra flag must be
        added here to ensure the potential is not
        grouped with identical (however unlikely) potentials
        which are not rigid.

        """
        a, b, c, data = angle
        a_data, b_data, c_data = self.graph.nodes[a], self.graph.nodes[b], self.graph.nodes[c]
        atype = a_data['force_field_type']
        btype = b_data['force_field_type']
        ctype = c_data['force_field_type']

        assert (b_data['element'] == "O")

        data['potential'] = AnglePotential.Harmonic()
        if not self.lammps_implicit:
            data['potential'].K = 550000.0
            data['potential'].special_flag = "rigid"
        else:
            data['potential'].K = 55.0
            data['potential'].special_flag = "shake"

        if atype == "HW" and ctype == "HW":
            data['potential'].theta0 = self.HOH
        elif (set([atype,ctype]) == set(["X", "HW"])) and (not self.lammps_implicit):
            data['potential'].theta0 = self.graph.compute_angle_between(a,b,c)

        return 1

    def dihedral_term(self, dihedral):
        """
        No dihedral potential in TIP4P water model.

        """
        return None

    def improper_term(self, improper):
        """
        No improper potential in TIP4P water model.

        """
        return None

    def pair_terms(self, node, data, cutoff, **kwargs):
        """
        Lennard - Jones potential for OW and HW.

        """
        # check to see if the user wants to use the built-in TIP4P function,
        # or the explicit dummy model.
        if self.lammps_implicit:
            data['pair_potential'] = PairPotential.LjCutTip4pLong()
            data['pair_potential'].qdist = 0.1250
        else:
            data['pair_potential'] = PairPotential.LjCutCoulLong()
        data['pair_potential'].eps = TIP4P_atoms[data['force_field_type']][2]
        data['pair_potential'].sig = TIP4P_atoms[data['force_field_type']][1]
        data['pair_potential'].cutoff = cutoff

    def special_commands(self):
        st = [
              "%-15s %s"%("pair_modify", "tail yes")
             ]
        return st

    def detect_ff_terms(self):
        """Water consists of O and H, not too difficult.

        """
        for node, data in self.graph.nodes_iter2(data=True):
            if data['element'] == "O":
                fftype = "OW"
            elif data['element'] == "H":
                fftype = "HW"
            elif data['element'] == "X":
                fftype = "X"
            else:
                print("ERROR: could not find the proper force field type for atom %i"%(data['index'])+
                        " with element: '%s'"%(data['element']))
                sys.exit()
            data['force_field_type'] = fftype
            data['mass'] = TIP4P_atoms[fftype][0]
            data['charge'] = TIP4P_atoms[fftype][3]

class TIP5P(ForceField):
    def __init__(self, graph=None, **kwargs):
        self.pair_in_data = True
        for key, value in kwargs.items():
            setattr(self, key, value)

        if (graph is not None):
            self.graph = graph
            self.graph.rigid = True
            self.detect_ff_terms()
            self.compute_force_field_terms()

    def bond_term(self, edge):
        """Harmonic term

        E = 0.5 * K * (R - Req)^2

        just a placeholder in LAMMPS.
        The bond is rigid and fixed by SHAKE in LAMMPS
        therefore an unambiguous extra flag must be
        added here to ensure the potential is not
        grouped with identical (however unlikely) potentials
        which are not rigid.

        """
        n1, n2, data = edge

        n1data = self.graph.nodes[n1]
        n2data = self.graph.nodes[n2]
        data['potential'] = BondPotential.Harmonic()
        if (n1data['force_field_type'] == "X") or (n2data['force_field_type'] == "X"):
            data['potential'].R0 = 0.7
        else:
            data['potential'].R0 = 0.9572
        data['potential'].K = 450000.0 # strong bond potential to ensure that the structure
                                       # will not deviate far from its intended form
                                       # during a minimization
        data['potential'].special_flag = "rigid"
        return 1

    def angle_term(self, angle):
        """Harmonic angle term.

        E = 0.5 * K * (theta - theta0)^2

        just a placeholder in LAMMPS.
        The angle is rigid and fixed by SHAKE in LAMMPS
        therefore an unambiguous extra flag must be
        added here to ensure the potential is not
        grouped with identical (however unlikely) potentials
        which are not rigid.

        """
        a, b, c, data = angle
        a_data, b_data, c_data = self.graph.nodes[a], self.graph.nodes[b], self.graph.nodes[c]
        atype = a_data['force_field_type']
        btype = b_data['force_field_type']
        ctype = c_data['force_field_type']

        assert (b_data['element'] == "O")
        data['potential'] = AnglePotential.Harmonic()
        if atype == "X" and ctype == "X":
            data['potential'].theta0 = 109.47
        elif set([atype,ctype]) == set(["X", "HW"]):
            data['potential'].theta0 = self.graph.compute_angle_between(a,b,c)
        else:
            data['potential'].theta0 = 104.52 # HW - OW - HW

        data['potential'].K = 5550.0 # very strong angle term to ensure that if a minimization
                                     # is requested, the molecule does not deviate far from
                                     # its designed geometry
        data['potential'].special_flag = "rigid"
        return 1

    def dihedral_term(self, dihedral):
        """
        No dihedral potential in TIP5P water model.

        """
        return None

    def improper_term(self, improper):
        """
        No improper potential in TIP5P water model.

        """
        return None

    def pair_terms(self, node, data, cutoff, **kwargs):
        """
        Lennard - Jones potential for OW and HW.

        """
        data['pair_potential'] = PairPotential.LjCutCoulLong()
        data['pair_potential'].eps = TIP5P_atoms[data['force_field_type']][2]
        data['pair_potential'].sig = TIP5P_atoms[data['force_field_type']][1]
        data['pair_potential'].cutoff = cutoff

    def special_commands(self):
        st = [
              "%-15s %s"%("pair_modify", "tail yes")
             ]
        return st

    def detect_ff_terms(self):
        """Water consists of O and H, not too difficult.

        """
        for node, data in self.graph.nodes_iter2(data=True):
            if data['element'] == "O":
                fftype = "OW"
            elif data['element'] == "H":
                fftype = "HW"
            elif data['element'] == "X":
                fftype = "X"
            else:
                print("ERROR: could not find the proper force field type for atom %i"%(data['index'])+
                        " with element: '%s'"%(data['element']))
                sys.exit()
            data['force_field_type'] = fftype
            data['mass'] = TIP5P_atoms[fftype][0]
            data['charge'] = TIP5P_atoms[fftype][3]

class EPM2_CO2(ForceField):

    def __init__(self, graph=None, **kwargs):
        self.pair_in_data = True
        for key, value in kwargs.items():
            setattr(self, key, value)

        if (graph is not None):
            self.graph = graph
            self.graph.rigid = True
            self.detect_ff_terms()
            self.compute_force_field_terms()

    def bond_term(self, edge):
        """Harmonic term

        E = 0.5 * K * (R - Req)^2

        just a placeholder in LAMMPS.
        The bond is rigid
        therefore an unambiguous extra flag must be
        added here to ensure the potential is not
        grouped with identical (however unlikely) potentials
        which are not rigid.

        """
        n1, n2, data = edge

        n1data = self.graph.nodes[n1]
        n2data = self.graph.nodes[n2]
        data['potential'] = BondPotential.Harmonic()
        data['potential'].R0 = 1.149
        data['potential'].K = 450000.0 # strong bond potential to ensure that the structure
                                       # will not deviate far from its intended form
                                       # during a minimization
        data['potential'].special_flag = "rigid"
        return 1

    def angle_term(self, angle):
        """Harmonic angle term.

        E = 0.5 * K * (theta - theta0)^2

        Can be rigid or not with EPM2

        """
        a, b, c, data = angle
        a_data, b_data, c_data = self.graph.nodes[a], self.graph.nodes[b], self.graph.nodes[c]
        atype = a_data['force_field_type']
        btype = b_data['force_field_type']
        ctype = c_data['force_field_type']

        assert (b_data['element'] == "C")
        assert (a_data['element'] == "O")
        assert (c_data['element'] == "O")
        data['potential'] = AnglePotential.Harmonic()
        data['potential'].theta0 = EPM2_angles["_".join([atype,btype,ctype])][1]
        data['potential'].K = EPM2_angles["_".join([atype,btype,ctype])][0]/2.
        data['potential'].special_flag = "rigid"
        return 1

    def dihedral_term(self, dihedral):
        """
        No dihedral potential in EPM2 CO2 model.

        """
        return None

    def improper_term(self, improper):
        """
        No improper potential in EPM2 CO2 model.

        """
        return None

    def pair_terms(self, node, data, cutoff, **kwargs):
        """
        Lennard - Jones potential for Cx and Ox.

        """
        data['pair_potential'] = PairPotential.LjCutCoulLong()
        data['pair_potential'].eps = EPM2_atoms[data['force_field_type']][2]
        data['pair_potential'].sig = EPM2_atoms[data['force_field_type']][1]
        data['pair_potential'].cutoff = cutoff

    def special_commands(self):
        st = [
              "%-15s %s"%("pair_modify", "tail yes")
             ]
        return st

    def detect_ff_terms(self):
        """CO2 consists of C and O, not too difficult.

        """
        for node, data in self.graph.nodes_iter2(data=True):
            if data['element'] == "O":
                fftype = "Ox"
            elif data['element'] == "C":
                fftype = "Cx"
            else:
                print("ERROR: could not find the proper force field type for atom %i"%(data['index'])+
                        " with element: '%s'"%(data['element']))
                sys.exit()
            data['force_field_type'] = fftype
            data['mass'] = EPM2_atoms[fftype][0]
            data['charge'] = EPM2_atoms[fftype][3]
