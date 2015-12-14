from uff import UFF_DATA
from dreiding import DREIDING_DATA
from uff_nonbonded import UFF_DATA_nonbonded
from structure_data import Structure, Atom, Bond, Angle, Dihedral, PairTerm
from lammps_potentials import BondPotential, AnglePotential, DihedralPotential, ImproperPotential, PairPotential
import math
import numpy as np
from operator import mul
import itertools
import abc
import re
import sys
DEG2RAD = math.pi/180.

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
    
    @abc.abstractmethod
    def unique_atoms(self):
        """Computes the number of unique atoms in the structure"""
        count = 0
        ff_type = {}
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

    
    @abc.abstractmethod
    def unique_bonds(self):
        """Computes the number of unique bonds in the structure"""
        count = 0
        bb_type = {}
        for bond in self.structure.bonds:
            idx1, idx2 = bond.indices
            atm1, atm2 = self.structure.atoms[idx1], self.structure.atoms[idx2]
            
            self.bond_term(bond)        
            try:
                type = bb_type[(atm1.ff_type_index, atm2.ff_type_index, bond.order)]
            except KeyError:
                try:
                    type = bb_type[(atm2.ff_type_index, atm1.ff_type_index, bond.order)]
                except KeyError:
                    count += 1
                    type = count
                    bb_type[(atm1.ff_type_index, atm2.ff_type_index, bond.order)] = type

                    self.unique_bond_types[type] = bond 
            bond.ff_type_index = type
    
    @abc.abstractmethod
    def unique_angles(self):
        ang_type = {}
        count = 0
        for angle in self.structure.angles:
            atom_a, atom_b, atom_c = angle.atoms
            type_a, type_b, type_c = atom_a.ff_type_index, atom_b.ff_type_index, atom_c.ff_type_index
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

    @abc.abstractmethod
    def unique_dihedrals(self):
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
                    self.dihedral_term(dihedral)
                    self.unique_dihedral_types[type] = dihedral 
            dihedral.ff_type_index = type

    @abc.abstractmethod
    def unique_impropers(self):
        """How many times to list the same set of atoms ???"""
        count = 0
        improper_type = {}
        for improper in self.structure.impropers:
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
                type = improper_type[d1]
            elif d2 in improper_type.keys():
                type = improper_type[d2]
            elif d3 in improper_type.keys():
                type = improper_type[d3]
            elif d4 in improper_type.keys():
                type = improper_type[d4]
            elif d5 in improper_type.keys():
                type = improper_type[d5]
            elif d6 in improper_type.keys():
                type = improper_type[d6]
            else:
                count += 1
                type = count
                improper_type[d1] = type
                self.improper_term(improper)
                self.unique_improper_types[type] = improper

            improper.ff_type_index = type     

    @abc.abstractmethod
    def unique_pair_terms(self):
        """This is force field dependant."""
        count = 0 
        pair_type = {}
        for pair in self.structure.pairs:
            p1 = (pair.atoms[0].ff_type_index, pair.atoms[1].ff_type_index)
            p2 = (pair.atoms[1].ff_type_index, pair.atoms[0].ff_type_index)
            if p1 in pair_type.keys():
                type = pair_type[p1]
            elif p2 in pair_type.keys():
                type = pair_type[p2]
            else:
                count += 1
                type = count
                pair_type[p1] = type
                self.pair_term(pair)
                self.unique_pair_types[type] = pair
            pair.ff_type_index = type
        return

    @abc.abstractmethod
    def define_styles(self):
        self.kspace_style = "ewald %f"%(0.001)
        bonds = set([j.potential.name for j in list(self.unique_bond_types.values())])
        if len(list(bonds)) > 1:
            self.bond_style = "hybrid %s"%" ".join(list(bonds))
        else:
            self.bond_style = "%s"%list(bonds)[0]
            for b in list(self.unique_bond_types.values()):
                b.potential.reduced = True

        angles = set([j.potential.name for j in list(self.unique_angle_types.values())])
        if len(list(angles)) > 1:
            self.angle_style = "hybrid %s"%" ".join(list(angles))
        else:
            self.angle_style = "%s"%list(angles)[0]
            for a in list(self.unique_angle_types.values()):
                a.potential.reduced = True

        dihedrals = set([j.potential.name for j in list(self.unique_dihedral_types.values())])
        if len(list(dihedrals)) > 1:
            self.dihedral_style = "hybrid %s"%" ".join(list(dihedrals))
        else:
            self.dihedral_style = "%s"%list(dihedrals)[0]
            for d in list(self.unique_dihedral_types.values()):
                d.potential.reduced = True

        impropers = set([j.potential.name for j in list(self.unique_improper_types.values())])
        if len(list(impropers)) > 1:
            self.improper_style = "hybrid %s"%" ".join(list(impropers))
        else:
            self.improper_style = "%s"%list(impropers)[0]
            for i in list(self.unique_improper_types.values()):
                i.potential.reduced = True

        pairs = set([j.potential.name for j in list(self.unique_pair_types.values())])
        if len(list(pairs)) > 1:
            self.pair_style = "hybrid %s"%" ".join(list(pairs))
        else:
            self.pair_style = "%s"%list(pairs)[0]
            for p in list(self.unique_pair_types.values()):
                p.potential.reduced = True

    @abc.abstractmethod
    def compute_force_field_terms(self):
        self.unique_atoms()
        self.unique_bonds()
        self.unique_angles()
        self.unique_dihedrals()
        self.unique_impropers()
        self.unique_pair_terms()
        self.define_styles()


class UserFF(ForceField):

    def __init__(self, struct):
        self.structure = struct
        self.unique_atom_types = {}
        self.unique_bond_types = {}
        self.unique_angle_types = {}
        self.unique_dihedral_types = {}
        self.unique_improper_types = {}
        self.unique_pair_types = {}

    def bond_term(self, bond):
        pass
    def angle_term(self, angle):
        pass
    def dihedral_term(self, dihedral):
        pass
    def improper_term(self, improper):
        pass

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
                #self.improper_term(improper)
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

class UFF(ForceField):
    """Parameterize the periodic material with the UFF parameters.
    NB: I have recently come across important information regarding the
    implementation of UFF from the author of MCCCS TOWHEE.
    It can be found here: (as of 05/11/2015)
    http://towhee.sourceforge.net/forcefields/uff.html

    The ammendments mentioned that document are included here
    """
    
    def __init__(self, struct):
        self.structure = struct

        self.unique_atom_types = {}
        self.unique_bond_types = {}
        self.unique_angle_types = {}
        self.unique_dihedral_types = {}
        self.unique_improper_types = {}
        self.unique_pair_types = {}

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
        # The values for K in the UFF paper were set such that in the final
        # harmonic function, they would be divided by '2' to satisfy the
        # form K/2(R-Req)**2
        # in Lammps, the value for K is already assumed to be divided by '2'
        K = 664.12*(UFF_DATA[fflabel1][5]*UFF_DATA[fflabel2][5])/(r0**3) / 2.

        bond.potential = BondPotential.Harmonic()
        bond.potential.K = K
        bond.potential.R0 = r0

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
        
        r_ab = ab_bond.potential.R0
        r_bc = bc_bond.potential.R0
        r_ac = math.sqrt(r_ab*r_ab + r_bc*r_bc - 2.*r_ab*r_bc*cosT0)

        beta = 664.12/r_ab/r_bc
        ka = beta*(za*zc /(r_ac**5.))
        ka *= (3.*r_ab*r_bc*(1. - cosT0*cosT0) - r_ac*r_ac*cosT0)
        if angle_type in sf or (angle_type == 'tetrahedral' and int(theta0) == 90):
            if angle_type == 'linear':
                kappa = ka
                c0 = 1.
                c1 = 1.
            # the description of the actual parameters for 'n' are not obvious
            # for the tetrahedral special case from the UFF paper or the write up in TOWHEE.
            # The values were found in the TOWHEE source code (eg. Bi3+3).
            if angle_type == 'tetrahedral': 
                kappa = ka/4.
                c0 = 1.
                c1 = 2.

            if angle_type == 'trigonal-planar':
                kappa = ka/9.
                c0 = -1.
                c1 = 3.

            if angle_type == 'square-planar' or angle_type == 'octahedral':
                kappa = ka/16.
                c0 = -1.
                c1 = 4.

            angle.potential = AnglePotential.FourierSimple()
            angle.potential.K = kappa
            angle.potential.c = c0
            angle.potential.n = c1
        # general-nonlinear
        else:

            #TODO: a bunch of special cases which require molecular recognition here..
            # water, for example has it's own theta0 angle.

            c2 = 1. / (4.*sinT0*sinT0)
            c1 = -4.*c2*cosT0
            c0 = c2*(2.*cosT0*cosT0 + 1)
            kappa = ka
            angle.potential = AnglePotential.Fourier()
            angle.potential.K = kappa
            angle.potential.C0 = c0
            angle.potential.C1 = c1
            angle.potential.C2 = c2

    def uff_angle_type(self, angle):
        l, c, r = angle.atoms
        
        name = angle.b_atom.force_field_type
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
        else:
            print("ERROR: Cannot find coordination type for %s"%name)
            sys.exit()
        #bent_types = ['O_3', 'S_3', 'O_R', 'N_2']
        #if len(c.neighbours) == 2:
        #    if name[:3] in bent_types:
        #        return 'bent'
        #    return 'linear'

        #trig_types = ['C_R', 'C_2', 'O_3_z', 'O_3', 'N_R']
        #if len(c.neighbours) == 3:
        #    if name in trig_types:
        #        return 'trigonal-planar'
        #    return 'trigonal-pyramidal'

        ## Need flag for Zn4O type MOFs where Zn is tetrahedral.
        ## vs. Zn paddlewheel where Zn is square planar.
        #sqpl_types = ['Fe6+2', 'Cu3+1']
        #if len(c.neighbours) == 4:
        #    if name in sqpl_types:
        #        return 'square-planar'
        #    return 'tetrahedral'

        #if len(c.neighbours) == 6:
        #    return 'octahedral'

    def uff_type(self, ufftype):
        if ufftype[2] == '3':
            return 'sp3'
        elif ufftype[2] == '2' or ufftype[2] == 'R':
            return 'sp2'
        elif ufftype[2] == '1':
            return 'sp'

    def dihedral_term(self, dihedral):
        """Use a small cosine Fourier expansion

        E_phi = 1/2*V_phi * [1 - cos(n*phi0)*cos(n*phi)]


        this is available in Lammps in the form of a harmonic potential
        E = K * [1 + d*cos(n*phi)]

        NB: the d term must be negated to recover the UFF potential.
        """
        atom_a = dihedral.a_atom
        atom_b = dihedral.b_atom
        atom_c = dihedral.c_atom
        atom_d = dihedral.d_atom

        torsiontype = dihedral.bc_bond.order
        coord_bc = (len(atom_b.neighbours), len(atom_c.neighbours))
        bc = (atom_b.force_field_type, atom_c.force_field_type)
        M = mul(*coord_bc)
        V = 0
        n = 0
        #FIXME(pboyd): coord = (4, x) in cases probably copper paddlewheel

        mixed_case = (self.uff_type(bc[0]) == 'sp2' and
                      self.uff_type(bc[1]) == 'sp3') or \
                (self.uff_type(bc[0]) == 'sp3' and 
                 self.uff_type(bc[1]) == 'sp2') 
        all_sp2 = (self.uff_type(bc[0]) == 'sp2' and
                   self.uff_type(bc[1]) == 'sp2')
        all_sp3 = (self.uff_type(bc[0]) == 'sp3' and 
                   self.uff_type(bc[1]) == 'sp3')

        phi0 = 0
        if all_sp3:
            phi0 = 60.0
            n = 3
            vi = UFF_DATA[atom_b.force_field_type][6]
            vj = UFF_DATA[atom_c.force_field_type][6]
            
            if atom_b.atomic_number == 8:
                vi = 2.
                n = 2
                phi0 = 90.
            elif atom_b.atomic_number in (16, 34, 52, 84):
                vi = 6.8
                n = 2
                phi0 = 90.0
            if atom_c.atomic_number == 8:
                vj = 2.
                n = 2
                phi0 = 90.0

            elif atom_c.atomic_number in (16, 34, 52, 84):
                vj = 6.8
                n = 2
                phi0 = 90.0

            V = (vi*vj)**0.5 # CHECK UNITS!!!!

        elif all_sp2: 
            ui = UFF_DATA[atom_b.force_field_type][7]
            uj = UFF_DATA[atom_c.force_field_type][7]
            phi0 = 180.0
            n = 2
            V = 5.0 * (ui*uj)**0.5 * (1. + 4.18*math.log(torsiontype))

        elif mixed_case: 
            phi0 = 180.0
            n = 3
            V = 2.  # CHECK UNITS!!!!
            
            if self.uff_type(bc[1]) == 'sp3':
                if atom_c.atomic_number in (8, 16, 34, 52):
                    n = 2
                    phi0 = 90.
            elif self.uff_type(bc[0]) == 'sp3': 
                if atom_b.atomic_number in (8, 16, 34, 52):
                    n = 2
                    phi0 = 90.0
            # special case group 6 elements
            if n==2: 
                ui = UFF_DATA[atom_b.force_field_type][7]
                uj = UFF_DATA[atom_c.force_field_type][7]
                V = 5.0 * (ui*uj)**0.5 * (1. + 4.18*math.log(torsiontype))

        V /= float(M)
        nphi0 = n*phi0

        if abs(math.sin(nphi0*DEG2RAD)) > 1.0e-3:
            print("WARNING!!! nphi0 = %r" % nphi0)
        
        dihedral.potential = DihedralPotential.Harmonic()
        dihedral.potential.K = 0.5*V
        dihedral.potential.d = -math.cos(nphi0*DEG2RAD)
        dihedral.potential.n = n

    def improper_term(self, improper):
        """
        The improper function can be described with a fourier function

        E = K*[C_0 + C_1*cos(w) + C_2*cos(2*w)

        """
        atom_a, atom_b, atom_c, atom_d = improper.atoms
        if atom_b.force_field_type in ('N_3', 'N_2', 'N_R', 'O_2', 'O_R'):
            c0 = 1.0
            c1 = -1.0
            c2 = 0.0
            koop = 6.0 
        elif atom_b.force_field_type in ('P_3+3', 'As3+3', 'Sb3+3', 'Bi3+3'):
            if atom_b.force_field_type == 'P_3+3':
                phi = 84.4339 * DEG2RAD
            elif atom_b.force_field_type == 'As3+3':
                phi = 86.9735 * DEG2RAD
            elif atom_b.force_field_type == 'Sb3+3':
                phi = 87.7047 * DEG2RAD
            else:
                phi = 90.0 * DEG2RAD
            c1 = -4.0 * math.cos(phi)
            c2 = 1.0
            c0 = -1.0*c1*math.cos(phi) + c2*math.cos(2.0*phi)
            koop = 22.0 
        elif atom_b.force_field_type in ('C_2', 'C_R'):
            c0 = 1.0
            c1 = -1.0
            c2 = 0.0
            koop = 6.0 
            if 'O_2' in (atom_a.force_field_type, atom_c.force_field_type, atom_d.force_field_type):
                koop = 50.0 
        else:
            return 
        
        #NB: TOWHEE divides by 3.
        koop /= 3. # Not clear in UFF paper, but division by the number of bonds is probably not appropriate. Should test on real systems..

        improper.potential = ImproperPotential.Fourier()
        improper.potential.K = koop
        improper.potential.C0 = c0
        improper.potential.C1 = c1
        improper.potential.C2 = c2

    def pair_term(self, pair):
        atom1 = pair.atoms[0]
        atom2 = pair.atoms[1]
        eps1 = UFF_DATA[atom1.force_field_type][3]
        sig1 = UFF_DATA[atom1.force_field_type][2]*(2**(-1./6.))
        eps2 = UFF_DATA[atom2.force_field_type][3]
        sig2 = UFF_DATA[atom2.force_field_type][2]*(2**(-1./6.))
        # default LB mixing.
        eps = np.sqrt(eps1*eps2)
        sig = (sig1 + sig2) / 2.
        pot = PairPotential.LjCutCoulLong()
        pot.eps = eps
        pot.sig = sig
        pair.potential = pot 

    @property
    def cutoff(self):
        return 12.5


class Dreiding(ForceField):

    def __init__(self, struct):
        self.structure = struct
        self.unique_atom_types = {}
        self.unique_bond_types = {}
        self.unique_angle_types = {}
        self.unique_dihedral_types = {}
        self.unique_improper_types = {}
        self.unique_hbond_types = {}
        self.unique_pair_types = {}
   
    def get_hybridization(self, type):
        try:
            hy = type[2]
        except IndexError:
            hy = '_'
        return hy 

    def get_bond_order(self, type1, type2, order=None):
        """Return the bond order based on the DREIDING atom types."""
        if order is None:

            o1 = self.get_hybridization(type1)
            o2 = self.get_hybridization(type2)

            if (o1 == '2') and (o2 == '2'):
                return 2.
            elif (o1 == '1') and (o2 == '1'):
                return 3.
            # This is not explicitly stated in the DREIDING paper, but
            # we are assuming here that an aromitic bond has an order
            # of 1.5
            elif (o1 == 'R') and (o2 == 'R'):
                return 1.5
            else:
                return 1.0
        else:
            return order

    def bond_term(self, bond, type='harmonic'):
        """The DREIDING Force Field contains two possible bond terms, harmonic and Morse.
        The authors recommend using harmonic as a default, and Morse potentials for more
        'refined' calculations. 
        Here we will assume a harmonic term by default, then the user can chose to switch
        to Morse if they so choose. (change type argument to 'morse')
        
        E = 0.5 * K * (R - Req)^2
        

        E = D * [exp{-(alpha*R - Req)} - 1]^2


        
        """
        atom1, atom2 = bond.atoms
        fflabel1, fflabel2 = atom1.force_field_type, atom2.force_field_type
        R1 = DREIDING_DATA[fflabel1][0]
        R2 = DREIDING_DATA[fflabel2][0]
        order = self.get_bond_order(fflabel1, fflabel2, bond.order)
        K = order*700.
        D = order*70.
        Re = R1 + R2 - 0.01
        if type.lower() == 'harmonic':

            bond.potential = BondPotential.Harmonic()
            bond.potential.K = K/2.

            bond.potential.R0 = Re

        elif type.lower() == 'morse':
            alpha = order * np.sqrt(K/2./D)
            bond.potential = BondPotential.Morse()
            bond.potential.D = D
            bond.potential.alpha = alpha
            bond.potential.R = Re

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
        K = 100.0
        a_atom, b_atom, c_atom = angle.atoms
        btype = b_atom.force_field_type
        theta0 = DREIDING_DATA[btype][1]
        if (theta0 == 180.):
            angle.potential = AnglePotential.Cosine()
            angle.potential.K = K
        else:
            angle.potential = AnglePotential.CosineSquared()
            K = 0.5*K/(np.sin(theta0*DEG2RAD))**2
            angle.potential.K = K
            angle.potential.theta0 = theta0

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

        a_atom, b_atom, c_atom, d_atom = dihedral.atoms

        btype = b_atom.force_field_type
        ctype = c_atom.force_field_type

        order = dihedral.bc_bond.order
        a_hyb = self.get_hybridization(a_atom.force_field_type)
        b_hyb = self.get_hybridization(btype)
        c_hyb = self.get_hybridization(ctype)
        d_hyb = self.get_hybridization(d_atom.force_field_type)

        # special cases associated with oxygen column, listed here
        oxygen_sp3 = ["O_3", "S_3", "Se3", "Te3"]
        non_oxygen_sp2 = ["C_R", "N_R", "B_2", "C_2", "N_2"]

        sp2 = ["R", "2"]

        # a)
        if((b_hyb == "3")and(c_hyb == "3")):
            V = 2.0
            n = 3
            phi0 = 180.0
            # h) special case..
            if((btype in oxygen_sp3) and (ctype in oxygen_sp3)): 
                V = 2.0
                n = 2
                phi0 = 90.0

        # b)
        elif(((b_hyb in sp2) and (c_hyb == "3"))or(
            (b_hyb == "3") and (c_hyb in sp2))):
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
        elif((b_hyb == "2") and (c_hyb == "2") and (order == 2.)):
            V = 45.0
            n = 2
            phi0 = 180.0

        # d)
        elif((b_hyb == "R") and (c_hyb == "R") and (order == 1.5)):
            V = 25.0
            n = 2
            phi0 = 180.0

        # e)
        elif((b_hyb in sp2) and (c_hyb in sp2) and (order == 1.0)):
            V = 5.0
            n = 2
            phi0 = 180.0
            # f) just check if neighbours are aromatic, then apply the exception
            # NB: this may fail for phenyl esters if the oxygen atoms are not 
            # labelled as "R" (i.e. will fail if they are O_2 or O_3)
            if(b_hyb == "R" and c_hyb == "R"):
                b_arom = True
                for id in b_atom.neighbours:
                    hyb = self.get_hybridization(
                            self.structure.atoms[id].force_field_type)
                    if (hyb != "R"):
                        b_arom = False
                c_arom = True
                for id in c_atom.neighbours:
                    hyb = self.get_hybridization(
                            self.structure.atoms[id].force_field_type)
                    if (hyb != "R"):
                        c_arom = False
                if (b_arom and c_arom):
                    V *= 2.0
        # g)
        elif((b_hyb == "1")or(b_hyb == "_")or
                (c_hyb == "1")or(c_hyb == "_")):
            V = 0.0
            n = 2
            phi0 = 180.0

        # divide V by the number of dihedral angles
        # to compute across this a-b bond
        b_neigh = len(b_atom.neighbours)
        c_neigh = len(c_atom.neighbours)
        norm = float(b_neigh * c_neigh)

        V /= norm
        d = n*phi0 + 180.0
        # default is to include the full 1-4 non-bonded interactions.
        w = 1.0 
        dihedral.potential = DihedralPotential.Charmm()
        dihedral.potential.K = V
        dihedral.potential.n = n
        dihedral.potential.d = d
        dihedral.potential.w = w

    def improper_term(self, improper):


