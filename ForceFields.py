from uff import UFF_DATA
from dreiding import DREIDING_DATA
from uff_nonbonded import UFF_DATA_nonbonded
from BTW import BTW_angles, BTW_dihedrals, BTW_opbends, BTW_atoms, BTW_bonds
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

    cutoff = 12.5

    @abc.abstractmethod
    def detect_ff_exist(self):
        """detects if all of the interactions are defined in the FF; for BTW_FF now!"""

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
        for node, data in self.graph.nodes_iter(data=True):
            if data['force_field_type'] is None:
                label = data['element']
            else:
                label = data['force_field_type']

            try:
                type = ff_type[label]
            except KeyError:
                count += 1
                type = count
                ff_type[label] = type  
                self.unique_atom_types[type] = atom 
            data['ff_type_index'] = type

    
    @abc.abstractmethod
    def unique_bonds(self):
        """Computes the number of unique bonds in the structure"""
        count = 0
        bb_type = {}
        for n1, n2, data in self.graph.edges_iter(data=True):
            self.bond_term((n1, n2, data))

            btype = "%s"%data['potential']
            try:
                type = bb_type[btype]

            except KeyError:
                count += 1
                type = count
                bb_type[btype] = type

                self.unique_bond_types[type] = bond

            data['ff_type_index'] = type
    
    @abc.abstractmethod
    def unique_angles(self):
        ang_type = {}
        count = 0
        for b, data in self.graph.nodes_iter(data=True):
            # compute and store angle terms
            try:
                ang_data = data['angle']
                for (a, c), val in ang_data.items():
                    self.angle_term((a, b, c, val))
                    atype = "%s"%val['potential']
                    try:
                        type = ang_type[atype]

                    except KeyError:
                        count += 1
                        type = count
                        ang_type[atype] = type
                        self.unique_angle_types[type] = angle 
                    val['ff_type_index'] = type

    @abc.abstractmethod
    def unique_dihedrals(self):
        count = 0
        dihedral_type = {}
        for dihedral in self.structure.dihedrals:
            # just use the potential parameter string
            self.dihedral_term(dihedral)
            dtype = "%s"%dihedral.potential
            try:
                type = dihedral_type[dtype]
            except KeyError:
                count += 1 
                type = count
                dihedral_type[dtype] = type
                self.unique_dihedral_types[type] = dihedral
            dihedral.ff_type_index = type

    @abc.abstractmethod
    def unique_impropers(self):
        """How many times to list the same set of atoms ???"""
        count = 0
        improper_type = {}
        remove = []
        for idx,improper in enumerate(self.structure.impropers):

            self.improper_term(improper)
            if improper.potential is not None:
                itype = "%s"%improper.potential
                try:
                    type = improper_type[itype]
                except KeyError:
                    count += 1
                    type = count
                    improper_type[itype] = type
                    self.unique_improper_types[type] = improper
                improper.ff_type_index = type
            else:
                remove.append(idx)
        for j in reversed(sorted(remove)):
            del(self.structure.impropers[j])

        # re-index the imporopers
        for idx, improper in enumerate(self.structure.impropers):
            improper.index = idx 

    @abc.abstractmethod
    def unique_pair_terms(self):
        """This is force field dependent."""
        return

    @abc.abstractmethod
    def define_styles(self):
        # should be more robust, some of the styles require multiple parameters specified on these lines
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
        elif len(list(impropers)) == 1:
            self.improper_style = "%s"%list(impropers)[0]
            for i in list(self.unique_improper_types.values()):
                i.potential.reduced = True
        else:
            self.improper_style = "" 
        pairs = set(["%r"%(j.potential) for j in list(self.unique_pair_types.values())])
        if len(list(pairs)) > 1:
            self.pair_style = "hybrid/overlay %s"%(" ".join(list(pairs)))
            # by default, turn off listing Pair Coeff in the data file if this is the case
            self.pair_in_data = False
        else:
            self.pair_style = list(pairs)[0]
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

    def __init__(self, graph):
        self.graph = graph 
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
    
    def __init__(self, struct):
        self.pair_in_data = False
        self.structure = struct
        self.unique_atom_types = {}
        self.unique_bond_types = {}
        self.unique_angle_types = {}
        self.unique_dihedral_types = {}
        self.unique_improper_types = {}
        self.unique_pair_types = {}


    def detect_ff_terms(self):
        # for each atom determine the ff type if it is None
        BTW_organics = [ "O", "C","H" ]
        BTW_metals = ["Zr","Cu","Zn"]
        for atom in self.structure.atoms:
            if atom.force_field_type is None:                                
                type_assigned=False
                neighbours = [self.structure.atoms[i] for i in atom.neighbours]
                neighbour_elements = [atom.element for atom in neighbours]
                if atom.element in BTW_organics:
                    if (atom.element == "O"):
                        if (set(neighbour_elements) <= set(BTW_metals + ["H"])):
                            if("H" in neighbour_elements): #O-H
                                atom.force_field_type="75"
                                atom.charge=BTW_atoms[atom.force_field_type][6]
                            else:     # O-inorganic
                                atom.force_field_type="171"
                                atom.charge=BTW_atoms[atom.force_field_type][6]
                        elif ("C" in neighbour_elements): # Carboxylate
                            atom.force_field_type="170"
                            atom.charge=BTW_atoms[atom.force_field_type][6]
                        else:
                            print("Oxygen number : %i could not be recognized!"%atom.index)
                            sys.exit()
        
                    elif (atom.element == "H"):
                        if ("O" in neighbour_elements):
                            atom.force_field_type="21"
                            atom.charge=BTW_atoms[atom.force_field_type][6]
                        elif("C" in neighbour_elements):
                            atom.force_field_type="915"
                            atom.charge=BTW_atoms[atom.force_field_type][6]
                        else:
                            print("Hydrogen number : %i could not be recognized!"%atom.index)            
                    else:# atom.element=="C"
                        if ("O" in neighbour_elements):
                            atom.force_field_type="913" # C-acid
                            atom.charge=BTW_atoms[atom.force_field_type][6]
                        elif ("H" in neighbour_elements):
                            atom.force_field_type="912" # C- benzene we should be careful that in this case C in ligand has also bond with H, but not in the FF
                            atom.charge=BTW_atoms[atom.force_field_type][6]
                        elif (set(neighbour_elements)<=set(["C"])):
                            for i in atom.neighbours:
                                neighboursofneighbour=[self.structure.atoms[j] for j in self.structure.atoms[i].neighbours]
                                neighboursofneighbour_elements=[at.element for at in neighboursofneighbour]
                                if ("O" in neighboursofneighbour_elements):
                                    atom.force_field_type="902"
                                    atom.charge=BTW_atoms[atom.force_field_type][6]
                                    type_assigned=True

                            if (type_assigned==False) and (atom.hybridization=="aromatic"):
                                atom.force_field_type="903"
                                atom.charge=BTW_atoms[atom.force_field_type][6]
                            elif (type_assigned==False):
                                print("Carbon number : %i could not be recognized! erorr1 %s "%(atom.index, atom.hybridization))
                        
                        else:
                            print("Carbon number : %i could not be recognized! error2"%atom.index)

                elif atom.element in BTW_metals:
                    if (atom.element == "Zr"):
                        atom.force_field_type="192"
                        atom.charge=BTW_atoms[atom.force_field_type][6]
                    elif (atom.element == "Cu"):
                        atom.force_field_type="185"
                        atom.charge=BTW_atoms[atom.force_field_type][6]
                    else: # atom type = Zn
                        atom.force_field_type="172"
                        atom.charge=BTW_atoms[atom.force_field_type][6]
                else:
                        print('Error!! Cannot detect atom types. Atom type does not exist in BTW-FF!')
            else:
                print('FFtype already assigned!')

    def detect_ff_exist(self):
        """
           checking angles
        """
        nonexisting_angles=[]
        missing_labels=[]
        for angle in self.structure.angles:
            a_atom, b_atom, c_atom = angle.atoms
            atom_a_fflabel, atom_b_fflabel, atom_c_fflabel = a_atom.force_field_type, b_atom.force_field_type, c_atom.force_field_type
            angle_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel
            if not angle_fflabel in BTW_angles:
                nonexisting_angles.append(angle.index)
                missing_labels.append(angle_fflabel)

        for ii , NE_angle in enumerate(nonexisting_angles):
            del self.structure.angles[NE_angle-ii]

        for ff_label in set(missing_labels):
                print ("%s angle does not exist in FF!"%(ff_label))
        """
           checking dihedrals 
        """
        missing_labels=[]
        nonexisting_dihedral=[]
        for dihedral in self.structure.dihedrals:
            atom_a = dihedral.a_atom
            atom_b = dihedral.b_atom
            atom_c = dihedral.c_atom
            atom_d = dihedral.d_atom
            atom_a_fflabel, atom_b_fflabel, atom_c_fflabel,atom_d_fflabel = atom_a.force_field_type, atom_b.force_field_type, atom_c.force_field_type, atom_d.force_field_type
            dihedral_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel

            if not dihedral_fflabel in BTW_dihedrals:
                nonexisting_dihedral.append(dihedral.index)
                missing_labels.append(dihedral_fflabel)

        for ii , NE_dihedral in enumerate(nonexisting_dihedral):
            del self.structure.dihedrals[NE_dihedral-ii]

        for ff_label in set(missing_labels):
                print ("%s dihedral does not exist in FF!"%(ff_label))

        missing_labels=[]
        nonexisting_improper=[]
        for improper in self.structure.impropers:
            atom_a, atom_b, atom_c, atom_d = improper.atoms
            atom_a_fflabel, atom_b_fflabel, atom_c_fflabel,atom_d_fflabel = atom_a.force_field_type, atom_b.force_field_type, atom_c.force_field_type, atom_d.force_field_type
            improper_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel

            if not improper_fflabel in BTW_opbends:
                nonexisting_improper.append(improper.index)

        for ii , NE_improper in enumerate(nonexisting_improper):
            del self.structure.impropers[NE_improper-ii]
       
        for ff_label in set(missing_labels):
                print ("%s improper does not exist in FF!"%(ff_label))
         

 
        return None


        
    def bond_term(self, edge):
        """class2 assumed"""
        atom1, atom2 = bond.atoms
        fflabel1, fflabel2 = atom1.force_field_type, atom2.force_field_type
        bond_fflabel=fflabel1+"_"+fflabel2
        Ks = BTW_bonds[bond_fflabel][0] # the value should be in kcal/mol
        l0 = BTW_bonds[bond_fflabel][1] # the value should be in Angstrom
        
        """
        Es=71.94*Ks*(l-l0)^2[1-2.55(l-l0)+(7/12)*2.55*(l-l0)^2]
        (Allinger et. al. J.Am.Chem.Soc., Vol. 111, No. 23, 1989)
        """
        """ 
        K2=71.94*Ks
        bond.potential = BondPotential.Harmonic()
        bond.potential.K = K2
        bond.potential.R0 = l0
        """
        """ 
        ### MM3
        K2=71.94*Ks
        K3=-2.55*K2
        K4=7.0/12.0*2.55*K2
        bond.potential = BondPotential.Class2()
        bond.potential.K2 = K2
        bond.potential.K3 = K3
        bond.potential.K4 = K4
        bond.potential.R0 = l0
        
        """ 
        
        ### SM1
        K2=66.64*Ks
        K3=-141.1*Ks
        K4=127.9*Ks
        bond.potential = BondPotential.Class2()
        bond.potential.K2 = K2
        bond.potential.K3 = K3
        bond.potential.K4 = K4
        bond.potential.R0 = l0
        
    def angle_term(self, angle):
        """
        Be careful that the 5and6 order terms are vanished here since they are not implemented in LAMMPS!!


        Etheta = 0.021914*Ktheta*(theta-theta0)^2[1-0.014(theta-theta0)+5.6(10^-5)*(theta-theta0)^2-7.0*(10^-7)*(theta-theta0)^3+9.0*(10^-10)*(theta-theta0)^4]        
        (Allinger et. al. J.Am.Chem.Soc., Vol. 111, No. 23, 1989)
        """
        a_atom, b_atom, c_atom = angle.atoms

        atom_a_fflabel, atom_b_fflabel, atom_c_fflabel = a_atom.force_field_type, b_atom.force_field_type, c_atom.force_field_type
        angle_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel
        theta0 = BTW_angles[angle_fflabel][1]
        Ktheta = BTW_angles[angle_fflabel][0]
        
        K2 = 0.021914*Ktheta/(DEG2RAD**2)
        K3 = -0.014*K2/(DEG2RAD**1)
        K4 = 5.6e-5*K2/(DEG2RAD**2)
        
       # angle.potential = AnglePotential.FourierSimple()
       # angle.potential.K = K2 
       # angle.potential.c = K3
       # angle.potential.n = K4

        angle.potential = AnglePotential.Class2()
        angle.potential.theta0 = theta0 
        angle.potential.K2 = K2
        angle.potential.K3 = K3 
        angle.potential.K4 = K4


    def dihedral_term(self, dihedral):
        """
        Ew = (V1/2)(1 + cos w) + (V2/2)(1 - cos 2*w) (V3/2)(1 + cos 3*w)
        (Allinger et. al. J.Am.Chem.Soc., Vol. 111, No. 23, 1989)
        """        
        atom_a = dihedral.a_atom
        atom_b = dihedral.b_atom
        atom_c = dihedral.c_atom
        atom_d = dihedral.d_atom
        
        atom_a_fflabel, atom_b_fflabel, atom_c_fflabel,atom_d_fflabel = atom_a.force_field_type, atom_b.force_field_type, atom_c.force_field_type, atom_d.force_field_type
        dihedral_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel
        kt1 = 0.5 * BTW_dihedrals[dihedral_fflabel][0]        
        kt2 = 0.5 * BTW_dihedrals[dihedral_fflabel][3]        
        kt3 = 0.5 * BTW_dihedrals[dihedral_fflabel][6]        
        phi1 = BTW_dihedrals[dihedral_fflabel][1] + 180        #ask Davide to be sure, TODO (Mohamad)
        phi2 = BTW_dihedrals[dihedral_fflabel][4] + 180        
        phi3 = BTW_dihedrals[dihedral_fflabel][7] + 180        



        dihedral.potential = DihedralPotential.Class2()
        dihedral.potential.K1=kt1
        dihedral.potential.phi1=phi1
        dihedral.potential.K2=kt2
        dihedral.potential.phi2=phi2
        dihedral.potential.K3=kt3
        dihedral.potential.phi3=phi3
        dihedral.K = kt2
        dihedral.d = phi2
        dihedral.n = phi1
    #    dihedral.potential = DihedralPotential.Harmonic()
    #    dihedral.potential.K = kt1
    #    dihedral.potential.d = phi1
    #    dihedral.potential.n = 1
 




    def improper_term(self, improper):
        """
        The improper function can be described with a fourier function

        E = K*[C_0 + C_1*cos(w) + C_2*cos(2*w)

        """
        atom_a, atom_b, atom_c, atom_d = improper.atoms
        atom_a_fflabel, atom_b_fflabel, atom_c_fflabel,atom_d_fflabel = atom_a.force_field_type, atom_b.force_field_type, atom_c.force_field_type, atom_d.force_field_type
        improper_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel
        
        Kopb = BTW_opbends[improper_fflabel][0]/(DEG2RAD**2)
        c0 =  BTW_opbends[improper_fflabel][1]
         
                
        improper.potential = ImproperPotential.Harmonic()
        improper.potential.K = Kopb 
        improper.potential.chi0 = c0

    def unique_pair_terms(self):  
        """This is force field dependent."""
        count = 0 
        pair_type = {}
        atom_types = list(self.unique_atom_types.keys())
        for at1 in sorted(atom_types):
            for at2 in sorted(atom_types[at1-1:]):
                atom1=self.unique_atom_types[at1]
                atom2=self.unique_atom_types[at2]
                
                p1 = (atom1.ff_type_index , atom2.ff_type_index)
                pair=PairTerm(atom1,atom2)

                if p1 in pair_type.keys():  # why we have this here?
                    type = pair_type[p1]
                else:
                    count += 1
                    type = count
                    pair_type[p1] = type
                    self.pair_term(pair)
                    self.unique_pair_types[type] = pair
                pair.ff_type_index = type
       
        return



    def pair_term(self, pair):
        """
        Buckingham equation in MM3 type is used!
        (TODO) check the units!

        """
        atom1 = pair.atoms[0]
        atom2 = pair.atoms[1]
        eps1 = BTW_atoms[atom1.force_field_type][4]
        sig1 = BTW_atoms[atom1.force_field_type][3]
        eps2 = BTW_atoms[atom2.force_field_type][4]
        sig2 = BTW_atoms[atom2.force_field_type][3]
        # MM3 mixing rules: Arithmetic mean for radii
        #                   Geometric mean for epsilon 
        eps = np.sqrt(eps1*eps2)
        Rv = sig1 + sig2 
        Rho = Rv/12.0
        A = 1.84e5 * eps
        C=2.25*(Rv)**6*eps
        
        pot = PairPotential.BuckCoulLong()
        pot.A = A
        pot.cutoff = self.cutoff
        pot.rho = Rho
        pot.C = C
        pair.potential = pot


    def special_commands(self):  # TODO (MOHAMAD) check if it is correct
        st = ""
        st += "%-15s %s\n"%("pair_modify", "tail yes")
        return st






class MOF_FF(ForceField):
    
    def __init__(self, struct):
        self.pair_in_data = False
        self.structure = struct
        self.unique_atom_types = {}
        self.unique_bond_types = {}
        self.unique_angle_types = {}
        self.unique_dihedral_types = {}
        self.unique_improper_types = {}
        self.unique_pair_types = {}


    def detect_ff_terms(self):
        # for each atom determine the ff type if it is None
        MOF_FF_organics = [ "O", "C","H"  ]
        MOF_FF_metals = ["Zr","Cu","Zn"]
        
        """
        Detecting the SBU type:
        Cu-paddle-wheel
        Zn cluster --> IRMOF series
        Zr cluster --> UiO series
        """
        for atom in self.structure.atoms:
            if atom.element=="Cu":
                SBU="Cu_paddle_wheel"
                break
            elif atom.element=="Zr":
                SBU="IRMOF"
                break
            elif atom.element=="Zn":
                SBU="UiO"
                break
            else:
                continue
        
        print("Cluster type detected as: %10s"%SBU)
 
        for atom in self.structure.atoms:
            if atom.force_field_type is None:                                
                type_assigned=False
                neighbours = [self.structure.atoms[i] for i in atom.neighbours]
                neighbour_elements = [atom.element for atom in neighbours]
                if atom.element in MOF_FF_organics:
                    if(SBU=="Cu_paddle_wheel"):       
                        if (atom.element == "O"):
                            if (set(neighbour_elements)<=set(["Cu","C"]) and len(atom.neighbours)==2):
                                atom.force_field_type="170"
                                atom.charge=BTW_atoms[atom.force_field_type][6]
                            else:
                                print("Oxygen number %i type cannot be detected in Cu-paddle-wheel MOF!"%atom.index)
                                sys.exit() 
#                            if (set(neighbour_elements) <= set(BTW_metals + ["H"])):
#                                if("H" in neighbour_elements): #O-H
#                                    atom.force_field_type="75"
#                                    atom.charge=BTW_atoms[atom.force_field_type][6]
#                                else:     # O-inorganic
#                                    atom.force_field_type="171"
#                                    atom.charge=MOF_FF_atoms[atom.force_field_type][6]
#                            elif ("C" in neighbour_elements): # Carboxylate
#                                atom.force_field_type="170"
#                                atom.charge=MOF_FF_atoms[atom.force_field_type][6]
#                            else:
#                                print("Oxygen number : %i could not be recognized!"%atom.index)
#                                sys.exit()
            
                        elif (atom.element == "H"):
                            if set(neighbour_elements)<=set(["C"]):
                                atom.force_field_type="915"
                                atom.charge=BTW_atoms[atom.force_field_type][6]
                            else:
                                print("Hydrogen number %i type cannot be detected in Cu-paddle-wheel MOF!"%atom.index)
                                sys.exit() 





#                            if ("O" in neighbour_elements):
#                                atom.force_field_type="21"
#                                atom.charge=BTW_atoms[atom.force_field_type][6]
#                            elif("C" in neighbour_elements):
#                                atom.force_field_type="915"
#                                atom.charge=BTW_atoms[atom.force_field_type][6]
#                            else:
#                                print("Hydrogen number : %i could not be recognized!"%atom.index)            
                        else:# atom.element=="C"
                            if ("O" in neighbour_elements):
                                atom.force_field_type="913" # C-acid
                                atom.charge=BTW_atoms[atom.force_field_type][6]
                            elif ("H" in neighbour_elements):
                                atom.force_field_type="912" # C- benzene we should be careful that in this case C in ligand has also bond with H, but not in the FF
                                atom.charge=BTW_atoms[atom.force_field_type][6]
                            elif (set(neighbour_elements)<=set(["C"])):
                                for i in atom.neighbours:
                                    neighboursofneighbour=[self.structure.atoms[j] for j in self.structure.atoms[i].neighbours]
                                    neighboursofneighbour_elements=[at.element for at in neighboursofneighbour]
                                    if ("O" in neighboursofneighbour_elements):
                                        atom.force_field_type="902"
                                        atom.charge=BTW_atoms[atom.force_field_type][6]
                                        type_assigned=True
    
                                if (type_assigned==False) and (atom.hybridization=="aromatic"):
                                    atom.force_field_type="912" # only C_ph exists in MOF-FF
                                    atom.charge=BTW_atoms[atom.force_field_type][6]
                                elif (type_assigned==False):
                                    print("Carbon number : %i could not be recognized! erorr1 %s "%(atom.index, atom.hybridization))
                            
                            else:
                                print("Carbon number : %i could not be recognized! error2"%atom.index)
    
                elif atom.element in MOF_FF_metals:
                    if (atom.element == "Zr"):
                        atom.force_field_type="192"
                        atom.charge=BTW_atoms[atom.force_field_type][6]
                    elif (atom.element == "Cu"):
                        atom.force_field_type="185"
                        atom.charge=BTW_atoms[atom.force_field_type][6]
                    else: # atom type = Zn
                        atom.force_field_type="172"
                        atom.charge=BTW_atoms[atom.force_field_type][6]
                else:
                        print('Error!! Cannot detect atom types. Atom type does not exist in BTW-FF!')
            else:
                print('FFtype already assigned!')

    def detect_ff_exist(self):
        """
           checking angles
        """
        nonexisting_angles=[]
        missing_labels=[]
        for angle in self.structure.angles:
            a_atom, b_atom, c_atom = angle.atoms
            atom_a_fflabel, atom_b_fflabel, atom_c_fflabel = a_atom.force_field_type, b_atom.force_field_type, c_atom.force_field_type
            angle_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel
            if not angle_fflabel in BTW_angles:
                nonexisting_angles.append(angle.index)
                missing_labels.append(angle_fflabel)

        for ii , NE_angle in enumerate(nonexisting_angles):
            del self.structure.angles[NE_angle-ii]

        for ff_label in set(missing_labels):
                print ("%s angle does not exist in FF!"%(ff_label))
        """
           checking dihedrals 
        """
        missing_labels=[]
        nonexisting_dihedral=[]
        for dihedral in self.structure.dihedrals:
            atom_a = dihedral.a_atom
            atom_b = dihedral.b_atom
            atom_c = dihedral.c_atom
            atom_d = dihedral.d_atom
            atom_a_fflabel, atom_b_fflabel, atom_c_fflabel,atom_d_fflabel = atom_a.force_field_type, atom_b.force_field_type, atom_c.force_field_type, atom_d.force_field_type
            dihedral_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel

            if not dihedral_fflabel in BTW_dihedrals:
                nonexisting_dihedral.append(dihedral.index)
                missing_labels.append(dihedral_fflabel)

        for ii , NE_dihedral in enumerate(nonexisting_dihedral):
            del self.structure.dihedrals[NE_dihedral-ii]

        for ff_label in set(missing_labels):
                print ("%s dihedral does not exist in FF!"%(ff_label))

        missing_labels=[]
        nonexisting_improper=[]
        for improper in self.structure.impropers:
            atom_a, atom_b, atom_c, atom_d = improper.atoms
            atom_a_fflabel, atom_b_fflabel, atom_c_fflabel,atom_d_fflabel = atom_a.force_field_type, atom_b.force_field_type, atom_c.force_field_type, atom_d.force_field_type
            improper_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel

            if not improper_fflabel in BTW_opbends:
                nonexisting_improper.append(improper.index)

        for ii , NE_improper in enumerate(nonexisting_improper):
            del self.structure.impropers[NE_improper-ii]
       
        for ff_label in set(missing_labels):
                print ("%s improper does not exist in FF!"%(ff_label))
         

 
        return None


        
    def bond_term(self, bond):
        """class2 assumed"""
        atom1, atom2 = bond.atoms
        fflabel1, fflabel2 = atom1.force_field_type, atom2.force_field_type
        bond_fflabel=fflabel1+"_"+fflabel2
        Ks = BTW_bonds[bond_fflabel][0] # the value should be in kcal/mol
        l0 = BTW_bonds[bond_fflabel][1] # the value should be in Angstrom
        
        """
        Es=71.94*Ks*(l-l0)^2[1-2.55(l-l0)+(7/12)*2.55*(l-l0)^2]
        (Allinger et. al. J.Am.Chem.Soc., Vol. 111, No. 23, 1989)
        """

        K2=71.94*Ks
        K3=-2.55*71.94*Ks
        K4=7.0/12.0*2.55*71.94*Ks
        bond.potential = BondPotential.Harmonic()
        bond.potential.K = K2
   #     bond.potential.K3 = K3
   #     bond.potential.K4 = K4
        bond.potential.R0 = l0




    def angle_term(self, angle):
        """
        Be careful that the 5and6 order terms are vanished here since they are not implemented in LAMMPS!!


        Etheta = 0.021914*Ktheta*(theta-theta0)^2[1-0.014(theta-theta0)+5.6(10^-5)*(theta-theta0)^2-7.0*(10^-7)*(theta-theta0)^3+9.0*(10^-10)*(theta-theta0)^4]        
        (Allinger et. al. J.Am.Chem.Soc., Vol. 111, No. 23, 1989)
        """
        a_atom, b_atom, c_atom = angle.atoms

        atom_a_fflabel, atom_b_fflabel, atom_c_fflabel = a_atom.force_field_type, b_atom.force_field_type, c_atom.force_field_type
        angle_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel
        theta0 = BTW_angles[angle_fflabel][1]
        Ktheta = BTW_angles[angle_fflabel][0]
        
        K2 = 0.021914*Ktheta/(DEG2RAD**2)
        K3 = -0.014*K2/(DEG2RAD**1)
        K4 = 5.6e-5*K2/(DEG2RAD**2)
        
       # angle.potential = AnglePotential.FourierSimple()
       # angle.potential.K = K2 
       # angle.potential.c = K3
       # angle.potential.n = K4

        angle.potential = AnglePotential.Class2()
        angle.potential.theta0 = theta0 
        angle.potential.K2 = K2
        angle.potential.K3 = K3 
        angle.potential.K4 = K4


    def dihedral_term(self, dihedral):
        """
        Ew = (V1/2)(1 + cos w) + (V2/2)(1 - cos 2*w) (V3/2)(1 + cos 3*w)
        (Allinger et. al. J.Am.Chem.Soc., Vol. 111, No. 23, 1989)
        """        
        atom_a = dihedral.a_atom
        atom_b = dihedral.b_atom
        atom_c = dihedral.c_atom
        atom_d = dihedral.d_atom
        
        atom_a_fflabel, atom_b_fflabel, atom_c_fflabel,atom_d_fflabel = atom_a.force_field_type, atom_b.force_field_type, atom_c.force_field_type, atom_d.force_field_type
        dihedral_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel
        kt1 = 0.5 * BTW_dihedrals[dihedral_fflabel][0]        
        kt2 = 0.5 * BTW_dihedrals[dihedral_fflabel][3]        
        kt3 = 0.5 * BTW_dihedrals[dihedral_fflabel][6]        
        phi1 = BTW_dihedrals[dihedral_fflabel][1] + 180        #ask Davide to be sure, TODO (Mohamad)
        phi2 = BTW_dihedrals[dihedral_fflabel][4] + 180        
        phi3 = BTW_dihedrals[dihedral_fflabel][7] + 180        



        dihedral.potential = DihedralPotential.Class2()
        dihedral.potential.K1=kt1
        dihedral.potential.phi1=phi1
        dihedral.potential.K2=kt2
        dihedral.potential.phi2=phi2
        dihedral.potential.K3=kt3
        dihedral.potential.phi3=phi3
        dihedral.K = kt2
        dihedral.d = phi2
        dihedral.n = phi1
    #    dihedral.potential = DihedralPotential.Harmonic()
    #    dihedral.potential.K = kt1
    #    dihedral.potential.d = phi1
    #    dihedral.potential.n = 1
 




    def improper_term(self, improper):
        """
        The improper function can be described with a fourier function

        E = K*[C_0 + C_1*cos(w) + C_2*cos(2*w)

        """
        atom_a, atom_b, atom_c, atom_d = improper.atoms
        atom_a_fflabel, atom_b_fflabel, atom_c_fflabel,atom_d_fflabel = atom_a.force_field_type, atom_b.force_field_type, atom_c.force_field_type, atom_d.force_field_type
        improper_fflabel=atom_a_fflabel+"_"+atom_b_fflabel+"_"+atom_c_fflabel+"_"+atom_d_fflabel
        
        Kopb = BTW_opbends[improper_fflabel][0]*(DEG2RAD**2)
        c0 =  BTW_opbends[improper_fflabel][1]
         
                
        improper.potential = ImproperPotential.Harmonic()
        improper.potential.K = Kopb 
        improper.potential.chi0 = c0

    def unique_pair_terms(self):  
        """This is force field dependent."""
        count = 0 
        pair_type = {}
        atom_types = list(self.unique_atom_types.keys())
        for at1 in sorted(atom_types):
            for at2 in sorted(atom_types[at1-1:]):
                atom1=self.unique_atom_types[at1]
                atom2=self.unique_atom_types[at2]
                
                p1 = (atom1.ff_type_index , atom2.ff_type_index)
                pair=PairTerm(atom1,atom2)

                if p1 in pair_type.keys():  # why we have this here?
                    type = pair_type[p1]
                else:
                    count += 1
                    type = count
                    pair_type[p1] = type
                    self.pair_term(pair)
                    self.unique_pair_types[type] = pair
                pair.ff_type_index = type
       
        return



    def pair_term(self, pair):
        """
        Buckingham equation in MM3 type is used!
        (TODO) check the units!

        """
        atom1 = pair.atoms[0]
        atom2 = pair.atoms[1]
        eps1 = BTW_atoms[atom1.force_field_type][4]
        sig1 = BTW_atoms[atom1.force_field_type][3]
        eps2 = BTW_atoms[atom2.force_field_type][4]
        sig2 = BTW_atoms[atom2.force_field_type][3]
        # MM3 mixing rules: Arithmetic mean for radii
        #                   Geometric mean for epsilon 
        eps = np.sqrt(eps1*eps2)
        Rv = sig1 + sig2 
        Rho = Rv/12.0
        A = 1.84e5 * eps
        C=2.25*(Rv)**6*eps
        
        pot = PairPotential.BuckCoulLong()
        pot.A = A
        pot.cutoff = self.cutoff
        pot.rho = Rho
        pot.C = C
        pair.potential = pot


    def special_commands(self):  # TODO (MOHAMAD) check if it is correct
        st = ""
        st += "%-15s %s\n"%("pair_modify", "tail yes")
        return st


class UFF(ForceField):
    """Parameterize the periodic material with the UFF parameters.
    NB: I have come across important information regarding the
    implementation of UFF from the author of MCCCS TOWHEE.
    It can be found here: (as of 05/11/2015)
    http://towhee.sourceforge.net/forcefields/uff.html

    The ammendments mentioned that document are included here
    """
    
    def __init__(self, graph):
        self.pair_in_data = True
        self.graph = graph
        self.keep_metal_geometry = False
        self.unique_atom_types = {}
        self.unique_bond_types = {}
        self.unique_angle_types = {}
        self.unique_dihedral_types = {}
        self.unique_improper_types = {}
        self.unique_pair_types = {}

    def detect_ff_exist(self):
        return None

    def bond_term(self, edge):
        """Harmonic assumed"""
        n1, n2, data = edge
        n1_data, n2_data = self.graph.node[n1], self.graph.node[n2]
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

        edge['potential'] = BondPotential.Harmonic()
        edge['potential'].K = K
        edge['potential'].R0 = r0

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
        (a, b, c), data = angle
        angle_type = self.uff_angle_type(b)
        
        a_data = self.graph.node[a]
        b_data = self.graph.node[b]
        c_data = self.graph.node[c]
        ab_bond = self.graph[a][b]
        bc_bond = self.graph[b][c]

        auff, buff, cuff = a_data['force_field_type'], b_data['force_field_type'], c_data['force_field_type']

        theta0 = UFF_DATA[buff][1]
        cosT0 = math.cos(theta0*DEG2RAD)
        sinT0 = math.cos(theta0*DEG2RAD)

        c2 = 1.0 / (4.0*sinT0*sinT0)
        c1 = -4.0 * c2 * cosT0
        c0 = c2 * (2.0*cosT0*cosT0 + 1.0)

        za = UFF_DATA[auff][5]
        zc = UFF_DATA[cuff][5]
        
        r_ab = ab_bond['potential'].R0
        r_bc = bc_bond['potential'].R0
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

            data['potential'] = AnglePotential.FourierSimple()
            data['potential'].K = kappa
            data['potential'].c = c0
            data['potential'].n = c1
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

    def uff_angle_type(self, b):
        name = self.graph.node[b]['force_field_type']
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

    def dihedral_term(self, dihedral):
        """Use a small cosine Fourier expansion

        E_phi = 1/2*V_phi * [1 - cos(n*phi0)*cos(n*phi)]


        this is available in Lammps in the form of a harmonic potential
        E = K * [1 + d*cos(n*phi)]

        NB: the d term must be negated to recover the UFF potential.
        """
        a,b,c,d, data = dihedral
        a_data = self.graph.node[a]
        b_data = self.graph.node[b]
        c_data = self.graph.node[c]
        d_data = self.graph.node[d]

        torsiontype = self.graph[b][c]['order']
        coord_bc = (self.graph.degree(b), self.graph.degree(c))
        bc = (b_data['force_field_type'], c_data['force_field_type'])
        M = mul(*coord_bc)
        V = 0
        n = 0
        #FIXME(pboyd): coord = (4, x) in cases probably copper paddlewheel
        #TODO(pboyd): all of the hybridization and bond order info 
        # is determined automatically by the program now.
        # this must be updated for the UFF ForceField data
        mixed_case = (b_data['hybridization'] == 'sp2' and
                      c_data['hybridization'] == 'sp3') or \
                (b_data['hybridization'] == 'sp3' and 
                 c_data['hybridization'] == 'sp2') 
        all_sp2 = (b_data['hybridization'] == 'sp2' and
                   c_data['hybridization'] == 'sp2')
        all_sp3 = (b_data['hybridization'] == 'sp3' and 
                   c_data['hybridization'] == 'sp3')

        phi0 = 0
        if all_sp3:
            phi0 = 60.0
            n = 3
            vi = UFF_DATA[b_data['force_field_type']][6]
            vj = UFF_DATA[c_data['force_field_type']][6]
            
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
            ui = UFF_DATA[b_data['force_field_type']][7]
            uj = UFF_DATA[c_data['force_field_type']][7]
            phi0 = 180.0
            n = 2
            V = 5.0 * (ui*uj)**0.5 * (1. + 4.18*math.log(torsiontype))

        elif mixed_case: 
            phi0 = 180.0
            n = 3
            V = 2.  # CHECK UNITS!!!!
            
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
        
        data['potential'] = DihedralPotential.Harmonic()
        data['potential'].K = 0.5*V
        data['potential'].d = -math.cos(nphi0*DEG2RAD)
        data['potential'].n = n

    def improper_term(self, improper):
        """
        The improper function can be described with a fourier function

        E = K*[C_0 + C_1*cos(w) + C_2*cos(2*w)]

        """
        a, b, c, d, data = improper
        b_data = self.graph.node[b]
        a_ff = self.graph.node[a]['force_field_type']
        c_ff = self.graph.node[c]['force_field_type']
        d_ff = self.graph.node[d]['force_field_type']
        if not b_data['atomic_number'] in (6, 7, 8, 15, 33, 51, 83):
            return
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
                koop = 50.0 
        else:
            return 
        
        #NB: TOWHEE divides by 3.
        koop /= 3. # Not clear in UFF paper, but division by the number of bonds is probably not appropriate. Should test on real systems..

        data['potential'] = ImproperPotential.Fourier()
        data['potential'].K = koop
        data['potential'].C0 = c0
        data['potential'].C1 = c1
        data['potential'].C2 = c2
    
    def unique_pair_terms(self):
        """This is force field dependent."""
        count = 0 
        pair_type = {}
        atom_types = list(self.unique_atom_types.keys())
        for at in sorted(atom_types):
            atom = self.unique_atom_types[at]
            #a1 = self.unique_atom_types[pair1]
            #a2 = self.unique_atom_types[pair2]

            p1 = (atom.ff_type_index, atom.ff_type_index)
            #pair = PairTerm(a1, a2)
            pair = PairTerm(atom, atom)
            if p1 in pair_type.keys():
                type = pair_type[p1]
            else:
                count += 1
                type = count
                pair_type[p1] = type
                self.pair_term(pair)
                self.unique_pair_types[type] = pair
            pair.ff_type_index = type
        return

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
        pot.cutoff = self.cutoff
        pair.potential = pot 
    
    def special_commands(self):
        st = ""
        st += "%-15s %s %s\n"%("pair_modify", "tail yes", "mix arithmetic")
        return st

    def detect_ff_terms(self):
        # for each atom determine the ff type if it is None
        organics = ["C", "N", "O", "S"]
        halides = ["F", "Cl", "Br", "I"]
        for node, data in self.graph.nodes_iter(data=True):
            if data['force_field_type'] is None:
                if data['element'] in organics:
                    if data['hybridization'] == "sp3":
                        data['force_field_type'] = "%s_3"%data['element']
                        if data['element'] == "O" and self.graph.degree(node) >= 2:
                            neigh_elem = set([self.graph.node[i]['element'] for i in self.graph.neighbors(node)])
                            if not neigh_elem <= set(organics) | set(halides):
                                data['force_field_type'] = "O_3_z"

                    elif data['hybridization'] == "aromatic":
                        data['force_field_type'] = "%s_R"%data['element']
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
                else:
                    ffs = list(UFF_DATA.keys())
                    for j in ffs:
                        if data['element'] == j[:2].strip("_"):
                            data['force_field_type'] = j
            if data['force_field_type'] is None:
                print("ERROR: could not find the proper force field type for atom %i"%(data['index'])+
                        " with element: '%s'"%(data['element']))
                sys.exit()

class Dreiding(ForceField):

    def __init__(self, struct):
        self.pair_in_data = False
        self.structure = struct
        self.unique_atom_types = {}
        self.unique_bond_types = {}
        self.unique_angle_types = {}
        self.unique_dihedral_types = {}
        self.unique_improper_types = {}
        self.unique_pair_types = {}


    def detect_ff_exist(self):
        return None
   
    def unique_atoms(self):
        """Computes the number of unique atoms in the structure"""
        count = 0
        ff_type = {}
        electro_neg_atoms = ["N", "O", "F"]
        # RE-TYPE H_ to H__HB if needed
        remember = {}
        for atom in self.structure.atoms:
            if atom.element == "H":
                for n in atom.neighbours:
                    atj = self.structure.atoms[n]
                    if ((atj.element in electro_neg_atoms)
                        and (atom.force_field_type != "H__HB")):
                        decision = True
                        if atj.molecule_id[0] is not None:
                            try:
                                decision = remember[atj.molecule_id[0]]
                            except KeyError:
                                decision = input("Would you like molecule with atoms (%s), index %i"%(
                                                 ','.join(atj.molecule_id[0]), atj.molecule_id[1]) + 
                                                 " at atom %i, %s to form hydrogen bonds? [y/n]"%(
                                                     atj.index, atj.element)).lower()
                                if (decision == 'n') or (decision == 'no'):
                                    decision = False
                                elif (decision == 'y') or (decision == 'yes'):
                                    decision = True
                                else:
                                    print("ERROR: command %s not recognized"%(decision))
                                    sys.exit()
                                rem = input("Would you like to remember this answer for this molecule?")
                                if rem == 'y' or 'yes':
                                    remember[atj.molecule_id[0]] = decision
                        if decision:

                            print("WARNING: the atom %i is mis-labeled as %s. "%(
                            atom.index, atom.force_field_type) +
                            "Renaming to H__HB for hydrogen bonding possibilities.")
                            atom.force_field_type = "H__HB"
                            atj.h_bond_donor = True

        for atom in self.structure.atoms:
            if atom.force_field_type is None:
                label = atom.element
            else:
                label = atom.force_field_type
            if atom.h_bond_donor:
                label += "_HB"
            try:
                type = ff_type[label]
            except KeyError:
                count += 1
                type = count
                ff_type[label] = type  
                self.unique_atom_types[type] = atom 

            atom.ff_type_index = type


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
            #angle.potential = AnglePotential.Harmonic()
            K = 0.5*K/(np.sin(theta0*DEG2RAD))**2
            #K = 0.5*K
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
            if(b_hyb == "R" and c_hyb == "R"):
                b_arom = True
                for cycle in b_atom.rings:
                    # Need to make sure this isn't part of the same ring.
                    if c_atom.index in cycle:
                        b_arom = False
                        print("Warning: two resonant atoms "+
                              "%s and %s"%(b_atom.ciflabel, c_atom.ciflabel)+
                              "in the same ring have a bond order of 1.0! "
                              "This will likely yield unphysical characteristics"+
                              " of your system.")


                c_arom = True
                for cycle in c_atom.rings:
                    # Need to make sure this isn't part of the same ring.
                    if b_atom.index in cycle:
                        c_arom = False
                        print("Warning: two resonant atoms "+
                              "%s and %s"%(b_atom.ciflabel, c_atom.ciflabel)+
                              "in the same ring have a bond order of 1.0! "
                              "This will likely yield unphysical characteristics"+
                              " of your system.")
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
        b_neigh = len(b_atom.neighbours) - 1
        c_neigh = len(c_atom.neighbours) - 1
        norm = float(b_neigh * c_neigh)
        V /= norm
        d = n*phi0 + 180
        # default is to include the full 1-4 non-bonded interactions.
        # but this breaks Lammps unless extra work-arounds are in place.
        # the weighting is added via a special_bonds keyword
        w = 0.0 
        dihedral.potential = DihedralPotential.Charmm()
        dihedral.potential.K = V/2.
        dihedral.potential.n = n
        dihedral.potential.d = d
        dihedral.potential.w = w

    def improper_term(self, improper):
        """

                a                        J
               /                        /
              /                        /
        c----b     , DREIDING =  K----I
              \                        \ 
               \                        \ 
                d                        L

        for all non-planar configurations, DREIDING uses

        E = 0.5*C*(cos(phi) - cos(phi0))^2

        For systems with planar equilibrium geometries, phi0 = 0
        E = K*[1 - cos(phi)]

        This is available in LAMMPS as the 'umbrella' improper potential.

        """
        
        a_atom, b_atom, c_atom, d_atom = improper.atoms
        btype = b_atom.force_field_type
        # special case: ignore N column 
        sp3_N = ["N_3", "P_3", "As3", "Sb3"]
        K = 40.0
        if b_atom.hybridization == "sp2" or b_atom.hybridizaton == "aromatic":
            K /= 3.
        if btype in sp3_N:
            return

        omega0 = DREIDING_DATA[btype][4]
        improper.potential = ImproperPotential.Umbrella()

        improper.potential.K = K
        improper.omega0 = omega0 

    def unique_pair_terms(self, nbpot="LJ", hbpot="morse"):
        """Include hydrogen bonding terms"""
        atom_types = sorted(list(self.unique_atom_types.keys()))
        electro_neg_atoms = ["N", "O", "F"]
        pair_count = 0
        for (i, j) in itertools.combinations_with_replacement(
                                     atom_types, 2):
            atomi = self.unique_atom_types[i]
            atomj = self.unique_atom_types[j]
            pair = PairTerm(atomi, atomj)
            self.pair_term(pair, nbpot)
            pair_count += 1
            self.unique_pair_types[pair_count] = pair
            # condition, two donors cannot form a hydrogen bond..
            # this might be too restrictive?
            if (atomi.h_bond_donor and 
                    (atomj.element in electro_neg_atoms) and 
                    (not atomj.h_bond_donor)):
                # get H__HB type
                htype = None
                for nn in atomi.neighbours:
                    at = self.structure.atoms[nn]
                    if at.force_field_type == "H__HB":
                        htype = at.ff_type_index
                pair2 = PairTerm(atomi, atomj)
                self.hbond_pair(pair2, hbpot, htype, flipped=False)
                pair_count += 1
                self.unique_pair_types[pair_count] = pair2
            if (atomj.h_bond_donor and 
                    (atomi.element in electro_neg_atoms) and
                    (not atomi.h_bond_donor)):
                # get H__HB type
                htype = None
                for nn in atomj.neighbours:
                    at = self.structure.atoms[nn]
                    if at.force_field_type == "H__HB":
                        htype = at.ff_type_index
                pair2 = PairTerm(atomi, atomj)
                self.hbond_pair(pair2, hbpot, htype, flipped=True)
                pair_count += 1
                self.unique_pair_types[pair_count] = pair2

    def hbond_pair(self, pair, nbpot, htype, flipped=False):
        """
        DREIDING can describe hbonded donor and acceptors
        using a lj function or a morse potential

        the morse potential is apparently better, so it
        will be default here
       
        DREIDING III h-bonding terms
        specified in 10.1021/ja8100227. 
        """
        if (nbpot == 'morse'):
            pair.potential = PairPotential.HbondDreidingMorse()
            pair.potential.htype = htype
            if(flipped):
                pair.potential.donor = 'j'

                atom1 = pair.atoms[1]
                atom2 = pair.atoms[0]
            else:
                atom1 = pair.atoms[0]
                atom2 = pair.atoms[1]
            ff1 = atom1.force_field_type
            ff2 = atom2.force_field_type
            D0 = 9.0
            R0 = 2.75
            # Table S3 of the SI of 10.1021/ja8100227 is poorly documented,
            # I have parameterized, to the best of my ability, what was intended
            # in that paper. This message posted on the lammps-users
            # message board http://lammps.sandia.gov/threads/msg36158.html
            # was helpful
            # N_3H == tertiary amine
            # N_3P == primary amine
            # N_3HP == protonated primary amine
            ineigh = []
            jneigh = []
            for n in atom1.neighbours:
                at = self.structure.atoms[n]
                ineigh.append(at.element)
            for n in atom2.neighbours:
                at = self.structure.atoms[n]
                jneigh.append(at.element)

            if(ff1 == "N_3"):
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
                    elif(ff2 == "O_2"):
                        D0 = 8.38
                        R0 = 2.77 
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
            else:
                # generic HB
                D0 = 9.5
                R0 = 2.75
            pair.potential.D0 = D0 
            pair.potential.alpha = 10.0/ 2. / R0
            pair.potential.R0 = R0
            pair.potential.n = 2
            # one can edit these values for bookkeeping.
            pair.potential.Rin = 9.0
            pair.potential.Rout = 11.0
            pair.potential.a_cut = 90.0

    def pair_term(self, pair, nbpot):
        """ DREIDING can adopt the exponential-6 or
        Ex6 = A*exp{-C*R} - B*R^{-6}

        the Lennard-Jones type interactions.
        Elj = A*R^{-12} - B*R^{-6}

        This will eventually be user-defined

        """

        atom1 = pair.atoms[0]
        atom2 = pair.atoms[1]


        eps1 = DREIDING_DATA[atom1.force_field_type][3]
        R1 = DREIDING_DATA[atom1.force_field_type][2]
        sig1 = R1*(2**(-1./6.))
        eps2 = DREIDING_DATA[atom2.force_field_type][3]
        R2 = DREIDING_DATA[atom2.force_field_type][2]
        sig2 = R2*(2**(-1./6.))

        if nbpot == "LJ":
            # default LB mixing.
            
            eps = np.sqrt(eps1*eps2)
            sig = (sig1 + sig2) / 2.
            pot = PairPotential.LjCutCoulLong()
            pot.eps = eps
            pot.sig = sig
            pot.cutoff = self.cutoff
            pair.potential = pot 

        else:
            S1 = DREIDING_DATA[atom1.force_field_type][5]
            S2 = DREIDING_DATA[atom2.force_field_type][5]

            A1 = eps1*(6./(S1 - 6.))*np.exp(S1)
            rho1 = R1
            C1 = eps1*(S1/(S1 - 6.)*R1**6)

            A2 = eps2*(6./(S2 - 6.))*np.exp(S2)
            rho2 = R2
            C2 = eps2*(S2/(S2 - 6.)*R2**6)

            pot = PairPotential.BuckLongCoulLong()
            pot.A = np.sqrt(A1*A2)
            pot.C = np.sqrt(C1*C2) 
            pot.rho = (rho1 + rho2)/2.
            pot.cutoff = self.cutoff
            pair.potential = pot 

    def special_commands(self):
        st = ""
        st += "%-15s %s\n"%("pair_modify", "tail yes")
        st += "%-15s %s\n"%("special_bonds", "dreiding") # equivalent to 'special_bonds lj 0.0 0.0 1.0'
        return st

    def detect_ff_terms(self):
        # for each atom determine the ff type if it is None
        organics = ["C", "N", "O", "S"]
        halides = ["F", "Cl", "Br", "I"]
        for atom in self.structure.atoms:

            if atom.force_field_type is None:
                if atom.element in organics:
                    if atom.hybridization == "sp3":
                        atom.force_field_type = "%s_3"%atom.element
                    elif atom.hybridization == "aromatic":
                        atom.force_field_type = "%s_R"%atom.element
                    elif atom.hybridization == "sp2":
                        atom.force_field_type = "%s_2"%atom.element
                    elif atom.hybridization == "sp":
                        atom.force_field_type = "%s_1"%atom.element
                    # default to sp3
                    else:
                        atom.force_field_type = "%s_3"%atom.element
                elif atom.element == "H":
                    atom.force_field_type = "H_"
                elif atom.element in halides:
                    atom.force_field_type = atom.element
                    if atom.element == "F":
                        atom.force_field_type += "_"
                else:
                    ffs = list(DREIDING_DATA.keys())
                    for j in ffs:
                        if atom.element == j[:2].strip("_"):
                            atom.force_field_type = j
            if atom.force_field_type is None:
                print(atom.hybridization)
                print("ERROR: could not find the proper force field type for atom %i"%(atom.index)+
                        " with element: '%s'"%(atom.element))
                sys.exit()

