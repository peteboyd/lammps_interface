#!/usr/bin/env python
"""
main.py

the program starts here.

"""

import sys
import math
import numpy as np
import networkx as nx
import ForceFields
import itertools
import operator
from structure_data import from_CIF, write_CIF, Structure
from CIFIO import CIF
from ccdc import CCDC_BOND_ORDERS
from datetime import datetime
from InputHandler import Options

class LammpsSimulation(object):
    def __init__(self, options):
        self.name = options.cif_file
        self.options = options
        self.molecules = []
        self.subgraphs = []
        self.unique_atom_types = {}
        self.unique_bond_types = {}
        self.unique_angle_types = {}
        self.unique_improper_types = {}
    
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
                self.unique_atom_types[type] = node 
            data['ff_type_index'] = type

    def unique_bonds(self):
        """Computes the number of unique bonds in the structure"""
        count = 0
        bb_type = {}
        for n1, n2, data in self.graph.edges_iter2(data=True):
            self.bond_term((n1, n2, data))

            btype = "%s"%data['potential']
            try:
                type = bb_type[btype]

            except KeyError:
                count += 1
                type = count
                bb_type[btype] = type

                self.unique_bond_types[type] = (n1, n2, data) 

            data['ff_type_index'] = type
    
    def unique_angles(self):
        ang_type = {}
        count = 0
        for b, data in self.graph.nodes_iter(data=True):
            # compute and store angle terms
            try:
                ang_data = data['angles']
                for (a, c), val in ang_data.items():
                    self.angle_term((a, b, c, val))
                    atype = "%s"%val['potential']
                    try:
                        type = ang_type[atype]

                    except KeyError:
                        count += 1
                        type = count
                        ang_type[atype] = type
                        self.unique_angle_types[type] = (a, b, c, val) 
                    val['ff_type_index'] = type
                    # update original dictionary
                    data['angles'][(a, c)] = val
            except KeyError:
                # no angle associated with this node.
                pass

    def unique_dihedrals(self):
        count = 0
        dihedral_type = {}
        for b, c, data in self.graph.edges_iter(data=True):
            try:
                dihed_data = data['dihedrals']
                for (a, d), val in dihed_data.items():
                    self.dihedral_term((a,b,c,d, val))
                    dtype = "%s"%val['potential']
                    try:
                        type = dihedral_type[dtype]
                    except KeyError:
                        count += 1 
                        type = count
                        dihedral_type[dtype] = type
                        self.unique_dihedral_types[type] = (a, b, c, d, val)
                    val['ff_type_index'] = type
                    # update original dictionary
                    data['dihedrals'][(a,d)] = val
            except KeyError:
                # no dihedrals associated with this edge
                pass

    def unique_impropers(self):
        count = 0
        improper_type = {}
        for b, data in self.graph.nodes_iter(data=True):
            try:
                imp_data = data['impropers']
                for (a, c, d), val in imp_data.items():
                    self.improper_term((a,b,c,d, val))
                    if val['potential'] is not None:
                        itype = "%s"%val['potential']
                        try:
                            type = improper_type[itype]
                        except KeyError:
                            count += 1
                            type = count
                            improper_type[itype] = type
                            self.unique_improper_types[type] = (a, b, c, d, val) 
                        val['ff_type_index'] = type
                        # update original dictionary
                        data['impropers'][(a, c, d)] = val
                    else:
                        data['impropers'].pop((a, c, d))
            except KeyError:
                # no improper terms associated with this atom
                pass

    def unique_pair_terms(self):
        """This is force field dependent."""
        return

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

    def set_graph(self, graph):
        self.graph = graph
        try:
            self.graph.compute_topology_information(self.cell)
        except AttributeError:
            # no cell set yet 
            pass

    def set_cell(self, cell):
        self.cell = cell
        try:
            self.graph.compute_topology_information(self.cell)
        except AttributeError:
            # no graph set yet
            pass

    def split_graph(self):

        self.compute_molecules()
        if (self.molecules): 
            print("Molecules found in the framework, separating.")
            for molecule in self.molecules:
                self.subgraphs.append(self.cut_molecule(molecule))
                # unwrap coordinates

    def assign_force_fields(self):

        try:
            getattr(ForceFields, self.options.force_field)(self.graph)
        except AttributeError:
            print("Error: could not find the force field: %s"%self.options.force_field)
            sys.exit()

    def compute_simulation_size(self):

        supercell = self.cell.minimum_supercell(self.options.cutoff)
        if np.any(np.array(supercell) > 1):
            print("Warning: unit cell is not large enough to"
                  +" support a non-bonded cutoff of %.2f Angstroms\n"%self.options.cutoff +
                   "Re-sizing to a %i x %i x %i supercell. "%(supercell))
            
            #TODO(pboyd): apply to subgraphs as well, if requested.
            self.graph.build_supercell(supercell, self.cell)
            for mgraph in self.subgraphs:
                mgraph.build_supercell(supercell, cell, track_molecule=True)
            self.cell.update_supercell(supercell)

    def compute_unique_atoms(self):
        #must ensure that graphs are merged at this point.
        pass

    def compute_unique_force_field_terms(self):
        #must ensure that graphs are merged at this point.
        pass

    def count_dihedrals(self):
        count = 0
        for edge, data in self.graph.edges_iter(data=True):
            try:
                for dihed in data['dihedrals'].keys():
                    count += 1
            except KeyError:
                pass
        return count

    def count_angles(self):
        count = 0
        for node, data in self.graph.nodes_iter(data=True):
            try:
                for angle in data['angles'].keys():
                    count += 1
            except KeyError:
                pass

    def count_impropers(self):
        count = 0
        for node, data in self.graph.nodes_iter(data=True):
            try:
                for angle in data['impropers'].keys():
                    count += 1
            except KeyError:
                pass

    def write_lammps_files(self):
        data_str = self.construct_data_file() 
        inp_str = self.construct_input_file()
   
        datafile = open("data.%s"%self.name, 'w')
        datafile.writelines(data_str)
        datafile.close()

        inpfile = open("in.%s"%self.name, 'w')
        inpfile.writelines(inp_str)
        inpfile.close()
        print("files created!")

    def construct_data_file(self):
    
        t = datetime.today()
        string = "Created on %s\n\n"%t.strftime("%a %b %d %H:%M:%S %Y %Z")
    
        if(len(ff.unique_atom_types.keys()) > 0):
            string += "%12i atoms\n"%(nx.number_of_nodes(self.graph))
        if(len(ff.unique_bond_types.keys()) > 0):
            string += "%12i bonds\n"%(nx.number_of_edges(self.graph))
        if(len(ff.unique_angle_types.keys()) > 0):
            string += "%12i angles\n"%(self.count_angles())
        if(len(ff.unique_dihedral_types.keys()) > 0):
            string += "%12i dihedrals\n"%(self.count_dihedrals())
        if (len(ff.unique_improper_types.keys()) > 0):
            string += "%12i impropers\n\n"%(self.count_impropers())
    
        if(len(ff.unique_atom_types.keys()) > 0):
            string += "%12i atom types\n"%(len(ff.unique_atom_types.keys()))
        if(len(ff.unique_bond_types.keys()) > 0):
            string += "%12i bond types\n"%(len(ff.unique_bond_types.keys()))
        if(len(ff.unique_angle_types.keys()) > 0):
            string += "%12i angle types\n"%(len(ff.unique_angle_types.keys()))
        if(len(ff.unique_dihedral_types.keys()) > 0):
            string += "%12i dihedral types\n"%(len(ff.unique_dihedral_types.keys()))
        if (len(ff.unique_improper_types.keys()) > 0):
            string += "%12i improper types\n"%(len(ff.unique_improper_types.keys()))
    
        string += "%19.6f %10.6f %s %s\n"%(0., self.cell.lx, "xlo", "xhi")
        string += "%19.6f %10.6f %s %s\n"%(0., self.cell.ly, "ylo", "yhi")
        string += "%19.6f %10.6f %s %s\n"%(0., self.cell.lz, "zlo", "zhi")
        if (np.any(np.array([self.cell.xy, self.cell.xz, self.cell.yz]) > 0.0)):
            string += "%19.6f %10.6f %10.6f %s %s %s\n"%(self.cell.xy, self.cell.xz, self.cell.yz, "xy", "xz", "yz")
    
        # Let's track the forcefield potentials that haven't been calc'd or user specified
        no_bond = []
        no_angle = []
        no_dihedral = []
        no_improper = []
        
        # this should be non-zero, but just in case..
        if(len(ff.unique_atom_types.keys()) > 0):
            string += "\nMasses\n\n"
            for key in sorted(ff.unique_atom_types.keys()):
                unq_atom = ff.unique_atom_types[key]
                mass, type = unq_atom.mass, unq_atom.force_field_type
                string += "%5i %8.4f # %s\n"%(key, mass, type)
    
        if(len(ff.unique_bond_types.keys()) > 0):
            string += "\nBond Coeffs\n\n"
            for key in sorted(ff.unique_bond_types.keys()):
                bond = ff.unique_bond_types[key]
                if bond.potential is None:
                    no_bond.append("%5i : %s %s"%(key, 
                                        bond.atoms[0].force_field_type, 
                                        bond.atoms[1].force_field_type))
                else:
                    ff1, ff2 = (bond.atoms[0].force_field_type, 
                                bond.atoms[1].force_field_type)
    
                    string += "%5i %s "%(key, bond.potential)
                    string += "# %s %s\n"%(ff1, ff2)
    
        class2angle = False
        if(len(ff.unique_angle_types.keys()) > 0):
            string += "\nAngle Coeffs\n\n"
            for key in sorted(ff.unique_angle_types.keys()):
                angle = ff.unique_angle_types[key]
                atom_a, atom_b, atom_c = angle.atoms
    
                if angle.potential is None:
                    no_angle.append("%5i : %s %s %s"%(key, 
                                          atom_a.force_field_type, 
                                          atom_b.force_field_type, 
                                          atom_c.force_field_type))
                else:
                    if (angle.potential.name == "class2"):
                        class2angle = True
    
                    string += "%5i %s "%(key, angle.potential)
                    string += "# %s %s %s\n"%(atom_a.force_field_type, 
                                              atom_b.force_field_type, 
                                              atom_c.force_field_type)
    
        if(class2angle):
            string += "\nBondBond Coeffs\n\n"
            for key in sorted(ff.unique_angle_types.keys()):
                angle = ff.unique_angle_types[key]
                atom_a, atom_b, atom_c = angle.atoms
                try:
                    string += "%5i %s "%(key, angle.potential.bb)
                    string += "# %s %s %s\n"%(atom_a.force_field_type, 
                                              atom_b.force_field_type, 
                                              atom_c.force_field_type)
                except AttributeError:
                    pass
        
            string += "\nBondAngle Coeffs\n\n"
            for key in sorted(ff.unique_angle_types.keys()):
                angle = ff.unique_angle_types[key]
                atom_a, atom_b, atom_c = angle.atoms
                try:
                    string += "%5i %s "%(key, angle.potential.ba)
                    string += "# %s %s %s\n"%(atom_a.force_field_type, 
                                              atom_b.force_field_type, 
                                              atom_c.force_field_type)
                except AttributeError:
                    pass   
    
        class2dihed = False
        if(len(ff.unique_dihedral_types.keys()) > 0):
            string +=  "\nDihedral Coeffs\n\n"
            for key in sorted(ff.unique_dihedral_types.keys()):
                dihedral = ff.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = dihedral.atoms
                if dihedral.potential is None:
                    no_dihedral.append("%5i : %s %s %s %s"%(key, 
                                       atom_a.force_field_type, 
                                       atom_b.force_field_type, 
                                       atom_c.force_field_type, 
                                       atom_d.force_field_type))
                else:
                    if(dihedral.potential.name == "class2"):
                        class2dihed = True
                    string += "%5i %s "%(key, dihedral.potential)
                    string += "# %s %s %s %s\n"%(atom_a.force_field_type, 
                                                 atom_b.force_field_type, 
                                                 atom_c.force_field_type, 
                                                 atom_d.force_field_type)
    
        if (class2dihed):
            string += "\nMiddleBondTorsion Coeffs\n\n"
            for key in sorted(ff.unique_dihedral_types.keys()):
                dihedral = ff.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = dihedral.atoms
                try:
                    string += "%5i %s "%(key, dihedral.potential.mbt) 
                    string += "# %s %s %s %s\n"%(atom_a.force_field_type, 
                                              atom_b.force_field_type, 
                                              atom_c.force_field_type,
                                              atom_d.force_field_type)
                except AttributeError:
                    pass
            string += "\nEndBondTorsion Coeffs\n\n"
            for key in sorted(ff.unique_dihedral_types.keys()):
                dihedral = ff.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = dihedral.atoms
                try:
                    string += "%5i %s "%(key, dihedral.potential.ebt) 
                    string += "# %s %s %s %s\n"%(atom_a.force_field_type, 
                                              atom_b.force_field_type, 
                                              atom_c.force_field_type,
                                              atom_d.force_field_type)
                except AttributeError:
                    pass
            string += "\nAngleTorsion Coeffs\n\n"
            for key in sorted(ff.unique_dihedral_types.keys()):
                dihedral = ff.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = dihedral.atoms
                try:
                    string += "%5i %s "%(key, dihedral.potential.at) 
                    string += "# %s %s %s %s\n"%(atom_a.force_field_type, 
                                              atom_b.force_field_type, 
                                              atom_c.force_field_type,
                                              atom_d.force_field_type)
                except AttributeError:
                    pass
            string += "\nAngleAngleTorsion Coeffs\n\n"
            for key in sorted(ff.unique_dihedral_types.keys()):
                dihedral = ff.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = dihedral.atoms
                try:
                    string += "%5i %s "%(key, dihedral.potential.aat) 
                    string += "# %s %s %s %s\n"%(atom_a.force_field_type, 
                                              atom_b.force_field_type, 
                                              atom_c.force_field_type,
                                              atom_d.force_field_type)
                except AttributeError:
                    pass
            string += "\nBondBond13 Coeffs\n\n"
            for key in sorted(ff.unique_dihedral_types.keys()):
                dihedral = ff.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = dihedral.atoms
                try:
                    string += "%5i %s "%(key, dihedral.potential.bb13) 
                    string += "# %s %s %s %s\n"%(atom_a.force_field_type, 
                                                 atom_b.force_field_type, 
                                                 atom_c.force_field_type,
                                                 atom_d.force_field_type)
                except AttributeError:
                    pass
        
        
        class2improper = False 
        if (len(ff.unique_improper_types.keys()) > 0):
            string += "\nImproper Coeffs\n\n"
            for key in sorted(ff.unique_improper_types.keys()):
                improper = ff.unique_improper_types[key]
                atom_a, atom_b, atom_c, atom_d = improper.atoms  
                if improper.potential is None:
                    no_improper.append("%5i : %s %s %s %s"%(key, 
                        atom_a.force_field_type, 
                        atom_b.force_field_type, 
                        atom_c.force_field_type, 
                        atom_d.force_field_type))
                else:
                    if(improper.potential.name == "class2"):
                        class2improper = True
                    string += "%5i %s "%(key, improper.potential)
                    string += "# %s %s %s %s\n"%(atom_a.force_field_type, 
                                                 atom_b.force_field_type, 
                                                 atom_c.force_field_type, 
                                                 atom_d.force_field_type)
        if (class2improper):
            string += "\nAngleAngle Coeffs\n\n"
            for key in sorted(ff.unique_improper_types.keys()):
                improper = ff.unique_improper_types[key]
                atom_a, atom_b, atom_c, atom_d = improper.atoms 
                try:
                    string += "%5i %s "%(key, improper.potential.aa)
                    string += "# %s %s %s %s\n"%(atom_a.force_field_type, 
                                                 atom_b.force_field_type, 
                                                 atom_c.force_field_type, 
                                                 atom_d.force_field_type)
                except AttributeError:
                    pass
    
        if((len(ff.unique_pair_types.keys()) > 0) and (ff.pair_in_data)):
            string += "\nPair Coeffs\n\n"
            for key, pair in sorted(ff.unique_pair_types.items()):
                string += "%5i %s "%(key, pair.potential)
                string += "# %s %s\n"%(pair.atoms[0].force_field_type, 
                                       pair.atoms[1].force_field_type)
        
        # Nest this in an if statement
        if any([no_bond, no_angle, no_dihedral, no_improper]):
        # WARNING MESSAGE for potentials we think are unique but have not been calculated
            print("WARNING: The following unique bonds/angles/dihedrals/impropers" +
                    " were detected in your crystal")
            print("But they have not been assigned a potential from user_input.txt"+
                    " or from an internal FF assignment routine!")
            print("Bonds")
            for elem in no_bond:
                print(elem)
            print("Angles")
            for elem in no_angle:
                print(elem)
            print("Dihedrals")
            for elem in no_dihedral:
                print(elem)
            print("Impropers")
            for elem in no_improper:
                print(elem)
            print("If you think you specified one of these in your user_input.txt " +
                  "and this is an error, please contact developers\n")
            print("CONTINUING...")
    
    
        #************[atoms]************
    	# Added 1 to all atom, bond, angle, dihedral, improper indices (LAMMPS does not accept atom of index 0)
        if(len(ff.unique_atom_types.keys()) > 0):
            string += "\nAtoms\n\n"
            for atom in ff.structure.atoms:
                molid = 444
                string += "%8i %8i %8i %11.5f %10.5f %10.5f %10.5f\n"%(atom.index+1, 
                                                                       molid, 
                                                                       atom.ff_type_index,
                                                                       atom.charge,
                                                                       atom.coordinates[0], 
                                                                       atom.coordinates[1], 
                                                                       atom.coordinates[2])
    
        #************[bonds]************
        if(len(ff.unique_bond_types.keys()) > 0):
            string += "\nBonds\n\n"
            for bond in ff.structure.bonds:
                atm1, atm2 = bond.atoms 
                string += "%8i %8i %8i %8i\n"%(bond.index+1, 
                                               bond.ff_type_index, 
                                               atm1.index+1, 
                                               atm2.index+1)
    
        #************[angles]***********
        if(len(ff.unique_angle_types.keys()) > 0):
            string += "\nAngles\n\n"
            for angle in ff.structure.angles:
                atm1, atm2, atm3 = angle.atoms 
                # what order are they presented? b, a, c? or a, b, c?
                string += "%8i %8i %8i %8i %8i\n"%(angle.index+1, 
                                                   angle.ff_type_index, 
                                                   atm1.index+1, 
                                                   atm2.index+1, 
                                                   atm3.index+1)
    
        #************[dihedrals]********
        if(len(ff.unique_dihedral_types.keys()) > 0):
            string += "\nDihedrals\n\n"
            for dihedral in ff.structure.dihedrals:
                atm1, atm2, atm3, atm4 = dihedral.atoms 
                # order?
                string += "%8i %8i %8i %8i %8i %8i\n"%(dihedral.index+1, 
                                                       dihedral.ff_type_index, 
                                                       atm1.index+1, 
                                                       atm2.index+1,
                                                       atm3.index+1, 
                                                       atm4.index+1)
    
        #************[impropers]********
        if(len(ff.unique_improper_types.keys()) > 0):
            string += "\nImpropers\n\n"
            for improper in ff.structure.impropers:
                atm1, atm2, atm3, atm4 = improper.atoms
                # order?
                string += "%8i %8i %8i %8i %8i %8i\n"%(improper.index+1,
                                                       improper.ff_type_index,
                                                       atm1.index+1, 
                                                       atm2.index+1,
                                                       atm3.index+1,
                                                       atm4.index+1)
    
    
        return string
    
    def construct_input_file(self):
        """Input file will depend on what the user wants to do"""
    
        # Eventually, this function should be dependent on some command line arguments
        # which will dictate what kind of simulation to run in LAMMPS
        inp_str = ""
        inp_str += "%-15s %s\n"%("units","real")
        inp_str += "%-15s %s\n"%("atom_style","full")
        inp_str += "%-15s %s\n"%("boundary","p p p")
        inp_str += "%-15s %s\n"%("dielectric","1")
        inp_str += "\n"
        if(len(ff.unique_pair_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("pair_style", ff.pair_style)
        if(len(ff.unique_bond_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("bond_style", ff.bond_style)
        if(len(ff.unique_angle_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("angle_style", ff.angle_style)
        if(len(ff.unique_dihedral_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("dihedral_style", ff.dihedral_style)
        if(len(ff.unique_improper_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("improper_style", ff.improper_style)
        if(ff.kspace_style): 
            inp_str += "%-15s %s\n"%("kspace_style", ff.kspace_style) 
        inp_str += "\n"
    
        # general catch-all for extra force field commands needed.
        inp_str += ff.special_commands()
    
        inp_str += "%-15s %s\n"%("box tilt","large")
        inp_str += "%-15s %s\n"%("read_data","data.%s"%(ff.structure.name))
    
        if(not ff.pair_in_data):
            inp_str += "#### Pair Coefficients ####\n"
            for k, pair in ff.unique_pair_types.items():
                inp_str += "%-15s %-4i %-4i %s # %s %s\n"%("pair_coeff", 
                        pair.atoms[0].ff_type_index, pair.atoms[1].ff_type_index,
                        pair.potential, pair.atoms[0].force_field_type,
                        pair.atoms[1].force_field_type)
            
            inp_str += "#### END Pair Coefficients ####\n\n"
    
        if(ff.structure.molecules):
            inp_str += "#### Atom Groupings ####\n"
            idx = 1
            for molecule in ff.structure.molecules.keys():
                mols = [item for sublist in sorted(ff.structure.molecules[molecule]) for item in sublist]
                inc_idx=False
                if (molecule == "framework"):
                    inp_str += "%-15s %-8s %s  "%("group", "fram", "id")
                else:
                    inp_str += "%-15s %-8s %s  "%("group", "%i"%(idx), "id")
                    inc_idx = True
                for x in groups(mols):
                    x = list(x)
                    if(len(x)>1):
                        inp_str += " %i:%i"%(x[0]+1, x[-1]+1)
                    else:
                        inp_str += " %i"%(x[0]+1)
                inp_str += "\n"
    
                for idy, mol in enumerate(ff.structure.molecules[molecule]):
                    inp_str += "%-15s %-8s %s  "%("group", "%i-%i"%(idx, idy+1), "id")
                    for g in sorted(groups(mol)):
                        g = list(g)
                        if(len(g)>1):
                            inp_str += " %i:%i"%(g[0]+1, g[-1]+1)
                        else:
                            inp_str += " %i"%(g[0]+1)
                    inp_str += "\n"
                if inc_idx:
                    idx += 1
            inp_str += "#### END Atom Groupings ####\n"
    
    
    
        inp_str += "%-15s %s\n"%("dump","%s_mov all xyz 1 %s_mov.xyz"%
                            (ff.structure.name, ff.structure.name))
        inp_str += "%-15s %s\n"%("dump_modify", "%s_mov element %s"%(
                                 ff.structure.name, 
                                 " ".join([ff.unique_atom_types[key].element 
                                            for key in sorted(ff.unique_atom_types.keys())])))
        inp_str += "%-15s %s\n"%("min_style","cg")
        inp_str += "%-15s %s\n"%("minimize","1.0e-4 1.0e-4 10000 100000")
        inp_str += "%-15s %s\n"%("fix","1 all box/relax tri 0.0 vmax 0.01")
        inp_str += "%-15s %s\n"%("minimize","1.0e-4 1.0e-4 10000 100000")
        inp_str += "%-15s %s\n"%("unfix", "1")
        inp_str += "%-15s %s\n"%("minimize","1.0e-4 1.0e-4 10000 100000")
    
    #    inp_str += "thermo_style custom step temp etotal ebond eangle edihed eimp\n thermo 1 \n timestep 0.5 \n fix   2 all nvt temp 300.0 300  100\n run  50000"
        return inp_str
    
    def groups(self, ints):
        ints = sorted(ints)
        for k, g in itertools.groupby(enumerate(ints), lambda ix : ix[0]-ix[1]):
            yield list(map(operator.itemgetter(1), g))

    # this needs to be somewhere else.
    def compute_molecules(self, size_cutoff=0.5):
        """Ascertain if there are molecules within the porous structure"""
        for j in nx.connected_components(self.graph):
            # return a list of nodes of connected graphs (decisions to isolate them will come later)
            if(len(j) <= self.graph.original_size*size_cutoff):
                self.molecules.append(j)
    
    def cut_molecule(self, nodes):
        mgraph = self.graph.subgraph(nodes).copy()
        self.graph.remove_nodes_from(nodes)
        indices = np.array(nodes) - 1
        mgraph.coordinates = self.graph.coordinates[indices,:].copy()
        mgraph.sorted_edge_dict = self.graph.sorted_edge_dict.copy()
        mgraph.distance_matrix = self.graph.distance_matrix[indices, indices].copy()
        for n1, n2 in mgraph.edges_iter():
            try:
                val = self.graph.sorted_edge_dict.pop((n1, n2))
                mgraph.sorted_edge_dict.update({(n1, n2):val})
            except KeyError:
                print("something went wrong")
            try:
                val = self.graph.sorted_edge_dict.pop((n2, n1))
                mgraph.sorted_edge_dict.update({(n2,n1):val})
            except KeyError:
                print("something went wrong")
        return mgraph

def main():

    # command line parsing
    options = Options()
    sim = LammpsSimulation(options)
    cell, graph = from_CIF(options.cif_file)
    sim.set_cell(cell)
    sim.set_graph(graph)
    sim.split_graph()
    sim.assign_force_fields()
    sim.compute_simulation_size()
    if options.output_cif:
        print("CIF file requested. Exiting...")
        write_CIF(graph, cell)
        sys.exit()

if __name__ == "__main__": 
    main()

