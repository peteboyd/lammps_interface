#!/usr/bin/env python
"""
main.py

the program starts here.

"""
import sys
import math
import re
import numpy as np
import networkx as nx
from .import ForceFields
import itertools
import operator
from .structure_data import from_CIF, write_CIF, clean
from .structure_data import write_RASPA_CIF, write_RASPA_sim_files, MDMC_config
from .CIFIO import CIF
from .ccdc import CCDC_BOND_ORDERS
from datetime import datetime
from .InputHandler import Options
from copy import deepcopy
from .import Molecules
import matplotlib
import matplotlib.pyplot as plt
import os
from .structure_data import MolecularGraph
from .atomic import ATOMIC_NUMBER


matplotlib.use('Agg')
if sys.version_info < (3, 0):
    input = raw_input


class LammpsSimulation(object):
    def __init__(self, options):
        self.name = clean(options.cif_file)
        self.special_commands = []
        self.options = options
        self.molecules = []
        self.subgraphs = []
        self.molecule_types = {}
        self.unique_atom_types = {}
        self.unique_bond_types = {}
        self.unique_angle_types = {}
        self.unique_dihedral_types = {}
        self.unique_improper_types = {}
        self.unique_pair_types = {}
        self.pair_in_data = True
        self.separate_molecule_types = True
        self.framework = False # Flag if a framework exists in the simulation.
        self.type_molecules = {}
        self.no_molecule_pair = True  # ensure that h-bonding will not occur between molecules of the same type
        self.fix_shake = {}
        self.fix_rigid = {}

    def set_MDMC_config(self, MDMC_config):
        self.MDMC_config = MDMC_config

    def unique_atoms(self):
        """Computes the number of unique atoms in the structure"""
        count = 0
        ff_type = {}
        fwk_nodes = sorted(self.graph.nodes())
        molecule_nodes = []
        for k in sorted(self.molecule_types.keys()):
            nds = []
            for m in self.molecule_types[k]:

                jnodes = sorted(self.subgraphs[m].nodes())
                nds += jnodes

                for n in jnodes:
                    del fwk_nodes[fwk_nodes.index(n)]
            molecule_nodes.append(nds)
        molecule_nodes.append(fwk_nodes)

        for node, data in self.graph.nodes_iter2(data=True):
            # add factor for h_bond donors
            if self.separate_molecule_types:
                molid = [j for j,mol in enumerate(molecule_nodes) if node in mol]
                if(len(molid) != 1):
                    print("ERROR!")
                molid = molid[0]
            else:
                molid = 0

            if data['force_field_type'] is None:
                if data['h_bond_donor']:
                    # add neighbors to signify type of hbond donor
                    label = (data['element'], data['h_bond_donor'], molid, tuple(sorted([self.graph.node[j]['element'] for j in self.graph.neighbors(node)])))
                else:
                    label = (data['element'], data['h_bond_donor'], molid)
            else:
                if data['h_bond_donor']:
                    # add neighbors to signify type of hbond donor
                    label = (data['force_field_type'], data['h_bond_donor'], molid, tuple(sorted([self.graph.node[j]['element'] for j in self.graph.neighbors(node)])))
                else:
                    label = (data['force_field_type'], data['h_bond_donor'], molid)

            try:
                type = ff_type[label]
            except KeyError:
                count += 1
                type = count
                ff_type[label] = type
                self.unique_atom_types[type] = node
                self.type_molecules[type] = molid
            data['ff_type_index'] = type

    def unique_bonds(self):
        """Computes the number of unique bonds in the structure"""
        count = 0
        bb_type = {}
        for n1, n2, data in self.graph.edges_iter2(data=True):
            btype = "%s"%data['potential']

            try:
                type = bb_type[btype]

            except KeyError:
                try:
                    if data['potential'].special_flag == 'shake':
                        self.fix_shake.setdefault('bonds', []).append(count+1)
                except AttributeError:
                    pass
                count += 1
                type = count
                bb_type[btype] = type

                self.unique_bond_types[type] = (n1, n2, data)

            data['ff_type_index'] = type

    def unique_angles(self):
        ang_type = {}
        count = 0
        for b, data in self.graph.nodes_iter2(data=True):
            # compute and store angle terms
            try:
                ang_data = data['angles']

                for (a, c), val in ang_data.items():
                    atype = "%s"%val['potential']
                    try:
                        type = ang_type[atype]

                    except KeyError:
                        count += 1
                        try:
                            if val['potential'].special_flag == 'shake':
                                self.fix_shake.setdefault('angles', []).append(count)
                        except AttributeError:
                            pass
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
        for b, c, data in self.graph.edges_iter2(data=True):
            try:
                dihed_data = data['dihedrals']
                for (a, d), val in dihed_data.items():
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

        for b, data in self.graph.nodes_iter2(data=True):
            try:
                rem = []
                imp_data = data['impropers']
                for (a, c, d), val in imp_data.items():
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
                    else:
                        rem.append((a,c,d))

                for m in rem:
                    data['impropers'].pop(m)

            except KeyError:
                # no improper terms associated with this atom
                pass

    def unique_pair_terms(self):
        pot_names = []
        nodes_list = sorted(self.unique_atom_types.keys())
        electro_neg_atoms = ["N", "O", "F"]
        for n, data in self.graph.nodes_iter2(data=True):
            if data['h_bond_donor']:
                pot_names.append('h_bonding')
            if data['tabulated_potential']:
                pot_names.append('table')
            pot_names.append(data['pair_potential'].name)
        # mix yourself

        table_str = ""
        if len(list(set(pot_names))) > 1 or (any(['buck' in i for i in list(set(pot_names))])):
            self.pair_in_data = False
            for (i, j) in itertools.combinations_with_replacement(nodes_list, 2):
                n1, n2 = self.unique_atom_types[i], self.unique_atom_types[j]
                i_data = self.graph.node[n1]
                j_data = self.graph.node[n2]
                mol1 = self.type_molecules[i]
                mol2 = self.type_molecules[j]
                # test to see if h-bonding to occur between molecules
                pairwise_test = ((mol1 != mol2 and self.no_molecule_pair) or (not self.no_molecule_pair))
                if i_data['tabulated_potential'] and j_data['tabulated_potential']:
                    table_pot = deepcopy(i_data)
                    table_str += table_pot['table_function'](i_data,j_data, table_pot)
                    table_pot['table_potential'].filename = "table." + self.name
                    self.unique_pair_types[(i, j, 'table')] = table_pot

                if (i_data['h_bond_donor'] and j_data['element'] in electro_neg_atoms and pairwise_test and not j_data['h_bond_donor']):
                    hdata = deepcopy(i_data)
                    hdata['h_bond_potential'] = hdata['h_bond_function'](n2, self.graph, flipped=False)
                    hdata['tabulated_potential'] = False
                    self.unique_pair_types[(i,j,'hb')] = hdata
                if (j_data['h_bond_donor'] and i_data['element'] in electro_neg_atoms and pairwise_test and not i_data['h_bond_donor']):
                    hdata = deepcopy(j_data)
                    hdata['tabulated_potential'] = False
                    hdata['h_bond_potential'] = hdata['h_bond_function'](n1, self.graph, flipped=True)
                    self.unique_pair_types[(i,j,'hb')] = hdata
                # mix Lorentz-Berthelot rules
                pair_data = deepcopy(i_data)
                if 'buck' in i_data['pair_potential'].name and 'buck' in j_data['pair_potential'].name:
                    eps1 = i_data['pair_potential'].eps
                    eps2 = j_data['pair_potential'].eps
                    sig1 = i_data['pair_potential'].sig
                    sig2 = j_data['pair_potential'].sig
                    eps = np.sqrt(eps1*eps2)
                    Rv = (sig1 + sig2)
                    Rho = Rv/12.0
                    A = 1.84e5 * eps
                    C=2.25*(Rv)**6*eps

                    pair_data['pair_potential'].A = A
                    pair_data['pair_potential'].rho = Rho
                    pair_data['pair_potential'].C = C
                    pair_data['tabulated_potential'] = False
                    # assuming i_data has the same pair_potential name as j_data
                    self.unique_pair_types[(i,j, i_data['pair_potential'].name)] = pair_data
                elif 'lj' in i_data['pair_potential'].name and 'lj' in j_data['pair_potential'].name:

                    pair_data['pair_potential'].eps = np.sqrt(i_data['pair_potential'].eps*j_data['pair_potential'].eps)
                    pair_data['pair_potential'].sig = (i_data['pair_potential'].sig + j_data['pair_potential'].sig)/2.
                    pair_data['tabulated_potential'] = False
                    self.unique_pair_types[(i,j, i_data['pair_potential'].name)] = pair_data

        # can be mixed by lammps
        else:
            for b in sorted(list(self.unique_atom_types.keys())):
                data = self.graph.node[self.unique_atom_types[b]]
                # compute and store angle terms
                pot = data['pair_potential']
                self.unique_pair_types[b] = data

        if (table_str):
            f = open('table.'+self.name, 'w')
            f.writelines(table_str)
            f.close()
        return

    def define_styles(self):
        # should be more robust, some of the styles require multiple parameters specified on these lines
        self.kspace_style = "ewald %f"%(0.000001)
        bonds = set([j['potential'].name for n1, n2, j in list(self.unique_bond_types.values())])
        if len(list(bonds)) > 1:
            self.bond_style = "hybrid %s"%" ".join(list(bonds))
        else:
            self.bond_style = "%s"%list(bonds)[0]
            for n1, n2, b in list(self.unique_bond_types.values()):
                b['potential'].reduced = True

        angles = set([j['potential'].name for a,b,c,j in list(self.unique_angle_types.values())])
        if len(list(angles)) > 1:
            self.angle_style = "hybrid %s"%" ".join(list(angles))
        else:
            self.angle_style = "%s"%list(angles)[0]
            for a,b,c,ang in list(self.unique_angle_types.values()):
                ang['potential'].reduced = True
                if (ang['potential'].name == "class2"):
                    ang['potential'].bb.reduced=True
                    ang['potential'].ba.reduced=True


        dihedrals = set([j['potential'].name for a,b,c,d,j in list(self.unique_dihedral_types.values())])
        if len(list(dihedrals)) > 1:
            self.dihedral_style = "hybrid %s"%" ".join(list(dihedrals))
        else:
            self.dihedral_style = "%s"%list(dihedrals)[0]
            for a,b,c,d, di in list(self.unique_dihedral_types.values()):
                di['potential'].reduced = True
                if (di['potential'].name == "class2"):
                    di['potential'].mbt.reduced=True
                    di['potential'].ebt.reduced=True
                    di['potential'].at.reduced=True
                    di['potential'].aat.reduced=True
                    di['potential'].bb13.reduced=True

        impropers = set([j['potential'].name for a,b,c,d,j in list(self.unique_improper_types.values())])
        if len(list(impropers)) > 1:
            self.improper_style = "hybrid %s"%" ".join(list(impropers))
        elif len(list(impropers)) == 1:
            self.improper_style = "%s"%list(impropers)[0]
            for a,b,c,d,i in list(self.unique_improper_types.values()):
                i['potential'].reduced = True
                if (i['potential'].name == "class2"):
                    i['potential'].aa.reduced=True
        else:
            self.improper_style = ""
        pairs = set(["%r"%(j['pair_potential']) for j in list(self.unique_pair_types.values())]) | \
                set(["%r"%(j['h_bond_potential']) for j in list(self.unique_pair_types.values()) if j['h_bond_potential'] is not None]) | \
                set(["%r"%(j['table_potential']) for j in list(self.unique_pair_types.values()) if j['tabulated_potential']])
        if len(list(pairs)) > 1:
            self.pair_style = "hybrid/overlay %s"%(" ".join(list(pairs)))
        else:
            self.pair_style = "%s"%list(pairs)[0]
            for p in list(self.unique_pair_types.values()):
                p['pair_potential'].reduced = True

    def set_graph(self, graph):
        self.graph = graph

        try:
            if(not self.options.force_field == "UFF") and (not self.options.force_field == "Dreiding"):
                self.graph.find_metal_sbus = True # true for UFF4MOF, BTW_FF and Dubbeldam
            if (self.options.force_field == "Dubbeldam"):
                self.graph.find_organic_sbus = True
            self.graph.compute_topology_information(self.cell, self.options.tol, self.options.neighbour_size)
        except AttributeError:
            # no cell set yet
            pass

    def set_cell(self, cell):
        self.cell = cell
        try:
            self.graph.compute_topology_information(self.cell, self.options.tol, self.options.neighbour_size)
        except AttributeError:
            # no graph set yet
            pass

    def split_graph(self):

        self.compute_molecules()
        if (self.molecules):
            print("Molecules found in the framework, separating.")
            molid=0
            for molecule in self.molecules:
                molid += 1
                sg = self.cut_molecule(molecule)
                sg.molecule_id = molid
                # unwrap coordinates
                sg.unwrap_node_coordinates(self.cell)
                self.subgraphs.append(sg)
        type = 0
        temp_types = {}
        for i, j in itertools.combinations(range(len(self.subgraphs)), 2):
            if self.subgraphs[i].number_of_nodes() != self.subgraphs[j].number_of_nodes():
                continue

            matched = self.subgraphs[i] | self.subgraphs[j]
            if (len(matched) == self.subgraphs[i].number_of_nodes()):
                if i not in list(temp_types.keys()) and j not in list(temp_types.keys()):
                    type += 1
                    temp_types[i] = type
                    temp_types[j] = type
                    self.molecule_types.setdefault(type, []).append(i)
                    self.molecule_types[type].append(j)
                else:
                    try:
                        type = temp_types[i]
                        temp_types[j] = type
                    except KeyError:
                        type = temp_types[j]
                        temp_types[i] = type
                    if i not in self.molecule_types[type]:
                        self.molecule_types[type].append(i)
                    if j not in self.molecule_types[type]:
                        self.molecule_types[type].append(j)
        unassigned = set(range(len(self.subgraphs))) - set(list(temp_types.keys()))
        for j in list(unassigned):
            type += 1
            self.molecule_types[type] = [j]

    def assign_force_fields(self):

        attr = {'graph':self.graph, 'cutoff':self.options.cutoff, 'h_bonding':self.options.h_bonding,
                'keep_metal_geometry':self.options.fix_metal, 'bondtype':self.options.dreid_bond_type}
        param = getattr(ForceFields, self.options.force_field)(**attr)

        self.special_commands += param.special_commands()

        # apply different force fields.
        for mtype in list(self.molecule_types.keys()):
            # prompt for ForceField?
            rep = self.subgraphs[self.molecule_types[mtype][0]]
            #response = input("Would you like to apply a new force field to molecule type %i with atoms (%s)? [y/n]: "%
            #        (mtype, ", ".join([rep.node[j]['element'] for j in rep.nodes()])))
            #ff = self.options.force_field
            #if response.lower() in ['y','yes']:
            #    ff = input("Please enter the name of the force field: ")
            #elif response.lower() in ['n', 'no']:
            #    pass
            #else:
            #    print("Unrecognized command: %s"%response)

            ff = self.options.mol_ff
            if ff is None:
                ff = self.options.force_field
                atoms = ", ".join([rep.node[j]['element'] for j in rep.nodes()])
                print("Warning: Molecule %s with atoms (%s) will be using the %s force field as no "%(mtype,atoms,ff)+
                      " value was set for molecules. To prevent this warning "+
                      "set --molecule-ff=[some force field] on the command line.")
            h_bonding = False
            if (ff == "Dreiding"):
                hbonding = input("Would you like this molecule type to have hydrogen donor potentials? [y/n]: ")
                if hbonding.lower() in ['y', 'yes']:
                    h_bonding = True
                elif hbonding.lower() in ['n', 'no']:
                    h_bonding = False
                else:
                    print("Unrecognized command: %s"%hbonding)
                    sys.exit()
            for m in self.molecule_types[mtype]:
                # Water check
                # currently only works on bare water without dummy atoms.
                ngraph = self.subgraphs[m]
                self.assign_molecule_ids(ngraph)
                if ff[-5:] == "Water":
                    self.add_water_model(ngraph, ff)
                    ff = ff[:-6] # remove _Water from end of name
                p = getattr(ForceFields, ff)(graph=self.subgraphs[m],
                                         cutoff=self.options.cutoff,
                                         h_bonding=h_bonding)
                self.special_commands += p.special_commands()

    def assign_molecule_ids(self, graph):
        for node in graph.nodes():
            graph.node[node]['molid'] = graph.molecule_id

    def molecule_template(self, mol):
        """ Construct a molecule template for
        reading and insertions in a LAMMPS simulation.

        Not sure how the bonding, angle, dihedral, improper,
        and pair terms will be dealt with yet..

        """

        #I think the Molecule class should be generalized so that
        #this kind of input can be generated easily
        molecule = getattr(Molecule, mol)()


    def add_water_model(self, ngraph, ff):
        size = ngraph.number_of_nodes()
        if size < 3 or size > 3:
            print("Error: cannot assign %s "%(ff) +
                  "to molecule of size %i, with "%(size)+
                  "atoms (%s)"%(", ".join([ngraph.node[kk]['element'] for
                                           kk in ngraph.nodes()])))
            print("If this is a water molecule with pre-existing "+
                    "dummy atoms for a particular force field, "+
                    "please remove them and re-run this code.")
            sys.exit()
        for node in ngraph.nodes():
            if ngraph.node[node]['element'] == "O":
                oid = node
                oatom = ngraph.node[node]
            elif ngraph.node[node]['element'] == "H":
                try:
                    hatom1
                    h2id = node
                    hatom2 = ngraph.node[node]
                except NameError:
                    h1id = node
                    hatom1 = ngraph.node[node]

        h2o = getattr(Molecules, ff)()
        h2o.approximate_positions(O_pos  = oatom['cartesian_coordinates'],
                                  H_pos1 = hatom1['cartesian_coordinates'],
                                  H_pos2 = hatom2['cartesian_coordinates'])
        # replace the current H positions with the force-field assigned
        # ones
        oatom['mass'] = h2o.O_mass
        oatom['force_field_type'] = "OW"
        hatom1['cartesian_coordinates'] = h2o.H_coord[0]
        hatom1['mass'] = h2o.H_mass
        hatom1['force_field_type'] = "HW"
        hatom2['cartesian_coordinates'] = h2o.H_coord[1]
        hatom2['mass'] = h2o.H_mass
        hatom2['force_field_type'] = "HW"
        for j in h2o.dummy:
            # increment graph size
            self.increment_graph_sizes()
            os = ngraph.original_size
            args = {'element': 'X',
                    'force_field_type': 'X',
                    'cartesian_coordinates': j,
                    'potential': None,
                    'rings': [],
                    'molid': ngraph.molecule_id,
                    'atomic_number': 0,
                    'h_bond_donor': False,
                    'h_bond_potential': None,
                    'tabulated_potential': False,
                    'table_potential': None,
                    'pair_potential': None
                    }
            ngraph.add_node(os, **args)
            ngraph.add_edge(oid, os, order=1.,
                            weight=1.,
                            length=h2o.Rdum,
                            symflag='1_555',
                            )
            ngraph.sorted_edge_dict.update({(oid, os): (oid, os)})
            ngraph.sorted_edge_dict.update({(os, oid): (oid, os)})

        # compute new angles between dummy atoms
        ngraph.compute_angles()


    def increment_graph_sizes(self, inc=1):
        self.graph.original_size += inc
        for mtype in list(self.molecule_types.keys()):
            for m in self.molecule_types[mtype]:
                graph = self.subgraphs[m]
                graph.original_size += 1

    def compute_simulation_size(self):

        supercell = self.cell.minimum_supercell(self.options.cutoff)
        if np.any(np.array(supercell) > 1):
            print("Warning: unit cell is not large enough to"
                  +" support a non-bonded cutoff of %.2f Angstroms."%self.options.cutoff)

        if(self.options.replication is not None):
            supercell = tuple(map(int, re.split('x| |, |,',self.options.replication)))
            if(len(supercell) != 3):
                if(supercell[0] < 1 or supercell[1] < 1 or supercell[2] < 1):
                    print("Incorrect supercell requested: %s\n"%(supercell))
                    print("Use <ixjxk> format")
                    print("Exiting...")
                    sys.exit()

        if np.any(np.array(supercell) > 1):
            print("Re-sizing to a %i x %i x %i supercell. "%(supercell))

            #TODO(pboyd): apply to subgraphs as well, if requested.
            self.graph.build_supercell(supercell, self.cell)
            molcount = 0
            if self.subgraphs:
                molcount = max([g.molecule_id for g in self.subgraphs])

            for mtype in list(self.molecule_types.keys()):
                # prompt for replication of this molecule in the supercell.
                rep = self.subgraphs[self.molecule_types[mtype][0]]
                response = input("Would you like to replicate molceule %i with atoms (%s) in the supercell? [y/n]: "%
                        (mtype, ", ".join([rep.node[j]['element'] for j in rep.nodes()])))
                if response in ['y', 'Y', 'yes']:
                    for m in self.molecule_types[mtype]:
                        self.subgraphs[m].build_supercell(supercell, self.cell, track_molecule=True, molecule_len=molcount)
            self.cell.update_supercell(supercell)

    def compute_cluster_box_size(self):
        """
        Added by MW b/c simbox size can need to be  even bigger than a normal
        simbox when we are making a cluster from one unit cell
        """

        supercell = self.cell.minimum_supercell(self.options.cutoff)
        # we really need a 3x3x3 grid of supercells to 100% ensure we get all components of cluster accurately
        supercell = (supercell[0]+2, supercell[1]+2, supercell[2]+2)
        self.supercell_tuple = (supercell[0], supercell[1], supercell[2])

        if np.any(np.array(supercell) > 1):
            print("Warning: unit cell is not large enough to"
                  +" support a non-bonded cutoff of %.2f Angstroms\n"%self.options.cutoff +
                   "Re-sizing to a %i x %i x %i supercell. "%(supercell))

            #TODO(pboyd): apply to subgraphs as well, if requested.
            self.graph.build_supercell(supercell, self.cell)
            for mtype in list(self.molecule_types.keys()):
                # prompt for replication of this molecule in the supercell.
                rep = self.subgraphs[self.molecule_types[mtype][0]]
                response = input("Would you like to replicate moleule %i with atoms (%s) in the supercell? [y/n]: "%
                        (mtype, ", ".join([rep.node[j]['element'] for j in rep.nodes()])))
                if response in ['y', 'Y', 'yes']:
                    for m in self.molecule_types[mtype]:
                        self.subgraphs[m].build_supercell(supercell, self.cell, track_molecule=True)
            self.cell.update_supercell(supercell)

    def count_dihedrals(self):
        count = 0
        for n1, n2, data in self.graph.edges_iter(data=True):
            try:
                for dihed in data['dihedrals'].keys():
                    count += 1
            except KeyError:
                pass
        return count

    def count_angles(self):
        count = 0
        for node, data in self.graph.nodes_iter2(data=True):
            try:
                for angle in data['angles'].keys():
                    count += 1
            except KeyError:
                pass
        return count

    def count_impropers(self):
        count = 0
        for node, data in self.graph.nodes_iter2(data=True):
            try:
                for angle in data['impropers'].keys():
                    count += 1
            except KeyError:
                pass
        return count

    def merge_graphs(self):
        for mgraph in self.subgraphs:
            self.graph += mgraph
        if sorted(self.graph.nodes()) != [i+1 for i in range(len(self.graph.nodes()))]:
            print("Re-labelling atom indices.")
            reorder_dic = {i:j+1 for i, j in zip(sorted(self.graph.nodes()), range(len(self.graph.nodes())))}
            self.graph.reorder_labels(reorder_dic)
            for mgraph in self.subgraphs:
                mgraph.reorder_labels(reorder_dic)

    def write_lammps_files(self):
        self.unique_atoms()
        self.unique_bonds()
        self.unique_angles()
        self.unique_dihedrals()
        self.unique_impropers()
        self.unique_pair_terms()
        self.define_styles()

        data_str = self.construct_data_file()
        datafile = open("data.%s"%self.name, 'w')
        datafile.writelines(data_str)
        datafile.close()

        inp_str = self.construct_input_file()
        inpfile = open("in.%s"%self.name, 'w')
        inpfile.writelines(inp_str)
        inpfile.close()
        print("files created!")

    def construct_data_file(self):

        t = datetime.today()
        string = "Created on %s\n\n"%t.strftime("%a %b %d %H:%M:%S %Y %Z")

        if(len(self.unique_atom_types.keys()) > 0):
            string += "%12i atoms\n"%(nx.number_of_nodes(self.graph))
        if(len(self.unique_bond_types.keys()) > 0):
            string += "%12i bonds\n"%(nx.number_of_edges(self.graph))
        if(len(self.unique_angle_types.keys()) > 0):
            string += "%12i angles\n"%(self.count_angles())
        if(len(self.unique_dihedral_types.keys()) > 0):
            string += "%12i dihedrals\n"%(self.count_dihedrals())
        if (len(self.unique_improper_types.keys()) > 0):
            string += "%12i impropers\n"%(self.count_impropers())

        if(len(self.unique_atom_types.keys()) > 0):
            string += "\n%12i atom types\n"%(len(self.unique_atom_types.keys()))
        if(len(self.unique_bond_types.keys()) > 0):
            string += "%12i bond types\n"%(len(self.unique_bond_types.keys()))
        if(len(self.unique_angle_types.keys()) > 0):
            string += "%12i angle types\n"%(len(self.unique_angle_types.keys()))
        if(len(self.unique_dihedral_types.keys()) > 0):
            string += "%12i dihedral types\n"%(len(self.unique_dihedral_types.keys()))
        if (len(self.unique_improper_types.keys()) > 0):
            string += "%12i improper types\n"%(len(self.unique_improper_types.keys()))

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
        if(len(self.unique_atom_types.keys()) > 0):
            string += "\nMasses\n\n"
            for key in sorted(self.unique_atom_types.keys()):
                unq_atom = self.graph.node[self.unique_atom_types[key]]
                mass, type = unq_atom['mass'], unq_atom['force_field_type']
                string += "%5i %15.9f # %s\n"%(key, mass, type)

        if(len(self.unique_bond_types.keys()) > 0):
            string += "\nBond Coeffs\n\n"
            for key in sorted(self.unique_bond_types.keys()):
                n1, n2, bond = self.unique_bond_types[key]
                atom1, atom2 = self.graph.node[n1], self.graph.node[n2]
                if bond['potential'] is None:
                    no_bond.append("%5i : %s %s"%(key,
                                                  atom1['force_field_type'],
                                                  atom2['force_field_type']))
                else:
                    ff1, ff2 = (atom1['force_field_type'],
                                atom2['force_field_type'])

                    string += "%5i %s "%(key, bond['potential'])
                    string += "# %s %s\n"%(ff1, ff2)

        class2angle = False
        if(len(self.unique_angle_types.keys()) > 0):
            string += "\nAngle Coeffs\n\n"
            for key in sorted(self.unique_angle_types.keys()):
                a, b, c, angle = self.unique_angle_types[key]
                atom_a, atom_b, atom_c = self.graph.node[a], \
                                         self.graph.node[b], \
                                         self.graph.node[c]

                if angle['potential'] is None:
                    no_angle.append("%5i : %s %s %s"%(key,
                                          atom_a['force_field_type'],
                                          atom_b['force_field_type'],
                                          atom_c['force_field_type']))
                else:
                    if (angle['potential'].name == "class2"):
                        class2angle = True

                    string += "%5i %s "%(key, angle['potential'])
                    string += "# %s %s %s\n"%(atom_a['force_field_type'],
                                              atom_b['force_field_type'],
                                              atom_c['force_field_type'])

        if(class2angle):
            string += "\nBondBond Coeffs\n\n"
            for key in sorted(self.unique_angle_types.keys()):
                a, b, c, angle = self.unique_angle_types[key]
                atom_a, atom_b, atom_c = self.graph.node[a], \
                                         self.graph.node[b], \
                                         self.graph.node[c]
                if (angle['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s\n"%(atom_a['force_field_type'],
                                              atom_b['force_field_type'],
                                              atom_c['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, angle['potential'].bb)
                        string += "# %s %s %s\n"%(atom_a['force_field_type'],
                                                  atom_b['force_field_type'],
                                                  atom_c['force_field_type'])
                    except AttributeError:
                        pass

            string += "\nBondAngle Coeffs\n\n"
            for key in sorted(self.unique_angle_types.keys()):
                a, b, c, angle = self.unique_angle_types[key]
                atom_a, atom_b, atom_c = self.graph.node[a],\
                                         self.graph.node[b],\
                                         self.graph.node[c]
                if (angle['potential'].name!="class2"):
                    string += "%5i skip  "%(key)
                    string += "# %s %s %s\n"%(atom_a['force_field_type'],
                                              atom_b['force_field_type'],
                                              atom_c['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, angle['potential'].ba)
                        string += "# %s %s %s\n"%(atom_a['force_field_type'],
                                                  atom_b['force_field_type'],
                                                  atom_c['force_field_type'])
                    except AttributeError:
                        pass

        class2dihed = False
        if(len(self.unique_dihedral_types.keys()) > 0):
            string +=  "\nDihedral Coeffs\n\n"
            for key in sorted(self.unique_dihedral_types.keys()):
                a, b, c, d, dihedral = self.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]
                if dihedral['potential'] is None:
                    no_dihedral.append("%5i : %s %s %s %s"%(key,
                                       atom_a['force_field_type'],
                                       atom_b['force_field_type'],
                                       atom_c['force_field_type'],
                                       atom_d['force_field_type']))
                else:
                    if(dihedral['potential'].name == "class2"):
                        class2dihed = True
                    string += "%5i %s "%(key, dihedral['potential'])
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                                 atom_b['force_field_type'],
                                                 atom_c['force_field_type'],
                                                 atom_d['force_field_type'])

        if (class2dihed):
            string += "\nMiddleBondTorsion Coeffs\n\n"
            for key in sorted(self.unique_dihedral_types.keys()):
                a, b, c, d, dihedral = self.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]

                if (dihedral['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                              atom_b['force_field_type'],
                                              atom_c['force_field_type'],
                                              atom_d['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, dihedral['potential'].mbt)
                        string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                                  atom_b['force_field_type'],
                                                  atom_c['force_field_type'],
                                                  atom_d['force_field_type'])
                    except AttributeError:
                        pass
            string += "\nEndBondTorsion Coeffs\n\n"
            for key in sorted(self.unique_dihedral_types.keys()):
                a, b, c, d, dihedral = self.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]
                if (dihedral['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                              atom_b['force_field_type'],
                                              atom_c['force_field_type'],
                                              atom_d['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, dihedral['potential'].ebt)
                        string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                                  atom_b['force_field_type'],
                                                  atom_c['force_field_type'],
                                                  atom_d['force_field_type'])
                    except AttributeError:
                        pass
            string += "\nAngleTorsion Coeffs\n\n"
            for key in sorted(self.unique_dihedral_types.keys()):
                a, b, c, d, dihedral = self.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]
                if (dihedral['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                              atom_b['force_field_type'],
                                              atom_c['force_field_type'],
                                              atom_d['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, dihedral['potential'].at)
                        string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                                  atom_b['force_field_type'],
                                                  atom_c['force_field_type'],
                                                  atom_d['force_field_type'])
                    except AttributeError:
                        pass
            string += "\nAngleAngleTorsion Coeffs\n\n"
            for key in sorted(self.unique_dihedral_types.keys()):
                a, b, c, d, dihedral = self.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]
                if (dihedral['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                              atom_b['force_field_type'],
                                              atom_c['force_field_type'],
                                              atom_d['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, dihedral['potential'].aat)
                        string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                                  atom_b['force_field_type'],
                                                  atom_c['force_field_type'],
                                                  atom_d['force_field_type'])
                    except AttributeError:
                        pass
            string += "\nBondBond13 Coeffs\n\n"
            for key in sorted(self.unique_dihedral_types.keys()):
                a, b, c, d, dihedral = self.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]
                if (dihedral['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                              atom_b['force_field_type'],
                                              atom_c['force_field_type'],
                                              atom_d['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, dihedral['potential'].bb13)
                        string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                                     atom_b['force_field_type'],
                                                     atom_c['force_field_type'],
                                                     atom_d['force_field_type'])
                    except AttributeError:
                        pass


        class2improper = False
        if (len(self.unique_improper_types.keys()) > 0):
            string += "\nImproper Coeffs\n\n"
            for key in sorted(self.unique_improper_types.keys()):
                a, b, c, d, improper = self.unique_improper_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]

                if improper['potential'] is None:
                    no_improper.append("%5i : %s %s %s %s"%(key,
                        atom_a['force_field_type'],
                        atom_b['force_field_type'],
                        atom_c['force_field_type'],
                        atom_d['force_field_type']))
                else:
                    if(improper['potential'].name == "class2"):
                        class2improper = True
                    string += "%5i %s "%(key, improper['potential'])
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                                 atom_b['force_field_type'],
                                                 atom_c['force_field_type'],
                                                 atom_d['force_field_type'])
        if (class2improper):
            string += "\nAngleAngle Coeffs\n\n"
            for key in sorted(self.unique_improper_types.keys()):
                a, b, c, d, improper = self.unique_improper_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]
                if (improper['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                                 atom_b['force_field_type'],
                                                 atom_c['force_field_type'],
                                                 atom_d['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, improper['potential'].aa)
                        string += "# %s %s %s %s\n"%(atom_a['force_field_type'],
                                                     atom_b['force_field_type'],
                                                     atom_c['force_field_type'],
                                                     atom_d['force_field_type'])
                    except AttributeError:
                        pass

        if((len(self.unique_pair_types.keys()) > 0) and (self.pair_in_data)):
            string += "\nPair Coeffs\n\n"
            for key, n in sorted(self.unique_atom_types.items()):
                pair = self.graph.node[n]
                string += "%5i %s "%(key, pair['pair_potential'])
                string += "# %s %s\n"%(self.graph.node[n]['force_field_type'],
                                       self.graph.node[n]['force_field_type'])


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
        sorted_nodes = sorted(self.graph.nodes())
        if(len(self.unique_atom_types.keys()) > 0):
            string += "\nAtoms\n\n"
            for node in sorted_nodes:
                atom = self.graph.node[node]
                string += "%8i %8i %8i %11.5f %10.5f %10.5f %10.5f\n"%(node,
                                                                       atom['molid'],
                                                                       atom['ff_type_index'],
                                                                       atom['charge'],
                                                                       atom['cartesian_coordinates'][0],
                                                                       atom['cartesian_coordinates'][1],
                                                                       atom['cartesian_coordinates'][2])

        #************[bonds]************
        if(len(self.unique_bond_types.keys()) > 0):
            string += "\nBonds\n\n"
            idx = 0
            for n1, n2, bond in sorted(list(self.graph.edges_iter2(data=True))):
                idx += 1
                string += "%8i %8i %8i %8i\n"%(idx,
                                               bond['ff_type_index'],
                                               n1,
                                               n2)

        #************[angles]***********
        if(len(self.unique_angle_types.keys()) > 0):
            string += "\nAngles\n\n"
            idx = 0
            for node in sorted_nodes:
                atom = self.graph.node[node]
                try:
                    for (a, c), angle in list(atom['angles'].items()):
                        idx += 1
                        string += "%8i %8i %8i %8i %8i\n"%(idx,
                                                           angle['ff_type_index'],
                                                           a,
                                                           node,
                                                           c)
                except KeyError:
                    pass

        #************[dihedrals]********
        if(len(self.unique_dihedral_types.keys()) > 0):
            string += "\nDihedrals\n\n"
            idx = 0
            for n1, n2, data in sorted(list(self.graph.edges_iter2(data=True))):
                try:
                    for (a, d), dihedral in list(data['dihedrals'].items()):
                        idx+=1
                        string += "%8i %8i %8i %8i %8i %8i\n"%(idx,
                                                              dihedral['ff_type_index'],
                                                              a,
                                                              n1,
                                                              n2,
                                                              d)
                except KeyError:
                    pass
        #************[impropers]********
        if(len(self.unique_improper_types.keys()) > 0):
            string += "\nImpropers\n\n"
            idx = 0
            for node in sorted_nodes:
                atom = self.graph.node[node]
                try:
                    for (a, c, d), improper in list(atom['impropers'].items()):
                        idx += 1
                        string += "%8i %8i %8i %8i %8i %8i\n"%(idx,
                                                               improper['ff_type_index'],
                                                               a,
                                                               node,
                                                               c,
                                                               d)
                except KeyError:
                    pass

        return string
    def fixcount(self, count=[]):
        count.append(1)
        return (len(count))

    def construct_input_file(self):
        """Input file will depend on what the user wants to do"""
        inp_str = ""
        # Eventually, this function should be dependent on some command line arguments
        # which will dictate what kind of simulation to run in LAMMPS
        inp_str += "%-15s %s\n"%("log","log.%s append"%(self.name))
        inp_str += "%-15s %s\n"%("units","real")
        inp_str += "%-15s %s\n"%("atom_style","full")
        inp_str += "%-15s %s\n"%("boundary","p p p")
        inp_str += "\n"
        if(len(self.unique_pair_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("pair_style", self.pair_style)
        if(len(self.unique_bond_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("bond_style", self.bond_style)
        if(len(self.unique_angle_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("angle_style", self.angle_style)
        if(len(self.unique_dihedral_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("dihedral_style", self.dihedral_style)
        if(len(self.unique_improper_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("improper_style", self.improper_style)
        if(self.kspace_style):
            inp_str += "%-15s %s\n"%("kspace_style", self.kspace_style)
        inp_str += "\n"

        # general catch-all for extra force field commands needed.
        inp_str += "\n".join(list(set(self.special_commands)))
        inp_str += "\n"
        inp_str += "%-15s %s\n"%("box tilt","large")
        inp_str += "%-15s %s\n"%("read_data","data.%s"%(self.name))

        if(not self.pair_in_data):
            inp_str += "#### Pair Coefficients ####\n"
            for pair,data in sorted(self.unique_pair_types.items()):
                n1, n2 = self.unique_atom_types[pair[0]], self.unique_atom_types[pair[1]]
                try:
                    if pair[2] == 'hb':
                        inp_str += "%-15s %-4i %-4i %s # %s %s\n"%("pair_coeff",
                            pair[0], pair[1], data['h_bond_potential'],
                            self.graph.node[n1]['force_field_type'],
                            self.graph.node[n2]['force_field_type'])
                    elif pair[2] == 'table':
                        inp_str += "%-15s %-4i %-4i %s # %s %s\n"%("pair_coeff",
                            pair[0], pair[1], data['table_potential'],
                            self.graph.node[n1]['force_field_type'],
                            self.graph.node[n2]['force_field_type'])
                    else:
                        inp_str += "%-15s %-4i %-4i %s # %s %s\n"%("pair_coeff",
                            pair[0], pair[1], data['pair_potential'],
                            self.graph.node[n1]['force_field_type'],
                            self.graph.node[n2]['force_field_type'])
                except IndexError:
                    pass
            inp_str += "#### END Pair Coefficients ####\n\n"

        if(self.molecules)and(len(self.molecule_types.keys()) < 32):
            # lammps cannot handle more than 32 groups including 'all'
            total_count = 0
            for k,v in self.molecule_types.items():
                total_count += len(v)
            list_individual_molecules = True
            if total_count > 31:
                list_individual_molecules = False

            inp_str += "\n#### Atom Groupings ####\n"
            idx = 1
            framework_atoms = self.graph.nodes()
            for mtype in list(self.molecule_types.keys()):

                inp_str += "%-15s %-8s %s  "%("group", "%i"%(mtype), "id")
                all_atoms = []
                for j in self.molecule_types[mtype]:
                    all_atoms += self.subgraphs[j].nodes()
                for x in self.groups(all_atoms):
                    x = list(x)
                    if(len(x)>1):
                        inp_str += " %i:%i"%(x[0], x[-1])
                    else:
                        inp_str += " %i"%(x[0])
                inp_str += "\n"
                for atom in reversed(sorted(all_atoms)):
                    del framework_atoms[framework_atoms.index(atom)]
                mcount = 0
                if list_individual_molecules:
                    for j in self.molecule_types[mtype]:
                        if (self.subgraphs[j].molecule_images):
                            for molecule in self.subgraphs[j].molecule_images:
                                mcount += 1
                                inp_str += "%-15s %-8s %s  "%("group", "%i-%i"%(mtype, mcount), "id")
                                for x in self.groups(molecule):
                                    x = list(x)
                                    if(len(x)>1):
                                        inp_str += " %i:%i"%(x[0], x[-1])
                                    else:
                                        inp_str += " %i"%(x[0])
                                inp_str += "\n"
                        elif len(self.molecule_types[mtype]) > 1:
                            mcount += 1
                            inp_str += "%-15s %-8s %s  "%("group", "%i-%i"%(mtype, mcount), "id")
                            molecule = self.subgraphs[j].nodes()
                            for x in self.groups(molecule):
                                x = list(x)
                                if(len(x)>1):
                                    inp_str += " %i:%i"%(x[0], x[-1])
                                else:
                                    inp_str += " %i"%(x[0])
                            inp_str += "\n"

            if(framework_atoms):
                self.framework = True
                inp_str += "%-15s %-8s %s  "%("group", "fram", "id")
                for x in self.groups(framework_atoms):
                    x = list(x)
                    if(len(x)>1):
                        inp_str += " %i:%i"%(x[0], x[-1])
                    else:
                        inp_str += " %i"%(x[0])
                inp_str += "\n"
            inp_str += "#### END Atom Groupings ####\n\n"

        if self.options.dump_dcd:
            inp_str += "%-15s %s\n"%("dump","%s_dcdmov all dcd 10 %s_mov.dcd"%
                            (self.name, self.name))
        elif self.options.dump_xyz:
            inp_str += "%-15s %s\n"%("dump","%s_xyzmov all xyz 1 %s_mov.xyz"%
                                (self.name, self.name))
            inp_str += "%-15s %s\n"%("dump_modify", "%s_xyzmov element %s"%(
                                     self.name,
                                     " ".join([self.graph.node[self.unique_atom_types[key]]['element']
                                                for key in sorted(self.unique_atom_types.keys())])))



        if (self.options.minimize):
            box_min = "tri"
            min_style="cg"
            nmins = 2
            #inp_str += "%-15s %s\n"%("min_style","fire")
            #inp_str += "%-15s %i %s\n"%("compute", 1, "all msd com yes")
            #inp_str += "%-15s %-10s %s\n"%("variable", "Dx", "equal c_1[1]")
            #inp_str += "%-15s %-10s %s\n"%("variable", "Dy", "equal c_1[2]")
            #inp_str += "%-15s %-10s %s\n"%("variable", "Dz", "equal c_1[3]")
            #inp_str += "%-15s %-10s %s\n"%("variable", "MSD", "equal c_1[4]")
            #inp_str += "%-15s %s %s\n"%("fix", "output all print 1", "\"$(vol),$(cella),$(cellb),$(cellc),${Dx},${Dy},${Dz},${MSD}\"" +
            #                                " file %s.min.csv title \"Vol,CellA,CellB,CellC,Dx,Dy,Dz,MSD\" screen no"%(self.name))

            inp_str += "%-15s %s\n"%("min_style", min_style)
            inp_str += "%-15s %s\n"%("minimize","1.0e-15 1.0e-15 10000 100000")
            #inp_str += "%-15s %i\n"%("run", 1)

            for j in range(nmins):
                fix = self.fixcount()
                #inp_str += "\n%-15s %s\n"%("min_style", min_style)
                inp_str += "%-15s %s\n"%("fix","%i all box/relax %s 0.0 vmax 0.01"%(fix, box_min))
                inp_str += "%-15s %s\n"%("minimize","1.0e-15 1.0e-15 10000 100000")
                inp_str += "%-15s %s\n"%("unfix", "%i"%fix)

                #inp_str += "%-15s %s\n"%("min_style","fire")
                inp_str += "%-15s %s\n"%("minimize","1.0e-15 1.0e-15 10000 100000")
                #inp_str += "%-15s %i\n"%("run", 1)

           # inp_str += "%-15s %s\n"%("unfix", "output")
        # delete bond types etc, for molecules that are rigid
        for mol in sorted(self.molecule_types.keys()):
            rep = self.subgraphs[self.molecule_types[mol][0]]
            if rep.rigid:
                inp_str += "%-15s %s\n"%("neigh_modify", "exclude molecule %i"%(mol))
                # find and delete all bonds, angles, dihedrals, and impropers associated
                # with this molecule, as they will consume unnecessary amounts of CPU time
                inp_str += "%-15s %i %s\n"%("delete_bonds", mol, "multi remove")


        if (self.fix_shake):
            shake_tol = 0.0001
            iterations = 20
            print_every = 0  # maybe set to non-zero, but output files could become huge.
            shk_fix = self.fixcount()
            shake_str = "b "+" ".join(["%i"%i for i in self.fix_shake['bonds']]) + \
                        " a " + " ".join(["%i"%i for i in self.fix_shake['angles']])
                       # fix  id group tolerance iterations print_every [bonds + angles]
            inp_str += "%-15s %i %s %s %f %i %i %s\n"%('fix', shk_fix, 'all', 'shake', shake_tol, iterations, print_every, shake_str)

        if (self.options.random_vel):
            inp_str += "%-15s %s\n"%("velocity", "all create %.2f %i"%(self.options.temp, np.random.randint(1,3000000)))

        if (self.options.nvt):
            inp_str += "%-15s %-10s %s\n"%("variable", "dt", "equal %.2f"%(1.0))
            inp_str += "%-15s %-10s %s\n"%("variable", "tdamp", "equal 100*${dt}")
            molecule_fixes = []
            mollist = sorted(list(self.molecule_types.keys()))
            for molid in mollist:
                id = self.fixcount()
                molecule_fixes.append(id)
                rep = self.subgraphs[self.molecule_types[molid][0]]
                if(rep.rigid):
                    inp_str += "%-15s %s\n"%("fix", "%i %s rigid/small molecule langevin %.2f %.2f ${tdamp} %i"%(id,
                                                                                            str(molid),
                                                                                            self.options.temp,
                                                                                            self.options.temp,
                                                                                            np.random.randint(1,3000000)
                                                                                            ))
                else:
                    inp_str += "%-15s %s\n"%("fix", "%i %s langevin %.2f %.2f ${tdamp} %i"%(id,
                                                                                        str(molid),
                                                                                        self.options.temp,
                                                                                        self.options.temp,
                                                                                        np.random.randint(1,3000000)
                                                                                        ))
                    id = self.fixcount()
                    molecule_fixes.append(id)
                    inp_str += "%-15s %s\n"%("fix", "%i %i nve"%(id,molid))
            if self.framework:
                id = self.fixcount()
                molecule_fixes.append(id)
                inp_str += "%-15s %s\n"%("fix", "%i %s langevin %.2f %.2f ${tdamp} %i"%(id,
                                                                                        "fram",
                                                                                        self.options.temp,
                                                                                        self.options.temp,
                                                                                        np.random.randint(1,3000000)
                                                                                        ))
                id = self.fixcount()
                molecule_fixes.append(id)
                inp_str += "%-15s %s\n"%("fix", "%i fram nve"%id)
            inp_str += "%-15s %i\n"%("thermo", 0)
            inp_str += "%-15s %i\n"%("run", self.options.neqstp)
            while(molecule_fixes):
                fid = molecule_fixes.pop(0)
                inp_str += "%-15s %i\n"%("unfix", fid)

            for molid in mollist:
                id = self.fixcount()
                molecule_fixes.append(id)
                rep = self.subgraphs[self.molecule_types[molid][0]]
                if(rep.rigid):
                    inp_str += "%-15s %s\n"%("fix", "%i %s rigid/nvt/small molecule temp %.2f %.2f ${tdamp}"%(id,
                                                                                            str(molid),
                                                                                            self.options.temp,
                                                                                            self.options.temp
                                                                                            ))
                else:
                    inp_str += "%-15s %s\n"%("fix", "%i %s nvt temp %.2f %.2f ${tdamp}"%(id,
                                                                                   str(molid),
                                                                                   self.options.temp,
                                                                                   self.options.temp
                                                                                   ))
            if self.framework:
                id = self.fixcount()
                molecule_fixes.append(id)
                inp_str += "%-15s %s\n"%("fix", "%i %s nvt temp %.2f %.2f ${tdamp}"%(id,
                                                                                   "fram",
                                                                                   self.options.temp,
                                                                                   self.options.temp
                                                                                   ))

            inp_str += "%-15s %i\n"%("thermo", 1)
            inp_str += "%-15s %i\n"%("run", self.options.nprodstp)

            while(molecule_fixes):
                fid = molecule_fixes.pop(0)
                inp_str += "%-15s %i\n"%("unfix", fid)

        if (self.options.npt):
            id = self.fixcount()
            inp_str += "%-15s %-10s %s\n"%("variable", "dt", "equal %.2f"%(1.0))
            inp_str += "%-15s %-10s %s\n"%("variable", "pdamp", "equal 1000*${dt}")
            inp_str += "%-15s %-10s %s\n"%("variable", "tdamp", "equal 100*${dt}")

            inp_str += "%-15s %s\n"%("fix", "%i all npt temp %.2f %.2f ${tdamp} tri %.2f %.2f ${pdamp}"%(id, self.options.temp, self.options.temp,
                                                                                                        self.options.pressure, self.options.pressure))
            inp_str += "%-15s %i\n"%("thermo", 0)
            inp_str += "%-15s %i\n"%("run", self.options.neqstp)
            inp_str += "%-15s %i\n"%("thermo", 1)
            inp_str += "%-15s %i\n"%("run", self.options.nprodstp)

            inp_str += "%-15s %i\n"%("unfix", id)

        if(self.options.bulk_moduli):
            min_style=True
            thermo_style=False

            inp_str += "\n%-15s %s\n"%("dump", "str all atom 1 initial_structure.dump")
            inp_str += "%-15s\n"%("run 0")
            inp_str += "%-15s %-10s %s\n"%("variable", "rs", "equal step")
            inp_str += "%-15s %-10s %s\n"%("variable", "readstep", "equal ${rs}")
            inp_str += "%-15s %-10s %s\n"%("variable", "rs", "delete")
            inp_str += "%-15s %s\n"%("undump", "str")

            inp_str += "\n%-15s %-10s %s\n"%("variable", "simTemp", "equal %.4f"%(self.options.temp))
            inp_str += "%-15s %-10s %s\n"%("variable", "dt", "equal %.2f"%(1.0))
            inp_str += "%-15s %-10s %s\n"%("variable", "tdamp", "equal 100*${dt}")

            inp_str += "%-15s %-10s %s\n"%("variable", "at", "equal cella")
            inp_str += "%-15s %-10s %s\n"%("variable", "bt", "equal cellb")
            inp_str += "%-15s %-10s %s\n"%("variable", "ct", "equal cellc")
            inp_str += "%-15s %-10s %s\n"%("variable", "a", "equal ${at}")
            inp_str += "%-15s %-10s %s\n"%("variable", "b", "equal ${bt}")
            inp_str += "%-15s %-10s %s\n"%("variable", "c", "equal ${ct}")
            inp_str += "%-15s %-10s %s\n"%("variable", "at", "delete")
            inp_str += "%-15s %-10s %s\n"%("variable", "bt", "delete")
            inp_str += "%-15s %-10s %s\n"%("variable", "ct", "delete")

            inp_str += "%-15s %-10s %s\n"%("variable", "N", "equal %i"%self.options.iter_count)
            inp_str += "%-15s %-10s %s\n"%("variable", "totDev", "equal %.5f"%self.options.max_dev)
            inp_str += "%-15s %-10s %s\n"%("variable", "sf", "equal ${totDev}/${N}*2")
            inp_str += "%-15s %s\n"%("print", "\"Loop,CellScale,Vol,Pressure,E_total,E_pot,E_kin" +
                                              ",E_bond,E_angle,E_torsion,E_imp,E_vdw,E_coul\""+
                                              " file %s.output.csv screen no"%(self.name))
            inp_str += "%-15s %-10s %s\n"%("variable", "do", "loop ${N}")
            inp_str += "%-15s %s\n"%("label", "loop")
            inp_str += "%-15s %s\n"%("read_dump", "initial_structure.dump ${readstep} x y z box yes format native")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleVar", "equal 1.00-${totDev}+${do}*${sf}")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleA", "equal ${scaleVar}*${a}")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleB", "equal ${scaleVar}*${b}")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleC", "equal ${scaleVar}*${c}")
            inp_str += "%-15s %s\n"%("change_box", "all x final 0.0 ${scaleA} y final 0.0 ${scaleB} z final 0.0 ${scaleC} remap")
            if (min_style):
                inp_str += "%-15s %s\n"%("min_style","fire")
                inp_str += "%-15s %s\n"%("minimize", "1.0e-15 1.0e-15 10000 100000")
                inp_str += "%-15s %s\n"%("print", "\"${do},${scaleVar},$(vol),$(press),$(etotal),$(pe),$(ke)"+
                                              ",$(ebond),$(eangle),$(edihed),$(eimp),$(evdwl),$(ecoul)\""+
                                              " append %s.output.csv screen no"%(self.name))
            elif (thermo_style):
                inp_str += "%-15s %s\n"%("velocity", "all create ${simTemp} %i"%(np.random.randint(1,3000000)))
                inp_str += "%-15s %s %s %s \n"%("fix", "bm", "all nvt", "temp ${simTemp} ${simTemp} ${tdamp} tchain 5")
                inp_str += "%-15s %i\n"%("run", self.options.neqstp)
                #inp_str += "%-15s %s\n"%("print", "\"STEP ${do} ${scaleVar} $(vol) $(press) $(etotal)\"")
                inp_str += "%-15s %s %s\n"%("fix", "output all print 10", "\"${do},${scaleVar},$(vol),$(press),$(etotal),$(pe),$(ke)" +
                                            ",$(ebond),$(eangle),$(edihed),$(eimp),$(evdwl),$(ecoul)\""+
                                            " append %s.output.csv screen no"%(self.name))
                inp_str += "%-15s %i\n"%("run", self.options.nprodstp)
                inp_str += "%-15s %s\n"%("unfix", "output")
                inp_str += "%-15s %s\n"%("unfix", "bm")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleVar", "delete")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleA", "delete")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleB", "delete")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleC", "delete")
            inp_str += "%-15s %s\n"%("next", "do")
            inp_str += "%-15s %s\n"%("jump", "SELF loop")
            inp_str += "%-15s %s\n"%("label", "break")
            inp_str += "%-15s %-10s %s\n"%("variable", "do", "delete")

        if (self.options.thermal_scaling):
            temperature = self.options.temp # kelvin
            equil_steps = self.options.neqstp
            prod_steps = self.options.nprodstp
            temprange = np.linspace(temperature, self.options.max_dev, self.options.iter_count).tolist()
            temprange.append(298.0)
            temprange.insert(0,1.0) # add 1 and 298 K simulations.

            inp_str += "\n%-15s %s\n"%("dump", "str all atom 1 initial_structure.dump")
            inp_str += "%-15s\n"%("run 0")
            inp_str += "%-15s %-10s %s\n"%("variable", "rs", "equal step")
            inp_str += "%-15s %-10s %s\n"%("variable", "readstep", "equal ${rs}")
            inp_str += "%-15s %-10s %s\n"%("variable", "rs", "delete")
            inp_str += "%-15s %s\n"%("undump", "str")

            inp_str += "%-15s %-10s %s\n"%("variable", "sim_temp", "index %s"%(" ".join(["%.2f"%i for i in sorted(temprange)])))
            inp_str += "%-15s %-10s %s\n"%("variable", "sim_press", "equal %.3f"%self.options.pressure) # atmospheres.
            #inp_str += "%-15s %-10s %s\n"%("variable", "a", "equal cella")
            #inp_str += "%-15s %-10s %s\n"%("variable", "myVol", "equal vol")
            #inp_str += "%-15s %-10s %s\n"%("variable", "t", "equal temp")
            # timestep in femtoseconds
            inp_str += "%-15s %-10s %s\n"%("variable", "dt", "equal %.2f"%(1.0))
            inp_str += "%-15s %-10s %s\n"%("variable", "pdamp", "equal 1000*${dt}")
            inp_str += "%-15s %-10s %s\n"%("variable", "tdamp", "equal 100*${dt}")
            inp_str += "%-15s %s\n"%("print", "\"Step,Temp,CellA,Vol\" file %s.output.csv screen no"%(self.name))
            inp_str += "%-15s %s\n"%("label", "loop")
            #fix1 = self.fixcount()

            inp_str += "%-15s %s\n"%("read_dump", "initial_structure.dump ${readstep} x y z box yes format native")
            inp_str += "%-15s %s\n"%("thermo_style", "custom step temp cella cellb cellc vol etotal")

            # the ave/time fix must be after read_dump, or the averages are reported as '0'
            #inp_str += "%-15s %s\n"%("fix", "%i all ave/time 1 %i %i v_t v_a v_myVol ave one"%(fix1, prod_steps,
            #                                                                                   prod_steps + equil_steps))
            id = self.fixcount()
            # creating velocity may cause instability at high temperatures.
            inp_str += "%-15s %s\n"%("velocity", "all create 50 %i"%(np.random.randint(1,3000000)))
            inp_str += "%-15s %i %s %s %s %s\n"%("fix", id,
                                        "all npt",
                                        "temp ${sim_temp} ${sim_temp} ${tdamp}",
                                        "tri ${sim_press} ${sim_press} ${pdamp}",
                                        "tchain 5 pchain 5")
            inp_str += "%-15s %i\n"%("thermo", 0)
            inp_str += "%-15s %i\n"%("run", equil_steps)
            inp_str += "%-15s %s %s\n"%("fix", "output all print 10", "\"${sim_temp},$(temp),$(cella),$(vol)\"" +
                                        " append %s.output.csv screen no"%(self.name))
            #inp_str += "%-15s %i\n"%("thermo", 10)
            inp_str += "%-15s %i\n"%("run", prod_steps)
            inp_str += "%-15s %s\n"%("unfix", "output")
            #inp_str += "\n%-15s %-10s %s\n"%("variable", "inst_t", "equal f_%i[1]"%(fix1))
            #inp_str += "%-15s %-10s %s\n"%("variable", "inst_a", "equal f_%i[2]"%(fix1))
            #inp_str += "%-15s %-10s %s\n"%("variable", "inst_v", "equal f_%i[3]"%(fix1))

            #inp_str += "%-15s %-10s %s\n"%("variable", "inst_t", "delete")
            #inp_str += "%-15s %-10s %s\n"%("variable", "inst_a", "delete")
            #inp_str += "%-15s %-10s %s\n\n"%("variable", "inst_v", "delete")
            inp_str += "%-15s %i\n"%("unfix", id)
            #inp_str += "%-15s %i\n"%("unfix", fix1)
            inp_str += "\n%-15s %s\n"%("next", "sim_temp")
            inp_str += "%-15s %s\n"%("jump", "SELF loop")
            inp_str += "%-15s %s\n"%("label", "break")
            inp_str += "%-15s %-10s %s\n"%("variable", "sim_temp", "delete")

        if self.options.dump_dcd:
            inp_str += "%-15s %s\n"%("undump", "%s_dcdmov"%(self.name))
        if self.options.dump_xyz:
            inp_str += "%-15s %s\n"%("undump", "%s_xyzmov"%(self.name))

        if self.options.restart:
            # for restart files we move xlo, ylo, zlo back to 0 so to have same origin as a cif file
            # also we modify to have unscaled coords so we can directly compute scaled coordinates WITH CIF BASIS
            inp_str += "\n# Dump last snapshot for restart\n"

            inp_str += "variable curr_lx equal lx\n"
            inp_str += "variable curr_ly equal ly\n"
            inp_str += "variable curr_lz equal lz\n"
            inp_str += "change_box all x final 0 ${curr_lx} y final 0 ${curr_ly} z final 0 ${curr_lz}\n\n"
            inp_str += "reset_timestep 0\n"
            inp_str += "%-15s %s\n"%("dump","%s_restart all atom 1 %s_restart.lammpstrj"%
                            (self.name, self.name))
            inp_str += "%-15s %s_restart scale no sort id\n"%("dump_modify",self.name)
            inp_str += "run 0\n"
            inp_str += "%-15s %s\n"%("undump", "%s_restart"%(self.name))

            # write a string that tells you how to read the dump file for this structure
            f=open("dump_restart_string.txt","w")
            f.write("read_dump %s_restart.lammpstrj %d x y z box yes"%(self.name,
                                                                       0))
            f.close()

        try:
            inp_str += "%-15s %i\n"%("unfix", shk_fix)
        except NameError:
            # no shake fix id in this input file.
            pass
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
            if(len(j) <= self.graph.original_size*size_cutoff) or (len(j) < 25):
                self.molecules.append(j)

    def cut_molecule(self, nodes):
        mgraph = self.graph.subgraph(nodes).copy()
        self.graph.remove_nodes_from(nodes)
        indices = np.array(list(nodes))
        indices -= 1
        mgraph.coordinates = self.graph.coordinates[indices,:].copy()
        mgraph.sorted_edge_dict = self.graph.sorted_edge_dict.copy()
        mgraph.distance_matrix = self.graph.distance_matrix.copy()
        mgraph.original_size = self.graph.original_size
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


class Cluster(object):

    def __init__(self, mgraph, xyz, offset, rcut):

        # the original graph from the super super box
        self.origraph = mgraph.copy()

        # a copied graph that we can disconnect and edit as desired
        self.disgraph = mgraph.copy()

        # The assigned center of the cluster, the size of the cluster,
        # and how far the cluster was offset from the original center
        # after simbox replications
        self.xyz = xyz
        self.rcut = rcut
        self.offset = offset

        # Index of the node closest to the assigned center of the cluster
        self.start_index = -1

        # all nodes that must be kept bc they are within the cutoff radius
        self.kept_nodes = set()
        self.num_keep = 0

        # dict of hydrogens to add later on
        self.hydrogens = {}

        # set of all metals that can exist in nanoprous materials
        self.metals = set([4,12,13,14,20,21,22,23,24,25,26,27,28,29,30,37,38,39,40,41,42,43,44,45,46,47,48])


    def cart_dist(self, pts1, pts2):
        """
        Cartesian distance between 2 points
        """
        return np.linalg.norm(pts1 - pts2)

    def parse_sym_flag_for_directionality(self, string):
        """
        symm flag looks 'like 1_455'
        where 4 denotes a periodic bond in the x direction
        where 5 denotes a non periodic bond in the y, z direction
        """
        ambiguous = False
        directionality = -1
        for i in range(3):
            if(string[2+i]!='5'):
                if(directionality == -1):
                    directionality = int(i)
                elif(directionality != -1 and directionality != int(i)):
                    ambiguous = True
                    return -1

        return directionality

    def get_start_and_kept_nodes(self):
        """
        Analyze a super box of a nanoporous material

        Determine which nodes are inside the cutoff and which nodes are outside

        Return the index of the starting node we will use to build the cluster
        """
        print("\n\nGETTING START NODE AND NODES INSIDE CUTOFF")
        print("--------------------------------------")

        min_dist = 100000000.0
        all_nodes = nx.number_of_nodes(self.origraph)
        for i in range(all_nodes):
            this_xyz = self.origraph.node[i+1]['cartesian_coordinates']
            cart_dist = self.cart_dist(this_xyz, self.xyz)

            if(cart_dist < min_dist):
                min_dist = float(cart_dist)
                self.start_index = int(i+1)

            if(cart_dist < self.rcut):
                self.kept_nodes.add(i+1)

        print("Start node: " + str(self.start_index))
        print("Sart node xyz: " + str(self.origraph.node[self.start_index]['cartesian_coordinates']))
        print("Num nodes inside rcut: " + str(len(self.kept_nodes)))

        return self.start_index

    def create_cluster_around_point_v3(self):
        """
        Basic strategy here is to disconnect the graph at valid truncations

        All the connected components are then condensed into a single node
        to form a secondary graph

        This is the best way to make sure that in the end we perform the capping
        properly
        """

        mat_type = self.identify_mat_type()
        start_index = self.get_start_and_kept_nodes()
        self.disconnect_external_building_blocks()
        one_D = self.identify_1D_building_blocks()
        if(one_D):
            print("1D rod identified")
            exit()
        else:
            # reset disgraph
            self.disgraph = self.origraph.copy()
            # identify all truncatable bonds
            self.identify_all_truncations()
            # remove all truncatable bonds from disgraph
            self.truncate_all()
            # create connected components from disgraph
            self.cxtd_comp_from_undirected()
            # id all cxtd comps with an atom inside the cutoff radius
            self.cxtd_comp_to_keep()
            # create a secondary graph out of all cxtd comps where the cxtd comp
            # is reduced to just one node
            self.cxtd_comp_secondary_graph()
            # add any additional cxtd comps to final cluster to make sure that
            # the final cluster is continuous (no disconnected comps)
            self.cxtd_comp_continuous()
            # count how many nodes from original structure are kept
            self.cxtd_comp_num_keep()
            # add hydrogens to all valid truncated bonds
            self.cxtd_comp_cap()
            # convert cxtd_comp to old style to write files easily b/c lazy for now
            self.cxtd_comp_convert_to_orig()
            # write all necessary files for DEC-MP2 calc
            self.write_cluster_to_xyz_host_guest_v2(include_guest=True)
            self.write_cluster_to_xyz_host_guest_v2(include_guest=False)
            self.write_cutoff()

            exit()


    def identify_all_truncations(self):
        """
        Identify all truncations to make in the graph
        """
        self.pot_truncs = set()
        self.pot_truncs_directed = set()
        self.pot_truncs_to_remove = set()
        added = 0
        for n1, n2, data in self.origraph.edges_iter2(data=True):
            if(self.truncation_criteria(n1, n2)):
                self.pot_truncs.add((n1, n2))

        print("Identified %s num of potential truncations" % (str(len(self.pot_truncs))))


    def cxtd_comp_from_undirected(self):
        """
        Find connected components of the disconnect graph
        """
        self.cxtd_comp = list(nx.connected_component_subgraphs(self.disgraph))
        print("Num of cxtd comps in disgarph: %d"%(len(self.cxtd_comp)))

    def cxtd_comp_to_keep(self):
        """
        Identify which connected components have an atom within the cutoff radius
        """

        self.cxtd_comp_to_keep = []
        self.cxtd_comp_primary = -1
        for i in range(len(self.cxtd_comp)):
            # get the component that contains the cluster origin (primary component)
            # ID primary component
            if(self.start_index in self.cxtd_comp[i]):
                self.cxtd_comp_primary = i

            # get all components with an atom inside the cutoff
            for node in self.cxtd_comp[i]:
                this_cart = self.origraph.node[node]['cartesian_coordinates']
                if(self.cart_dist(this_cart, self.xyz) < self.rcut):
                    self.cxtd_comp_to_keep.append(i)
                    break

        print("Num of cxtd comps to keep in disgraph: %d"%(len(self.cxtd_comp_to_keep)))
        print(self.cxtd_comp_to_keep)
        print([len(self.cxtd_comp[self.cxtd_comp_to_keep[i]]) for i in range(len(self.cxtd_comp_to_keep))])
        print("Primary comp is: %d" %(self.cxtd_comp_primary))

    def cxtd_comp_secondary_graph(self):
        """
        Create a secondary graph with all the connected components
        """
        self.secgraph = nx.Graph()
        self.secgraph.add_nodes_from(self.cxtd_comp)


        for edge in self.pot_truncs:
            n1 = edge[0]
            n2 = edge[1]

            for i in range(len(self.cxtd_comp)):
                if n1 in self.cxtd_comp[i]:
                    comp_edge_n1 = i
                if n2 in self.cxtd_comp[i]:
                    comp_edge_n2 = i

            # reverse edge will be taken care of because self.pot_truncs
            # has both forward and reverse edges
            self.secgraph.add_edge(comp_edge_n1, comp_edge_n2)

    def cxtd_comp_continuous(self):
        """
        This part is critical, make sure that the final list of connected components
        forms a continuous network in secondary graph

        In other words, the final cluster MUST not be a disconnected graph
        """
        print("Ensuring final cluster is continuous...")
        # any additional comps necessary to make sure final cluster is continuous
        add_comp = []

        for i in range(len(self.cxtd_comp_to_keep)):

            # get the shortest path between the primary component and each kept component
            this_path = nx.shortest_path(self.secgraph,source=self.cxtd_comp_to_keep[i],target=self.cxtd_comp_primary)

            # get mutually exclusive elements of the path
            ex = np.setxor1d(this_path, self.cxtd_comp_to_keep)

            # add any mutually exclusive element not already in the final list
            # of components that we are going to keep
            for comp in ex:
                if(comp not in self.cxtd_comp_to_keep):
                    print("Adding a cxtd comp to keep to make final cluster continuous: %d"%(comp))
                    self.cxtd_comp_to_keep.append(comp)
        print("Num of cxtd comps to keep in disgraph: %d"%(len(self.cxtd_comp_to_keep)))

    def cxtd_comp_num_keep(self):
        for i in range(len(self.cxtd_comp_to_keep)):
            self.num_keep += len(self.cxtd_comp[self.cxtd_comp_to_keep[i]])
        print("Num nodes kept from original structure: %d"%(self.num_keep))

    def cxtd_comp_cap(self):
        print("\n\nCAPPING 3D ORGANIC")
        print("--------------------------------------")

        all_nodes_to_keep = set()
        for component in self.cxtd_comp_to_keep:
            for node in self.cxtd_comp[component]:
                all_nodes_to_keep.add(node)

        for i in range(len(self.cxtd_comp_to_keep)):

            component = self.cxtd_comp[self.cxtd_comp_to_keep[i]]

            for node in component:

                #for nbr in self.tree.successors_iter(node):
                for nbr in self.origraph.neighbors(node):

                    #if((node, nbr) in self.actual_truncs_directed):
                    if(((node, nbr) in self.pot_truncs or \
                        (nbr, node) in self.pot_truncs) and \
                       nbr not in all_nodes_to_keep):
                       #((nbr, node) in self.actual_truncs_directed and \
                       # node not in all_nodes_to_keep)):

                       # NOTE this makes sure we don't place a cap somewhere
                       # that constitutes an atom we want to keep


                        print((node, nbr))

                        #print("Capping edge: " + str((node,nbr)))
                        bond_start = self.origraph.node[node]['cartesian_coordinates']
                        bond_end =    self.origraph.node[nbr]['cartesian_coordinates']
                        start_type = self.origraph.node[node]['atomic_number']
                        end_type = self.origraph.node[nbr]['atomic_number']

                        bond_vec_mag = self.cart_dist(bond_start, bond_end)
                        bond_vec = bond_end - bond_start

                        # these are the easy cases
                        if(start_type == 6):
                            h_dist = 1.09
                        elif(start_type == 7):
                            h_dist = 1.00
                        elif(start_type == 8):
                            h_dist = 0.96

                        scaled_bond_vec = h_dist/bond_vec_mag * (bond_vec)
                        new_bond_end = bond_start + scaled_bond_vec

                        # store necessary modifications
                        self.hydrogens[len(self.hydrogens.keys())] = {
                                                                      'cartesian_coordinates': new_bond_end,
                                                                      'atomic_number': 1,
                                                                      'element': 'H'
                                                                    }
                        self.num_keep += 1

        print("%s hydrogens added as caps: " % (str(len(self.hydrogens))))
        print("%s num atoms in final cluster: " % (str(self.num_keep)))

    def cxtd_comp_convert_to_orig(self):
        """
        Temporary convert so that the files can be written
        """
        self.components_to_keep = [self.cxtd_comp[id_] for id_ in self.cxtd_comp_to_keep]

    def get_BFS_tree(self):
        print("\n\nTURNING MOLECULAR GRAPH INTO BFS TREE")
        print("--------------------------------------")
        print("Start node: %d" % (self.start_index))
        print("Index of first node: %d" % (min([int(i) for i in self.origraph.nodes(data=False)])))

        # reset disgraph
        self.disgraph = self.origraph.copy()

        # get BFS tree of molecular graph
        tree = nx.bfs_tree(self.origraph, self.start_index)
        self.tree = tree

        # begin operating on this tree
        self.iterative_BFS_tree_structure(self.start_index)
        return tree


    def iterative_BFS_tree_structure(self, v):
        """
        Construct a dict with key that indexes depth of BFS tree,
        and the value is a set of all nodes at that depth
        """
        print("\n\nTURNING BFS TREE INTO DICT OF DEPTHS")
        print("--------------------------------------")


        # intitialize first level
        stack = set()
        stack.add(v)
        curr_depth = 0

        self.BFS_tree_dict = {
                                curr_depth: set(stack)
                             }

        curr_depth += 1


        # Move through every depth level in tree
        while(len(stack) != 0):

            # iterate over all up_nodes in stack
            for up_node in stack.copy():

                # get all down nodes from this up_node
                for down_node in self.tree.successors_iter(up_node):
                    stack.add(down_node)

                # after we've gotten all down nodes, remove this up node
                stack.remove(up_node)

            # add this depth and all nodes to the graph
            if(len(stack) != 0):
                self.BFS_tree_dict[curr_depth] = set(stack)
                curr_depth += 1



        print("Depth of BFS tree: " + str(len(self.BFS_tree_dict.keys())))
        for i in range(len(self.BFS_tree_dict.keys())):
            print("Level " + str(i) + ": " + str(len(self.BFS_tree_dict[i])))


        self.visualize_n_levels_of_tree(20)
        self.nodes_w_2plus_parents()
        #self.nodes_that_DNE_in_origraph() DEPRECATED
        #self.distree = self.tree.copy() DEPRECATED

        #print(self.get_nth_heirarcy_BFS_tree(46))
        self.preliminary_truncate_BFS_tree()
        self.truncate_all()

        #exit()

        #self.compute_cluster_in_tree()
        self.compute_cluster_in_disgraph()
        self.cap_by_material()


    def visualize_n_levels_of_tree(self, n):

        self.vistree = self.tree.copy()
        print(dir(self.vistree))

        for i in range(n, len(self.BFS_tree_dict.keys())):
            for node in self.BFS_tree_dict[i]:
                for child in self.tree.successors_iter(node):
                    self.vistree.remove_edge(node, child)

                self.vistree.remove_node(node)

        fig = plt.figure(figsize=(20,20))

        nx.spectral_layout(self.vistree)
        nx.draw_spectral(self.vistree, node_size = 50)
        homedir = os.path.expanduser('~')
        plt.savefig(homedir + '/Dropbox/sample_tree.png')

    def nodes_w_2plus_parents(self):
        print("\n\nFINDING NODES W/2+ PARENTS")
        print("--------------------------------------")
        num_Cu = 0
        for i in range(len(self.BFS_tree_dict.keys())):
            for node in self.BFS_tree_dict[i]:

                parents = len(self.tree.predecessors(node))

                if(parents > 1):
                    print("Node %s: %s parents" % (str(node), str(parents)))


    def nodes_that_DNE_in_origraph(self):
        print("\n\nFINDING MISSING EDGES IN ORIGRAPH")
        print("--------------------------------------")
        for n1, n2, data in self.origraph.edges_iter2(data=True):
            if((self.tree.has_edge(n1,n2)) or \
               (self.tree.has_edge(n2,n1))):
                pass
            else:
                print("BFS tree removed: %s!" % (str((n1,n2))))
                depth1 = -1
                depth2 = -1
                for i in range(len(self.BFS_tree_dict.keys())):
                    if(n1 in self.BFS_tree_dict[i]):
                        depth1 = int(i)
                        print("depth1 = %s" % (str(i)))
                    elif(n2 in self.BFS_tree_dict[i]):
                        depth2 = int(i)
                        print("depth2 = %s" % (str(i)))
                print("Added it back")
                if(depth1 < depth2):
                    self.tree.add_edge(n1, n2, data)
                else:
                    self.tree.add_edge(n2, n1, data)


    def get_nth_heirarcy_BFS_tree(self, n):
        return self.BFS_tree_dict[n]

    def truncation_criteria(self, up_node, down_node):
        # For 3D organics

        # For now only C-C bond truncation supported
        if(self.mat_type == "organic"):
            # get data associated with this edge
            data = self.origraph.get_edge_data(up_node, down_node)

            # first criteria is that we are truncating a C-C single bond
            if(self.origraph.node[up_node]['atomic_number'] in [6] and \
               self.origraph.node[down_node]['atomic_number'] in [6] and \
               data['order'] == 1.0):

                # get cartesian coords of nodes
                up_cart = self.origraph.node[up_node]['cartesian_coordinates']
                down_cart = self.origraph.node[down_node]['cartesian_coordinates']

                # get cartesian coords to cluster origin
                up_dist = self.cart_dist(up_cart, self.xyz)
                down_dist = self.cart_dist(down_cart, self.xyz)

                # do not truncate bond if both nodes inside the cutoff
                # radius of the cluster
                if((up_dist > self.rcut) and (down_dist > self.rcut)):
                    return True
                else:
                    return False
            else:
                return False
        # For zeolites
        elif(self.mat_type == "zeolite"):
            # For zeolites
            if((self.origraph.node[up_node]['atomic_number'] in [8,14] and \
                self.origraph.node[down_node]['atomic_number'] in [8, 14]) and \
               (len(self.tree.predecessors(down_node)) != 1 and \
                self.origraph.node[down_node]['atomic_number'] == 8) \
              ):
                return True
            else:
                return False
        elif(self.mat_type == "oned"):
            if(self.origraph.node[up_node]['atomic_number'] in self.metals and \
               self.origraph.node[down_node]['atomic_number'] in self.metals):
                return True
            else:
                return False
        else:
            print("Material type unknown, can't truncate")
            exit()



    def preliminary_truncate_BFS_tree(self):
        print("\n\nTRUNCATING BFS TREE BASED ON MATERIAL TYPE")
        print("--------------------------------------")

        # preliminary list of all bonds that can be truncated
        self.pot_truncs = set()
        self.pot_truncs_directed = set()
        self.pot_truncs_to_remove = set()
        added = 0
        for i in range(len(self.BFS_tree_dict.keys())):
            for up_node in self.BFS_tree_dict[i]:
                for down_node in self.tree.successors_iter(up_node):
                    this_edge = (self.origraph.sorted_edge_dict[(up_node, down_node)][0], \
                                 self.origraph.sorted_edge_dict[(up_node, down_node)][1])


                    data = self.origraph.get_edge_data(this_edge[0], this_edge[1])

                    # NOTE, looks like cyclic vs. acyclic works fine
                    #print(self.origraph.node[up_node].keys())
                    #print((self.origraph.node[up_node]['force_field_type'], self.origraph.node[up_node]['cycle']), self.origraph.node[up_node]['index'])


                    if(data['order'] == 1.0):
                            # For MOFs
                            # NOTE WHY ARE YOU LOOKING AT ORIGRAPH HERE???
                        if(self.truncation_criteria(up_node, down_node)):
                        #if((self.origraph.node[up_node]['atomic_number'] in [6] and \
                        #    self.origraph.node[down_node]['atomic_number'] in [6,7,8]) or \
                        #    # For zeolites
                        #   (self.origraph.node[up_node]['atomic_number'] == 8 and \
                        #    self.origraph.node[down_node]['atomic_number'] == 14)
                        #    # For 1D rod MOFs
                        #   #(self.origraph.node[up_node]['atomic_number'] in self.metals and \
                        #   # self.origraph.node[down_node]['atomic_number'] in self.metals)
                        #  ):
                        #    # Will need another condition for 1D rod MOFs

                        #    #print("Num parents of down node:")
                        #    #print(len(self.tree.predecessors(down_node)))
                        #    #print("Num children of up node:")
                        #    #print(len(self.tree.successors(up_node)))
                        #    if(len(self.tree.predecessors(down_node)) == 1):
                        #        #self.hydrogens[(this_edge[0], this_edge[1])] = {
                        #        #                      'cartesian_coordinates': self.origraph.node[up_node]['cartesian_coordinates'],
                        #        #                      'atomic_number': 10000,
                        #        #                      'element': 'A'
                        #        #                    }
                        #        #self.hydrogens[this_edge] = {
                        #        #                      'cartesian_coordinates': self.origraph.node[down_node]['cartesian_coordinates'],
                        #        #                      'atomic_number': 10000,
                        #        #                      'element': 'X'
                        #        #                    }
                        #        #self.num_keep += 1
                        #        pass

                        #        #print("Warning, cutting out a node that has 2+ parents")
                        #    if(len(self.tree.predecessors(down_node)) != 1 and self.origraph.node[down_node]['atomic_number'] == 8):
                        #        pass
                        #    else:
                        #    #if(True):
                        #        #print("Cutting at node that has exactly 1 parents")

                                up_cart, down_cart = self.origraph.node[up_node]['cartesian_coordinates'], \
                                                     self.origraph.node[down_node]['cartesian_coordinates']

                                #print(self.origraph.node[up_node].keys())
                                #print(self.origraph.node[up_node]['cycle'])
                                #exit()

                                up_dist = self.cart_dist(up_cart, self.xyz)
                                down_dist = self.cart_dist(down_cart, self.xyz)

                                # NOTE: original if statement here
                                # only consider truncation if both atoms are
                                # outside the cutoff radius
                                if((up_dist > self.rcut) and (down_dist > self.rcut)):

                                # NOTE: new if statment test here
                                #if((up_dist > self.rcut) and (down_dist > self.rcut)
                                #   and self.origraph.node[up_node]['cycle']):

                                #if((up_dist > self.rcut and down_dist > self.rcut) and \
                                #   (up_dist < down_dist)):
                                #if((up_dist < down_dist)):
                                    # NOTE here's the tricky part I was missing before
                                    # Now that we are cutting a DIRECTED graph, we can look at the end node of each cut
                                    # If this end node occurs in two different cuts, NEITHER are valid! \
                                    # We simply don't count this cut, and mark the other one for deletion
                                    to_add = True
                                    #print("\nCheck Overlap: " + str(added))
                                    #print("Down node: " + str(down_node))
                                    #for directed_edge in self.pot_truncs_directed:
                                    #    #print(directed_edge)
                                    #
                                    #    if(down_node == directed_edge[1]):
                                    #        #print("Gotcha!")
                                    #        to_add = False
                                    #        self.pot_truncs_to_remove.add(directed_edge)
                                    #        break

                                    if(to_add):
                                        self.pot_truncs.add((this_edge[0], this_edge[1]))
                                        self.pot_truncs_directed.add((up_node, down_node))
                                        added += 1
                                        #self.hydrogens[this_edge] = {
                                        #                      'cartesian_coordinates': self.origraph.node[down_node]['cartesian_coordinates'],
                                        #                      'atomic_number': 10000,
                                        #                      'element': 'X'
                                        #                    }
                                        #self.num_keep += 1


        print("Identified %s num of potential truncations" % (str(len(self.pot_truncs))))
        print("%s num of which are invalid because the directed child node are shared with another truncation" \
               % (str(len(self.pot_truncs_to_remove))))

        #for edge in self.pot_truncs:
        #    print("%d %d %s %s"%(edge[0], edge[1], self.origraph.node[edge[0]]['element'],
        #                                           self.origraph.node[edge[1]]['element']))

        #for directed_edge in self.pot_truncs_to_remove:
        #    this_edge = (self.origraph.sorted_edge_dict[directed_edge][0], \
        #                 self.origraph.sorted_edge_dict[directed_edge][1])

        #    self.pot_truncs.remove(this_edge)
        #    self.pot_truncs_directed.remove(directed_edge)


    def truncate_all(self):
        print("\n\nFINALIZE ALL TRUNCTAIONS TO MAKE")
        print("--------------------------------------")
        print("Truncating disgraph at all finalized truncation locations")
        self.actual_truncs = set()
        self.actual_truncs_directed = set()

        for n1, n2, data in self.origraph.edges_iter2(data=True):
            this_edge = (n1, n2)

            # if we identified a potential truncation bond that is periodic
            # our simbox wasn't big enough to start with
            if(data['symflag'] != '.'):
                #if(this_edge in self.pot_truncs):
                #    print("ERROR! Truncating a periodic edge that is too close to the cutoff...")
                #    print("Modify source code to start with more replications of unit cell...\nExiting...")
                #    exit()
                #self.disgraph.remove_edge(this_edge[0], this_edge[1])
                #self.pot_truncs.add(this_edge)
                pass


        for this_edge in self.pot_truncs:
            # remove the current edge from the disconnect graph
            # remember this is still the undirected graph, but we remove the reverse
            # edge because both forward and reverse edges are stored in self.pot_truncs
            self.disgraph.remove_edge(this_edge[0], this_edge[1])

            # keep track of the edge removed
            self.actual_truncs.add(this_edge)

            # keep track of the direction of this removed edge only if we
            # are keeping track of a directed version of the graph
            try:
                if(this_edge[0] in self.tree.successors(this_edge[1])):
                    self.actual_truncs_directed.add((this_edge[1], this_edge[0]))
                elif(this_edge[1] in self.tree.successors(this_edge[0])):
                    self.actual_truncs_directed.add((this_edge[0], this_edge[1]))
            except:
                pass

    def compute_cluster_in_tree(self):
        print("\n\nIDENTIFYING PRIMARY CLUSTER IN TREE")
        print("--------------------------------------")

        self.components = []
        self.components_to_keep = []
        primary_cluster = set()
        primary_cluster_ind = -1
        max_size = 0
        iter_ = 0
        for component in nx.connected_components(self.tree):
            print("Comp %s: %s nodes" % (str(iter_), str(len(component))))
            self.components.append(component)
            if(self.start_index in component):
                primary_cluster = component.copy()
                primary_cluster_ind = int(iter_)
            iter_ += 1

        print("Total number of components: %d" % (iter_))
        print("Primary cluster determined:")
        print("Comp %s: %s nodes" % (str(primary_cluster_ind), str(len(self.components[primary_cluster_ind]))))
        self.components_to_keep.append(primary_cluster)
        #self.update_num_keep()

    def compute_cluster_in_disgraph(self):
        print("\n\nIDENTIFYING PRIMARY CLUSTER IN DISGRAPH")
        print("--------------------------------------")

        self.components = []
        self.components_to_keep = []
        self.components_to_keep_ind = []
        self.components_to_keep_temp = set()
        primary_cluster = set()
        primary_cluster_ind = -1
        max_size = 0
        iter_ = 0

        # iterate through all components in graph
        for component in nx.connected_components(self.disgraph):
            print("Comp %s: %s nodes" % (str(iter_), str(len(component))))
            self.components.append(component)

            # ID primary component
            if(self.start_index in component):
                primary_cluster = component.copy()
                primary_cluster_ind = int(iter_)
                self.components_to_keep_temp.add(iter_)
                # primary component identified and break
                break

            # ID of a component that has at least one atom in cutoff
            for node in component:
                this_cart = self.origraph.node[node]['cartesian_coordinates']
                if(self.cart_dist(this_cart, self.xyz) < self.rcut):
                    self.components_to_keep_temp.add(iter_)

            # keep track of index of each component
            iter_ += 1

        # add primary component to the list to keep
        self.components_to_keep.append(primary_cluster)
        self.components_to_keep_ind.append(primary_cluster_ind)

        # add each component that we need to keep to a list
        for keep_ind in self.components_to_keep_temp:
            if(keep_ind != primary_cluster_ind):
                self.components_to_keep.append(components[keep_ind])
                self.components_to_keep_ind.append(keep_ind)

        print("Total number of components: %d" % (iter_))
        print("Primary component determined:")
        print("Comp %s: %s nodes" % (str(primary_cluster_ind), str(len(self.components[primary_cluster_ind]))))

        # Count how many truncations are associated with primary cluster
        num_truncs = 0
        for edge in self.actual_truncs_directed:
            #edge = self.actual_truncs_directed[i]
            if(edge[0] in self.components[primary_cluster_ind] or \
               edge[1] in self.components[primary_cluster_ind]):
                print(edge)
                num_truncs += 1

        print("Has %d truncations made to it" % (num_truncs))

        #self.components_to_keep.append(primary_cluster)
        #self.components_to_keep_ind.append(primary_cluster_ind)
        print("Additional components to keep (%s in total):"%(len(self.components_to_keep)-1))
        for i in range(1,len(self.components_to_keep)):
            print("Comp %d: %d" % (self.components_to_keep_ind[i], len(self.components_to_keep[i])))

        self.update_num_keep()
        print("%s num atoms in final cluster" % (str(self.num_keep)))



    def update_num_keep(self):
        for i in range(len(self.components_to_keep)):
            self.num_keep += len(self.components_to_keep[i])


    def cap_by_material(self):
        print("\n\nCAPPING BY MATERIAL TYPE")
        print("--------------------------------------")
        if(self.mat_type == 'zeolite'):
            print(self.mat_type)
            self.cap_zeolite_v2()
        elif(self.mat_type == 'organic'):
            print(self.mat_type)
            self.cap_3D_organic()
        else:
            pass


    def cap_zeolite_v2(self):
        """
        Cap a zeolite, not an extremeley difficult case
        """
        print("\n\nCAPPING ZEOLITE")
        print("--------------------------------------")


        print("%d Atom X debug probes" % len(self.hydrogens.keys()))
        self.nodes_to_replace = set()
        #print(self.actual_truncs_directed)

        for i in range(len(self.components_to_keep)):

            component = self.components_to_keep[i]

            for node in component:

                for nbr in self.tree.successors_iter(node):

                    if(nbr in self.hydrogens.keys()):
                        #self.hydrogens.pop(nbr, None)
                        pass
                    else:
                        if((node, nbr) in self.actual_truncs_directed):
                            print("Capping edge: " + str((node,nbr)))

                            # Don't cap if we're wrapping around into a node that's already been kept
                            to_add = True
                            for component in self.components_to_keep:
                                if(nbr in component):
                                    to_add = False
                                    break
                            if(to_add):
                                bond_start = self.origraph.node[node]['cartesian_coordinates']
                                bond_end =    self.origraph.node[nbr]['cartesian_coordinates']
                                start_type = self.origraph.node[node]['atomic_number']
                                end_type = self.origraph.node[nbr]['atomic_number']

                                bond_vec_mag = self.cart_dist(bond_start, bond_end)
                                bond_vec = bond_end - bond_start

                                # these are the easy cases
                                if(start_type == 8):
                                    h_dist = 0.96
                                elif(start_type == 14):
                                    h_dist = 1.46

                                scaled_bond_vec = h_dist/bond_vec_mag * (bond_vec)
                                new_bond_end = bond_start + scaled_bond_vec

                                # store necessary modifications
                                self.hydrogens[nbr] = {
                                                          'cartesian_coordinates': new_bond_end,
                                                          'atomic_number': 1,
                                                          'element': 'H'
                                                        }
                                self.num_keep += 1

            print("%s hydrogens added as caps: " % (str(len(self.hydrogens))))

    def cap_zeolite(self):
        """
        Cap a zeolite, not an extremeley difficult case
        """
        print("\n\nCAPPING ZEOLITE")
        print("--------------------------------------")


        print("%d Atom X debug probes" % len(self.hydrogens.keys()))
        self.nodes_to_replace = set()

        for i in range(len(self.components_to_keep)):

            component = self.components_to_keep[i]

            for node in component:

                for nbr in self.tree.successors_iter(node):

                    if(nbr in self.hydrogens.keys()):
                        #self.hydrogens.pop(nbr, None)
                        pass
                    else:
                        if((node, nbr) in self.actual_truncs_directed):
                            #print("Capping edge: " + str((node,nbr)))
                            to_add = True
                            for component in self.components_to_keep:
                                if(nbr in component):
                                    to_add = False
                                    break


                            if(to_add):
                                bond_start = self.origraph.node[node]['cartesian_coordinates']
                                bond_end =    self.origraph.node[nbr]['cartesian_coordinates']
                                start_type = self.origraph.node[node]['atomic_number']
                                end_type = self.origraph.node[nbr]['atomic_number']

                                bond_vec_mag = self.cart_dist(bond_start, bond_end)
                                bond_vec = bond_end - bond_start

                                # these are the easy cases
                                if(start_type == 8):
                                    h_dist = 0.96
                                elif(start_type == 14):
                                    h_dist = 1.46
                                elif(start_type == 8):
                                    h_dist = 0.96

                                scaled_bond_vec = h_dist/bond_vec_mag * (bond_vec)
                                new_bond_end = bond_start + scaled_bond_vec

                                # store necessary modifications
                                self.hydrogens[nbr] = {
                                                          'cartesian_coordinates': new_bond_end,
                                                          'atomic_number': 1,
                                                          'element': 'H'
                                                        }
                                self.num_keep += 1

            print("%s hydrogens added as caps: " % (str(len(self.hydrogens))))
            print("%s num atoms in final cluster: " % (str(self.num_keep)))



    def cap_3D_organic(self):
        """
        Cap a MOF (or COF, COP, whatever), that is a 3-D network (i.e. doens't have 1D rods)
        """
        print("\n\nCAPPING 3D ORGANIC")
        print("--------------------------------------")

        print(self.num_keep)
        all_nodes_to_keep = set()
        for component in self.components_to_keep:
            for node in component:
                all_nodes_to_keep.add(node)

        for i in range(len(self.components_to_keep)):

            component = self.components_to_keep[i]

            for node in component:

                #for nbr in self.tree.successors_iter(node):
                for nbr in self.origraph.neighbors(node):

                    #if((node, nbr) in self.actual_truncs_directed):
                    if(((node, nbr) in self.actual_truncs_directed and \
                        nbr not in all_nodes_to_keep)):
                       #((nbr, node) in self.actual_truncs_directed and \
                       # node not in all_nodes_to_keep)):

                       # NOTE this makes sure we don't place a cap somewhere
                       # that constitutes an atom we want to keep


                        print((node, nbr))

                        #print("Capping edge: " + str((node,nbr)))
                        bond_start = self.origraph.node[node]['cartesian_coordinates']
                        bond_end =    self.origraph.node[nbr]['cartesian_coordinates']
                        start_type = self.origraph.node[node]['atomic_number']
                        end_type = self.origraph.node[nbr]['atomic_number']

                        bond_vec_mag = self.cart_dist(bond_start, bond_end)
                        bond_vec = bond_end - bond_start

                        # these are the easy cases
                        if(start_type == 6):
                            h_dist = 1.09
                        elif(start_type == 7):
                            h_dist = 1.00
                        elif(start_type == 8):
                            h_dist = 0.96

                        scaled_bond_vec = h_dist/bond_vec_mag * (bond_vec)
                        new_bond_end = bond_start + scaled_bond_vec

                        # store necessary modifications
                        self.hydrogens[len(self.hydrogens.keys())] = {
                                                                      'cartesian_coordinates': new_bond_end,
                                                                      'atomic_number': 1,
                                                                      'element': 'H'
                                                                    }
                        self.num_keep += 1

        print("%s hydrogens added as caps: " % (str(len(self.hydrogens))))
        print("%s num atoms in final cluster: " % (str(self.num_keep)))


    def cap_1D_organic(self):
        pass

    def identify_mat_type(self):
        """
        For now just a basic check to classify a zeolite vs an organic (MOF, COF, etc)
        """
        print("\n\nClassifying material")
        print("--------------------------------------")
        print("0: zeolite")
        print("1: organic1 (1D-rod MOF)")
        print("2: organic  (3D-rod MOF)")
        could_be_Zeo = False
        could_be_Organic = False
        for node1,node2,data in self.origraph.edges_iter2(data=True):
            if(self.origraph.node[node1]['atomic_number'] == 14 and self.origraph.node[node2]['atomic_number'] == 8):
                could_be_Zeo = True
            if(self.origraph.node[node1]['atomic_number'] == 6 or self.origraph.node[node2]['atomic_number'] == 6):
                could_be_Organic = True

        if(could_be_Organic):
            self.mat_type = 'organic'
            self.iterable1 = [6]
            self.iterable2 = [6,7,8]
        else:
            self.mat_type = 'zeolite'

        print(self.mat_type)
        return self.mat_type


    def disconnect_external_building_blocks(self):
        """
        Break a super simulation box into every possible component where each disconnected
        bond represents a cappable bond BUT we only break bonds that straddle or are external
        to the cluster cutoff radius
        """
        print("\n\nDISCONNECTING EXTERNAL BUILDING BLOCKS")
        print("--------------------------------------")
        self.edges_to_cut = set()
        self.all_edges = {}
        for node1,node2,data in self.origraph.edges_iter2(data=True):
            # store all the data for later lookup so we don't have to iterate every time
            # just to get the data associated with an edge we want
            self.all_edges[(node1, node2)] = data

            if(data['order'] == 1.0):
                # no point in identifying a Hydrogen bond to cleave only to cap it again right after
                if(self.origraph.node[node1]['atomic_number'] != 1 and \
                   self.origraph.node[node2]['atomic_number'] != 1):

                    # For now we have to limit ourselves to only cutting single C-C bonds
                    # it becomes too difficult to handle edge cases otherwise
                    if(self.origraph.node[node1]['atomic_number'] in [6,7,8,14] and \
                       self.origraph.node[node2]['atomic_number'] in [6,7,8,14]):
                    #if((self.origraph.node[node1]['atomic_number'] in [6,7,8] and \
                    #    self.origraph.node[node2]['atomic_number'] in [6,7,8]) and \
                    #   (self.origraph.node[node1]['hybridization'] == '3' and \
                    #    self.origraph.node[node2]['hybridization'] == '3')):
                            # If all these criteria satsified, then we know how to cap a dangling bond
                            cart1, cart2 = self.origraph.node[node1]['cartesian_coordinates'], \
                                           self.origraph.node[node2]['cartesian_coordinates']

                            if(self.cart_dist(cart1, self.xyz) > self.rcut or \
                               self.cart_dist(cart2, self.xyz) < self.rcut):
                                self.edges_to_cut.add((node1, node2))
                                #print("Cutting edge: " + str((node1, node2)) + " " + \
                                #      str((self.origraph.node[node1]['element'], \
                                #           self.origraph.node[node2]['element'])))
                                self.disgraph.remove_edge(node1, node2)
        self.components = []
        for component in nx.connected_components(self.disgraph):
            self.components.append(component)
        print("Num edges in original graph: " + str(len(self.all_edges)))
        print("Num disconnected components: "  + str(len(self.components)))


    def identify_1D_building_blocks(self):
        """
        By going through each component determined from all_external_building_blocks() we can determine
        if the MOF is 1D rod.  If a component has two edges that eg have 'symflag' attribute of (4,x,x) and
        (6,x,x) respectively, then we found a component that spans across one crystallographic direction
        and reconnects with itself.  This is the definition of a 1D rod MOF
        """

        print("\n\nCHECKING FOR 1D BUILDING BLOCKS")
        print("-------------------------------")

        self.oneD_vec = []
        self.directionality = []
        self.final_direct = -1
        print("Checking for dimensionality of components")
        self.components = []
        for component in nx.connected_components(self.disgraph):
            self.components.append(component)
            #print(str(len(self.components)) + ": " + str(len(component)))
            #for e in component:
            #    print(self.origraph.node[e])
        print("Num disconnected components: "  + str(len(self.components)))


        for i in range(len(self.components)):
            #print("Comp " + str(i) + ": " + str(len(self.components[i])))

            # we need a subgraph construct to actually determine if this chunk is 1D rod
            this_subgraph = nx.Graph()
            could_be_1D = False
            possible_directionality = None

            for node in self.components[i]:
                #print("For node: " + str(node))
                for nbr in self.origraph[node]:
                    #print("Finding nbr: " + str(nbr))
                    # only form an edge if it is periodic
                    if (node, nbr) in self.all_edges:
                        symflag = self.all_edges[(node, nbr)]['symflag']
                        #print(symflag)
                        if(symflag == '.'):
                            this_subgraph.add_edge(node, nbr)
                        else:
                            could_be_1D = True
                            possible_directionality = self.parse_sym_flag_for_directionality(symflag)
                    elif (nbr, node) in self.all_edges:
                        symflag = self.all_edges[(nbr, node)]['symflag']
                        #print(symflag)
                        if(symflag == '.'):
                            this_subgraph.add_edge(node, nbr)
                        else:
                            could_be_1D = True
                            possible_directionality = self.parse_sym_flag_for_directionality(symflag)

                    else:
                        print("Ya done messed up A-aron")
                        exit()

            if(could_be_1D and self.mat_type != 'zeolite'):
                # if a component is still a continuous graph (1 segment) after disconnecting all edges inside it that
                # are periodic, then it must be a 1D rod type structure
                this_subgraph_comps = 0
                for this_comp in nx.connected_components(this_subgraph):
                    this_subgraph_comps += 1

                if(this_subgraph_comps>1):
                    pass
                    #print(False)
                else:
                    self.oneD_vec.append(i)
                    self.directionality.append(possible_directionality)
                    #print(True)
            else:
                pass
                #print(False)

        print("All components that are 1D rods:")
        print(len(self.oneD_vec))
        print(self.oneD_vec)

        if(len(set(self.directionality)) > 1):
            print("ERROR! Ambiguous dimensionality of rods. Check structure and/or modify source for this edge case...")
            print("Rods have directionality of(0 = a, 1 = b, 2 = c): ")
            print(self.final_direct)
            print("Non directionality of:")
            print(self.final_nondirect)
            print("Exiting....")
            exit()
        elif(len(set(self.directionality)) == 1):
            self.final_direct = self.directionality[0]
            self.final_nondirect = [int(i) for i in range(0,3) if i != self.final_direct]
            print("Rods have directionality of(0 = a, 1 = b, 2 = c): ")
            print(self.final_direct)
            print("Non directionality of:")
            print(self.final_nondirect)
        else:
            print("No 1D rods detected")




        if(len(self.oneD_vec)>0):
            self.mat_type == "oned"
            return True
        else:
            return False


    def disconnect_1D_building_blocks(self):
        """
        Disconnect 1D rods so they can actually be capped

        Best thing to do is still apply the same disconnection algorithm, only this time we are allowed to
        break a bond between a type in self.metals and [6,7,8]
        """

        # NOTE
        # NOTE
        # NOTE
        # IF WE HAVE A 1D ROD MOF, the primary cluster is still going to be identified
        # as a 1D rod!!!!!!!!!!
        print("\n\nDISCONNECTING 1D ROD BUILDING BLOCKS")
        print("------------------------------------")


        self.disgraph = self.origraph.copy()

        self.edges_to_cut = set()
        self.all_edges = {}
        #for i in range(len(self.oneD_vec)):

        for node,nbr,data in self.origraph.edges_iter2(data=True):
                    self.all_edges[(node, nbr)] = data
            #this_rod = self.oneD_vec[i]
            #print("Comp " + str(this_rod) + ":")
            #for node in self.components[this_rod]:
            #    for nbr in self.origraph[node]:
            #        if (node, nbr) in self.all_edges.keys():
            #            this_edge = (node, nbr)
            #        elif (nbr, node) in self.all_edges.keys():
            #            this_edge = (nbr, node)
            #        else:
            #            print("Ya done messed up A-aron")
            #            exit()
                    this_edge = (node, nbr)

                    # we can only disconnect single bonds
                    # NOTE we assume that metal oxide rod bonds are always determined as single
                    # otherwise this will fail
                    if(self.all_edges[this_edge]['order'] == 1.0):

                        # Don't disconnect any covalent hydrogen bonds
                        if((self.origraph.node[this_edge[0]]['atomic_number'] != 1 and \
                            self.origraph.node[this_edge[1]]['atomic_number'] != 1)):
                           #(self.origraph.node[this_edge[0]]['atomic_number'] not in self.metals and \
                           # self.origraph.node[this_edge[1]]['atomic_number'] not in self.metals)):

                            # we can cut a C-C bond
                            #if((self.origraph.node[this_edge[0]]['atomic_number'] in [6,7,8] and \
                            #    self.origraph.node[this_edge[1]]['atomic_number'] in [6,7,8]) or \
                            #   # or an M-O bond
                            #   (self.origraph.node[this_edge[0]]['atomic_number'] in [6,8,7] and \
                            #    self.origraph.node[this_edge[1]]['atomic_number'] in self.metals) or \
                            #   # or an M-N bond
                            #   (self.origraph.node[this_edge[1]]['atomic_number'] in [6,8,7] and \
                            #    self.origraph.node[this_edge[0]]['atomic_number'] in self.metals)):
                            if((self.origraph.node[this_edge[0]]['atomic_number'] in self.metals or \
                                self.origraph.node[this_edge[1]]['atomic_number'] in self.metals) or \
                               (self.origraph.node[this_edge[0]]['atomic_number'] in [6,8] and \
                                self.origraph.node[this_edge[1]]['atomic_number'] in [6,8])):

                                cart1, cart2 = self.origraph.node[this_edge[0]]['cartesian_coordinates'], \
                                               self.origraph.node[this_edge[1]]['cartesian_coordinates']

                                #print("Axial dist: " + str(cart1[self.final_direct]) + " " + str(self.xyz[self.final_direct]))
                                #print("Rad dist: " + str(cart1[self.final_nondirect]) + " " + str(self.xyz[self.final_nondirect]))
                                axial_dist1 = self.cart_dist(cart1[self.final_direct],self.xyz[self.final_direct])
                                axial_dist2 = self.cart_dist(cart2[self.final_direct],self.xyz[self.final_direct])
                                rad_dist1 = self.cart_dist(cart1[self.final_nondirect],self.xyz[self.final_nondirect])
                                rad_dist2 = self.cart_dist(cart2[self.final_nondirect],self.xyz[self.final_nondirect])
                                cart_dist1 = self.cart_dist(cart1,self.xyz)
                                cart_dist2 = self.cart_dist(cart2,self.xyz)

                                if((axial_dist1 > self.rcut and axial_dist2 > self.rcut) or \
                                   (rad_dist1 > self.rcut and rad_dist2 > self.rcut)):
                                #if(cart_dist1 > self.rcut and cart_dist2 > self.rcut):
                                    # to expensive for exception handling, just do O(n) lookup of this_edge in  edges_to_cut
                                    if(this_edge in self.edges_to_cut):
                                        pass
                                    else:
                                        #print("Cutting edge: " + str(this_edge) + " " + str(axial_dist1) + " " + str(rad_dist1) + " " + str(axial_dist2) + " " + str(rad_dist2))
                                        self.edges_to_cut.add((this_edge))
                                        self.disgraph.remove_edge(this_edge[0], this_edge[1])

        self.components = []
        for component in nx.connected_components(self.disgraph):
            self.components.append(component)
        print("Num disconnected components: "  + str(len(self.components)))

    def debug_edges_to_cut(self):
        print("\n\nDEBUG EDGES TO CUT")
        print("-------------------------")
        for this_edge in self.edges_to_cut:
            if(self.origraph.node[this_edge[0]]['atomic_number'] == 1 or
               self.origraph.node[this_edge[0]]['atomic_number'] == 1):
                print("ERROR: you cut an H covalent bond")
                exit()

        print("Pass")




    def compute_primary_cluster(self):
        print("\n\nCOMPUTING PRIMARY CLUSTER")
        print("-------------------------")
        print("Node has keys of:")
        print(self.origraph.node[1].keys())

        print("Computing connected components")
        self.components = []
        for component in nx.connected_components(self.disgraph):
            self.components.append(component)
        print("Num disconnected components: "  + str(len(self.components)))


        self.components_to_keep = []
        for i in range(len(self.components)):
            #print(component)
            #print("Comp " + str(i) + ": " + str(len(self.components[i])))
            for node in self.components[i]:
                #print("neighbors of " + str(node) + ":")
                cart1 = self.origraph.node[node]['cartesian_coordinates']
                if(self.cart_dist(cart1,self.xyz) < self.rcut):
                    self.components_to_keep.append(i)
                    break
                #for nbr in self.origraph[node]:
                #    pass


        print("Components to keep: "  + str(len(self.components_to_keep)))
        self.num_keep = 0
        for i in range(len(self.components_to_keep)):
            self.num_keep += len(self.components[self.components_to_keep[i]])
            #print("Comp " + str(self.components_to_keep[i]) + ": " + \
            #      str(len(self.components[self.components_to_keep[i]])))

    def compute_required_caps(self):

        additional_components = set()

        # iterate over all kept components
        for i in range(len(self.components_to_keep)):

            # look at each node in all kept components
            for node in self.components[self.components_to_keep[i]]:

                # look at each neighbor of each node in kept components
                #           Nbr3
                #            |
                #            |
                #            |
                # Nbr1 --X-- Node ----- Nbr4
                #            |
                #            X
                #            |
                #           Nbr2
                for nbr in self.origraph[node]:

                    # if the current edge was previously disconnected, we procede
                    if (node, nbr) in self.edges_to_cut or (nbr, node) in self.edges_to_cut:

                        # store the edge so that we can quickly look it up in self.edges_to_cut
                        if (node, nbr) in self.edges_to_cut:
                            this_edge = (node, nbr)
                        elif (nbr, node) in self.edges_to_cut:
                            this_edge = (nbr, node)

                        for j in range(len(self.components_to_keep)):
                            if(nbr in self.components[self.components_to_keep[j]]):
                                attempt_to_cap = False
                                break







    def cap_primary_cluster(self):
        print("\n\nCAPPING PRIMARY CLUSTER")
        print("-------------------------")
        print("Capping previously disconnected bonds w/hydrogen")

        # a set of all nodes that need to be included in the final cluster
        component_to_add = set()
        # a dict of all properties that need to be modified before writing the final cluster
        self.mods = {}
        # Loop over every component we want want to keep
        for i in range(len(self.components_to_keep)):
            #print("Comp " + str(self.components_to_keep[i]) + ": " + \
            #      str(len(self.components[self.components_to_keep[i]])))
            # loop over every node in that component to get the broken bonds in this
            for node in self.components[self.components_to_keep[i]]:
                # get neighbor of each node in component
                for nbr in self.origraph[node]:
                    # by default we attempt to cap
                    attempt_to_cap = True
                    skip_standard_cap = False

                    # if the current edge was previously disconnected, we procede
                    if (node, nbr) in self.edges_to_cut or (nbr, node) in self.edges_to_cut:
                        if (node, nbr) in self.edges_to_cut:
                            print("Cut bond: " + str((node,nbr)) + " " + str((self.origraph.node[node]['element'], self.origraph.node[nbr]['element'])))
                        elif (nbr, node) in self.edges_to_cut:
                            print("Reverse cut bond: " + str((nbr,node)) + " " + str((self.origraph.node[nbr]['element'], self.origraph.node[node]['element'])))

                        # now we need to arduously go back and check that this start/end
                        # combo doesn't link two components in self.components_to_keep
                        for j in range(len(self.components_to_keep)):
                            #if(j != i):
                            if(nbr in self.components[self.components_to_keep[j]]):
                                attempt_to_cap = False
                                break

                        if(nbr in component_to_add):
                            attempt_to_cap = False

                        if(attempt_to_cap):
                            bond_start = self.origraph.node[node]['cartesian_coordinates']
                            bond_end =    self.origraph.node[nbr]['cartesian_coordinates']
                            start_type = self.origraph.node[node]['atomic_number']
                            end_type = self.origraph.node[nbr]['atomic_number']

                            bond_vec_mag = self.cart_dist(bond_start, bond_end)
                            bond_vec = bond_end - bond_start

                            # these are the easy cases
                            if(start_type == 6):
                                h_dist = 1.09
                            elif(start_type == 7):
                                h_dist = 1.00
                            elif(start_type == 8):
                                h_dist = 0.96
                            # this is the really hard case, if the bond broken was Metal-X (or X-Metal)
                            # NOTE M-X bonds can only be severed when 1-D rods are identified, hence this
                            # fancy capping modifications only done for 1D rods

                            # NOTE this also counts for Si-O bonds in zeolites
                            elif(start_type in self.metals and end_type != start_type):
                                # now each nbr of nbr (denoted nbrnbr) is a candidate to become a hydrogen
                                # NOTE we assume that the metal must be coordinated to O or N (aka nbr = O, N)
                                for nbrnbr in self.origraph[nbr]:
                                    # first make sure we don't replace the metal we are trying to cap
                                    if(nbrnbr != node):
                                        bond_start = self.origraph.node[nbr]['cartesian_coordinates']
                                        bond_end =    self.origraph.node[nbrnbr]['cartesian_coordinates']
                                        start_type = self.origraph.node[nbr]['atomic_number']
                                        end_type = self.origraph.node[nbrnbr]['atomic_number']
                                        start_elem = self.origraph.node[nbr]['element']
                                        end_elem = self.origraph.node[nbrnbr]['element']
                                        target_h = int(nbrnbr)

                                        bond_vec_mag = self.cart_dist(bond_start, bond_end)
                                        bond_vec = bond_end - bond_start
                                        if(start_type == 6):
                                            h_dist = 1.09
                                        elif(start_type == 7):
                                            h_dist = 1.00
                                        elif(start_type == 8):
                                            h_dist = 0.96
                                        else:
                                            h_dist = 1.0
                                            pass
                                            #raise ValueError("ERROR! Unrecongnized coordination env of " + \
                                            #                 self.origraph.node[node]['element'] + "-" + \
                                            #                 self.origraph.node[nbr]['element'])

                                        scaled_bond_vec = h_dist/bond_vec_mag * (bond_vec)
                                        new_bond_end = bond_start + scaled_bond_vec

                                        # store necessary modifications
                                        self.mods[nbrnbr] = {
                                                              'cartesian_coordinates': new_bond_end,
                                                              'atomic_number': 1,
                                                              'element': 'H',
                                                              'old_cartesian_coordinates': bond_end,
                                                              'old_atomic_number': end_type,
                                                              'old_elem': end_elem
                                                            }

                                        component_to_add.add(nbrnbr)
                                        self.num_keep += 1
                                        component_to_add.add(nbr)
                                        self.num_keep += 1
                                        break
                                ## skip the regular capping step since we just did it
                                skip_standard_cap = True
                            elif(start_type in self.metals and end_type == start_type):
                                # NOTE we have to ignore metal-metal bonds in rods, if they actually exist
                                # in the structure then we can't do it accurately for now
                                pass
                            else:
                                raise ValueError("ERROR! Trying to cap a bond with " + \
                                                 self.origraph.node[node]['element'] + " node as start type")

                            if(skip_standard_cap):
                                continue
                            else:
                                scaled_bond_vec = h_dist/bond_vec_mag * (bond_vec)
                                new_bond_end = bond_start + scaled_bond_vec

                                # store necessary modifications
                                self.mods[nbr] = {
                                                      'cartesian_coordinates': new_bond_end,
                                                      'atomic_number': 1,
                                                      'element': 'H'
                                                    }

                                component_to_add.add(nbr)
                                self.num_keep += 1
                        else:
                            print("No capping because we have two components of self.components_to_keep that were originally connected")

        print(component_to_add)
        self.components.append(component_to_add)
        self.components_to_keep.append(len(self.components)-1)


    def modify_structure_w_hydrogens(self):
        for this_node in self.mods.keys():
            self.origraph.node[this_node]['cartesian_coordinates'] = self.mods[this_node]['cartesian_coordinates']
            self.origraph.node[this_node]['atomic_number'] = self.mods[this_node]['atomic_number']
            self.origraph.node[this_node]['element'] = self.mods[this_node]['element']


    def write_map_from_cluster_xyz_to_unique_types(self):
        struct = 'host'
        guest = 'guest'
        filename = struct + '_' + guest + '_' + str(self.rcut) + '.XYZmap'

        outfile = open(filename,"w")
        # NOTE, we don't want to write the capped hydrogens out
        unique_elements = dict()
        iter_ = 0
        for i in range(len(self.components_to_keep)):
            for node in self.components_to_keep[i]:
                this_id = "%s %s %s %s\n"%(self.origraph.node[node]['element'],
                                               str(node),
                                               str(iter_),
                                               self.origraph.node[node]['force_field_type'])
                outfile.write(this_id)

                elem = self.origraph.node[node]['element']
                if(elem in unique_elements.keys()):
                    unique_elem.append(this_id)
                else:
                    unique_elements[elem] = [this_id]

                iter_ += 1

        outfile.close()

        filename = struct + '_' + guest + '_' + str(self.rcut) + '.MOLECULEmap'
        filename1 = "MOLECULE.INP"
        outfile = open(filename,"w")
        outfile1 = open(filename1,"w")
        # now write a map because
        outfile1.write("BASIS\n6-31G\n\n\nAtomtypes=%s Nosymmetry Angstrom"%(len(unique_elements.keys())))
        for elem in unique_elements.keys():
            for entry in elem:
                outfile.write(elem)


    def write_map_from_MOLECULE_to_unique_types(self):
        struct = 'host'
        guest = 'guest'
        filename = struct + '_' + guest + '_' + str(self.rcut) + '.MOLECULEmap'


        iter_ = 0
        for i in range(len(self.components_to_keep)):
            for node in self.components_to_keep[i]:
                unique_elements.add(self.origraph.node[node]['element'])


        outstring = "BASIS\n6-31G\n\n\nAtomtypes=%s Nosymmetry Angstrom"%(len(unique_elements))

    def write_LSDALTON(self,location):
        """
        Write the LSDALTON file for DEC-MP2
        """

        outstring = "**WAVE FUNCTIONS\n.HF\n*DENSOPT\n.START\nATOMS\n**DEC\n.MP2\n.FOT\n1.0e-4\n.FROZENCORE\n.MEMORY\n 2.0\n*END OF INPUT"
        outfile = open(location, "w")
        outfile.write(outstring)
        outfile.close()


    def write_cutoff(self):
        """
        Write the cutoff that was used to make this cluster
        """
        filename = "rcut_" + str(self.rcut) + ".dat"
        f = open(filename,"w")
        f.write("%s"%str(self.rcut))
        f.close()




    def write_cluster_to_xyz(self):
        """
        Write the computed and capped cluster to an xyz file
        """

        struct = 'host'
        guest = 'guest'
        filename = struct + '_' + guest + '_' + str(self.rcut) + '.xyz'
        #home = os.path.expanduser('~')
        #outdir = home + '/Dropbox/ForceFields/data/MP2_input_files/'

        ##if not os.path.exists(outdir)
        #outname = home + '/Dropbox/ForceFields/data/MP2_input_files/' + filename

        outname = os.getcwd() + "/" + filename

        print("Writing cluster to <" + outname + ">")

        outfile = open(outname, 'w')
        outfile.write(str(self.num_keep)+'\n')
        outfile.write('cluster formation of test struct\n')
        atom = 1
        for i in range(len(self.components_to_keep)):
            #print("Comp " + str(self.components_to_keep[i]) + ": " + \
            #      str(len(self.components[self.components_to_keep[i]])))
            for node in self.components[self.components_to_keep[i]]:
                outfile.write("%s %s %s %s\n"%(self.origraph.node[node]['element'],
                                               self.origraph.node[node]['cartesian_coordinates'][0],
                                               self.origraph.node[node]['cartesian_coordinates'][1],
                                               self.origraph.node[node]['cartesian_coordinates'][2]))
                atom +=1
        outfile.close()


    def write_cluster_to_host_xyz_v2(self):
        """
        Write the computed and capped cluster to an xyz file
        """

        #self.components_to_keep = [self.kept_nodes]

        struct = 'host'
        guest = 'guest'
        filename = struct + '_' + str(self.rcut) + '.xyz'
        #home = os.path.expanduser('~')
        #outdir = home + '/Dropbox/ForceFields/data/MP2_input_files/'

        ##if not os.path.exists(outdir)
        #outname = home + '/Dropbox/ForceFields/data/MP2_input_files/' + filename

        outname = os.getcwd() + "/" + filename


        print("Writing cluster to <" + outname + ">")
        outfile = open(outname, 'w')
        outfile.write(str(self.num_keep)+'\n')
        outfile.write('cluster formation of host only\n')
        atom = 1
        for i in range(len(self.components_to_keep)):
            #print("Comp " + str(self.components_to_keep[i]) + ": " + \
            #      str(len(self.components[self.components_to_keep[i]])))
            for node in self.components_to_keep[i]:
                outfile.write("%s %s %s %s\n"%(self.origraph.node[node]['element'],
                                               self.origraph.node[node]['cartesian_coordinates'][0],
                                               self.origraph.node[node]['cartesian_coordinates'][1],
                                               self.origraph.node[node]['cartesian_coordinates'][2]))
                atom +=1

        for i in self.hydrogens.keys():
            outfile.write("%s %s %s %s\n"%(self.hydrogens[i]['element'],
                                           self.hydrogens[i]['cartesian_coordinates'][0],
                                           self.hydrogens[i]['cartesian_coordinates'][1],
                                           self.hydrogens[i]['cartesian_coordinates'][2]))
            atom += 1

        outfile.close()


    def write_cluster_to_xyz_host_guest_v2(self,include_guest):
        """
        Write the computed and capped cluster to an xyz file
        """

        #self.components_to_keep = [self.kept_nodes]

        struct = 'host_'
        guest = 'guest_'

        if(include_guest == True):
            subdir = "/host_guest_MP2/"
        else:
            subdir = "/host_MP2/"
            guest = ""


        if not os.path.exists(os.getcwd() + subdir):
            os.makedirs(os.getcwd() + subdir)

        # All the files we need to write for DEC-MP2
        xyzfilename = struct + guest + str(self.rcut) + '.xyz'
        xyzoutname = os.getcwd() + subdir + xyzfilename

        xyzmapfilename = struct + guest + str(self.rcut) + '.xyzmap'
        xyzmapoutname = os.getcwd() + subdir + xyzmapfilename

        MOLfilename = "MOLECULE.INP"
        MOLoutname = os.getcwd() + subdir + MOLfilename

        MOLmapfilename = struct + guest + str(self.rcut) + '.molmap'
        MOLmapoutname = os.getcwd() + subdir + MOLmapfilename

        LSDALTONfilename = "LSDALTON.INP"
        LSDALTONoutname = os.getcwd() + subdir + LSDALTONfilename
        self.write_LSDALTON(LSDALTONoutname)

        # get probe data
        f = open("probe.xyz","r")
        lines = f.readlines()
        probe_xyz = []
        probe_type = []
        for i in range(2, len(lines)):
            parsed = lines[i].strip().split()
            probe_type.append(parsed[0])
            probe_xyz.append([float(i) for i in parsed[1:4]])
        probe_xyz = np.array(probe_xyz)
        f.close()

        # strings to be written to file
        xyzstring = ""
        xyzmapstring = ""
        MOLstring = ""
        MOLmapstring = ""

        unique_elements=[]
        unique_elements_data=[]
        mol_map = []
        atom = 0
        for i in range(len(self.components_to_keep)):
            #print("Comp " + str(self.components_to_keep[i]) + ": " + \
            #      str(len(self.components[self.components_to_keep[i]])))
            for node in self.components_to_keep[i]:
                elem = self.origraph.node[node]['element']
                string = "%s %s %s %s\n"%(self.origraph.node[node]['element'],
                                               self.origraph.node[node]['cartesian_coordinates'][0],
                                               self.origraph.node[node]['cartesian_coordinates'][1],
                                               self.origraph.node[node]['cartesian_coordinates'][2])
                xyzstring += string

                mapstring = "%s %s %s %s %s %s %s\n"%(elem,
                                               str(node),
                                               str(atom),
                                               self.origraph.node[node]['force_field_type'],
                                               self.origraph.node[node]['cartesian_coordinates'][0],
                                               self.origraph.node[node]['cartesian_coordinates'][1],
                                               self.origraph.node[node]['cartesian_coordinates'][2])

                xyzmapstring += mapstring

                if(elem in unique_elements):
                    ind = unique_elements.index(elem)
                    unique_elements_data[ind].append(string)
                    mol_map[ind].append(mapstring)
                else:
                    unique_elements.append(elem)
                    unique_elements_data.append([string])
                    mol_map.append([mapstring])

                atom +=1

        for i in self.hydrogens.keys():
            elem = self.hydrogens[i]['element']
            string = "%s %s %s %s\n"%(self.hydrogens[i]['element'],
                                           self.hydrogens[i]['cartesian_coordinates'][0],
                                           self.hydrogens[i]['cartesian_coordinates'][1],
                                           self.hydrogens[i]['cartesian_coordinates'][2])

            xyzstring += string

            mapstring = "%s %s %s %s %s %s %s\n"%(elem,
                                           str(node),
                                           str(atom),
                                           "H_cap",
                                           self.hydrogens[i]['cartesian_coordinates'][0],
                                           self.hydrogens[i]['cartesian_coordinates'][1],
                                           self.hydrogens[i]['cartesian_coordinates'][2])

            xyzmapstring += mapstring

            if(elem in unique_elements):
                ind = unique_elements.index(elem)
                unique_elements_data[ind].append(string)
                mol_map[ind].append(mapstring)
            else:
                unique_elements.append(elem)
                unique_elements_data.append([string])
                mol_map.append([mapstring])

            atom += 1

        if(include_guest == True):
            for i in range(np.shape(probe_xyz)[0]):
                elem = probe_type[i]
                probe_xyz[i,0] += self.offset[0]
                probe_xyz[i,1] += self.offset[1]
                probe_xyz[i,2] += self.offset[2]
                string = "%s %s %s %s\n"%(probe_type[i],
                                               probe_xyz[i,0],
                                               probe_xyz[i,1],
                                               probe_xyz[i,2])

                xyzstring += string

                mapstring = "%s %s %s %s %s %s %s\n"%(elem,
                                               str(node),
                                               str(atom),
                                               "p0b" + str(i),
                                               probe_xyz[i,0],
                                               probe_xyz[i,1],
                                               probe_xyz[i,2])

                xyzmapstring += mapstring

                if(elem in unique_elements):
                    ind = unique_elements.index(elem)
                    unique_elements_data[ind].append(string)
                    mol_map[ind].append(mapstring)
                else:
                    unique_elements.append(elem)
                    unique_elements_data.append([string])
                    mol_map.append([mapstring])

                atom += 1

        print("Writing xyz host and guest to <" + xyzoutname + ">")
        outfile = open(xyzoutname, 'w')
        outfile.write(str(atom)+'\nhost+guest xyz file\n')
        outfile.write(xyzstring)
        outfile.close()

        print("Writing xyzmap to <" + xyzmapoutname + ">")
        outfile = open(xyzmapoutname, 'w')
        outfile.write(xyzmapstring)
        outfile.close()

        print("Writing MOLECULE.INP to <" + MOLoutname + ">")
        MOLstring += "BASIS\n6-31G\n\n\nAtomtypes=%s Nosymmetry Angstrom\n"%(len(unique_elements))

        for i in range(len(unique_elements)):
            elem = unique_elements[i]
            MOLstring += "Charge=%2f Atoms=%d\n"%(float(ATOMIC_NUMBER.index(elem)),
                                                len(unique_elements_data[i]))
            for j in range(len(unique_elements_data[i])):
                MOLstring += unique_elements_data[i][j]
                MOLmapstring += mol_map[i][j]
        outfile = open(MOLoutname, "w")
        outfile.write(MOLstring)
        outfile.close()

        print("Writing molmap to <" + MOLmapoutname + ">")
        outfile = open(MOLmapoutname, "w")
        outfile.write(MOLmapstring)
        outfile.close()


    def cut_cappable_bonds(self):
        print(type(self.graph))
        print(self.graph.__dict__.keys())

        print(type(self.graph.edge))
        print(self.graph.edge.keys())
        print(self.graph.edge[1].keys())
        # dictionary data for edge 1-49
        print(self.graph[1][49])
        # list of dictionaries of all connections
        print(self.graph[1][49])
        # node dictionary data for node1
        print(self.graph.node[1])

        # iterator for all nodes (ordered to match ) and data for edge
        for node1,node2,data in self.graph.edges_iter2(data=True):
            print(str(node1) + str(node2))

            #print(type(edge))

            #print(type(self.graph.edge))
            #print(type(self.graph.graph))
            #print(self.graph.graph.keys())
            #print(self.graph.graph['name'])
            #print(self.graph.edge[1].keys())
            #print(str(edge) + str(edge.order))

    def create_cluster_around_point_v2(self):
        """
        A BFS search approach to creating clusters
        """
        mat_type = self.identify_mat_type()
        start_index = self.get_start_and_kept_nodes()
        self.disconnect_external_building_blocks()
        one_D = self.identify_1D_building_blocks()
        if(one_D):
            print("1D rod identified")
            exit()
        else:
            tree = self.get_BFS_tree()
            #self.write_cluster_to_xyz_v2()
            self.write_cluster_to_xyz_host_guest_v2(include_guest=True)
            self.write_cluster_to_xyz_host_guest_v2(include_guest=False)
            #self.write_map_from_cluster_xyz_to_unique_types()
            self.write_cutoff()


    def create_cluster_around_point(self):
        # STEP 1:
        # Dislocate all bonds in the super simbox that straddle
        # or are outside the cluster cutoff

        mat_type = self.identify_mat_type()
        self.disconnect_external_building_blocks()
        self.debug_edges_to_cut()

        # STEP 2:
        # This step is very important, we now need to handle the edge cases of a 1D rod MOF
        # If it is indeed 1D rod, we will reset the calculation with a diff version of STEP 1
        one_D = self.identify_1D_building_blocks()
        if(one_D):
            self.disconnect_1D_building_blocks()
        self.debug_edges_to_cut()


        # STEP 3:
        # NOTE for now using the simplest capping algorithm
        # All disconnected components that have an atom within the cutoff are kept
        # If a kept component has disconnected edge that orignally connected to another kept component, reconnect
        # At this point the cluster is ready to be capped
        if(one_D):
            self.compute_primary_cluster()
        else:
            self.compute_primary_cluster()
        self.debug_edges_to_cut()


        # STEP 4:
        # Any bonds that remain disconnected after calculation of the primary cluster are identified
        # for capping according to proper chemical bonding rules
        if(one_D):
            self.cap_primary_cluster()
        else:
            self.cap_primary_cluster()


        # STEP 5:
        # structure is capped with hydrogen
        self.modify_structure_w_hydrogens()

        # STEP 5:
        # the molecular crystal cluster (NOT GUEST) is written to xyz
        self.write_cluster_to_xyz()

def read_xyz_center(filename):

    f = open(filename,"r")
    lines = f.readlines()
    xyz = [float(i) for i in lines[0].strip().split()]

    return np.array(xyz)

def main():

    # command line parsing
    #for r in [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]:

    options = Options()
    #options.cutoff = float(r)
    sim = LammpsSimulation(options)
    cell, graph = from_CIF(options.cif_file)
    sim.set_cell(cell)
    sim.set_graph(graph)
    sim.split_graph()

    #xyz = [25.0,25.0,25.0]
    # assume cwd is the one that has all the files in it
    xyz = read_xyz_center("cluster_center.xyz")
    print("Center of cluster: " + str(xyz))

    abc = np.dot(sim.cell.get_cell_inverse(),xyz)
    print("Abc coords: " + str(abc))

    abc = sim.cell.mod_to_UC(abc)
    print("Modded abc coords: " + str(abc))


    #sim.compute_simulation_size()
    sim.compute_cluster_box_size()

    abc = [(abc[0] + int(sim.supercell_tuple[0]/2))/sim.supercell_tuple[0], \
           (abc[1] + int(sim.supercell_tuple[1]/2))/sim.supercell_tuple[1], \
           (abc[2] + int(sim.supercell_tuple[2]/2))/sim.supercell_tuple[2]]
    print("Shifted to middle of cluster: " + str(abc))


    offset=np.dot(sim.cell.get_cell().T, abc)-xyz
    xyz = np.dot(sim.cell.get_cell().T, abc)
    print("Cluster origin: " + str(xyz))

    cluster = Cluster(sim.graph, xyz = xyz, offset=offset, rcut = options.cutoff)

    #sim.assign_force_fields()
    #cluster.create_cluster_around_point()
    #cluster.create_cluster_around_point_v2()
    cluster.create_cluster_around_point_v3()

    print(sim.cell.get_cell())
    opp_corner = np.dot(sim.cell.get_cell().T, [1,1,1])
    print(opp_corner)
    frac = np.dot(sim.cell.get_cell_inverse(),opp_corner)
    print(frac)

    #sim.merge_graphs()
    #if options.output_cif:
    #    print("CIF file requested. Exiting...")
    #    write_CIF(graph, cell)
    #    sys.exit()
    #sim.write_lammps_files()

if __name__ == "__main__":
    main()
