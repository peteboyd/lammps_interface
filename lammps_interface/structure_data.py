#/usr/bin/env python
"""
Molecular graph and structure io methods.
"""
from datetime import date
import numpy as np
from scipy.spatial import distance
import math
import shlex
import re
from .CIFIO import CIF, get_time
from .atomic import METALS, MASS, COVALENT_RADII
from copy import copy
from .mof_sbus import InorganicCluster, OrganicCluster
from copy import deepcopy
import itertools
import os
import sys
from .generic_raspa import GENERIC_PSEUDO_ATOMS_HEADER, GENERIC_PSEUDO_ATOMS
from .generic_raspa import GENERIC_FF_MIXING_HEADER, GENERIC_FF_MIXING
from .generic_raspa import GENERIC_FF_MIXING_FOOTER
from .uff import UFF_DATA
import networkx as nx
import operator

try:
    import networkx as nx
    from networkx.algorithms import approximation

except ImportError:
    print("Warning: could not load networkx module, this is needed to produce the lammps data file.")
    sys.exit()
from collections import OrderedDict
from .atomic import MASS, ATOMIC_NUMBER, COVALENT_RADII
from .atomic import organic, non_metals, noble_gases, metalloids, lanthanides, actinides, transition_metals
from .atomic import alkali, alkaline_earth, main_group, metals
from .ccdc import CCDC_BOND_ORDERS


DEG2RAD = np.pi / 180.


class MolecularGraph(nx.Graph):
    """Class to contain all information relating a structure file
    to a fully described classical system.
    Important specific arguments for atomic nodes:
    - mass
    - force_field_type
    - charge
    - cartesian_coordinates
    - description {contains all information about electronic environment
                   to make a decision on the final force_field_type}
        -hybridization [sp3, sp2, sp, aromatic]

    Important arguments for bond edges:
    - weight = 1
    - length
    - image_flag
    - force_field_type
    """
    node_dict_factory = OrderedDict
    def __init__(self, **kwargs):
        nx.Graph.__init__(self, **kwargs)
        # coordinates and distances will be kept in a matrix because
        # networkx edge and node lookup is slow.
        try:
            self.name = kwargs['name']
        except KeyError:
            self.name = 'default'
        self.coordinates = None
        self.distance_matrix = None
        self.original_size = 0
        self.molecule_id = 444
        self.inorganic_sbus = {}
        self.find_metal_sbus = False
        self.organic_sbus = {}
        self.find_organic_sbus = False
        self.cell = None
        self.rigid = False
        #TODO(pboyd): networkx edges do not store the nodes in order!
        # Have to keep a dictionary lookup to make sure the nodes
        # are referenced properly (particularly across periodic images)
        self.sorted_edge_dict = {}
        self.molecule_images = []


    def nodes_iter2(self, data=True):
        #FIXME(pboyd): latest version of NetworkX has removed nodes_iter...
        """Oh man, fixing to networkx 2.0

        This probably breaks a lot of stuff in the code. THANKS NETWORKX!!!!!!!1
	Extensive testing under way...

        """
        for node in self.nodes():
            if(data):
                data = self.node[node]
                yield (node, data)
            else:
                yield node

    def edges_iter2(self, data=True):
        for (n1,n2) in self.edges():
            v1,v2 = self.sorted_edge_dict[(n1,n2)]
            #d=self.edges[(n1,n2)]
            d=self[n1][n2]
            if(data):
                yield (v1,v2,d)
            else:
                yield (v1,v2)

        #for n1, n2, d in self.edges_iter(**kwargs):
        #    yield (self.sorted_edge_dict[(n1, n2)][0], self.sorted_edge_dict[(n1,n2)][1], d)

    def count_dihedrals(self):
        count = 0
        for n1, n2, data in self.edges_iter2(data=True):
            try:
                for dihed in data['dihedrals'].keys():
                    count += 1
            except KeyError:
                pass
        return count

    def count_angles(self):
        count = 0
        for node, data in self.nodes_iter2(data=True):
            try:
                for angle in data['angles'].keys():
                    count += 1
            except KeyError:
                pass
        return count

    def count_impropers(self):
        count = 0
        for node, data in self.nodes_iter2(data=True):
            try:
                for angle in data['impropers'].keys():
                    count += 1
            except KeyError:
                pass
        return count

    def reorder_labels(self, reorder_dic):
        """Re-order the labels of the nodes so that LAMMPS doesn't complain.
        This issue only arises when a supercell is built, but isolated molecules
        are not replicated in the supercell (by user request).
        This creates a discontinuity in the indices of the atoms, which breaks
        some features in LAMMPS.

        """

        old_nodes = sorted([(i,self.node[i]) for i in self.nodes()])
        #old_nodes = list(self.nodes_iter2(data=True))
        old_edges = list(self.edges_iter2(data=True))
        for node, data in old_nodes:

            if 'angles' in data:
                ang_data = list(data['angles'].items())
                for (a,c), val in ang_data:
                    data['angles'].pop((a,c))
                    data['angles'][(reorder_dic[a], reorder_dic[c])] = val
            if 'impropers' in data:
                imp_data = list(data['impropers'].items())
                for (a, c, d), val in imp_data:
                    data['impropers'].pop((a,c,d))
                    data['impropers'][(reorder_dic[a], reorder_dic[c], reorder_dic[d])] = val

            self.remove_node(node)
            data['index'] = reorder_dic[node]
            self.add_node(reorder_dic[node], **data)

        for b, c, data in old_edges:
            if 'dihedrals' in data:
                dihed_data = list(data['dihedrals'].items())
                for (a, d), val in dihed_data:
                    data['dihedrals'].pop((a,d))
                    data['dihedrals'][(reorder_dic[a], reorder_dic[d])] = val
            try:
                self.remove_edge(b,c)
            except nx.exception.NetworkXError:
                # edge already removed from 'remove_node' above
                pass
            self.add_edge(reorder_dic[b], reorder_dic[c], **data)

        old_edge_dict = self.sorted_edge_dict.items()
        self.sorted_edge_dict = {}
        for (a,b), val in old_edge_dict:
            self.sorted_edge_dict[(reorder_dic[a], reorder_dic[b])] = (reorder_dic[val[0]], reorder_dic[val[1]])

        old_images = self.molecule_images[:]
        self.molecule_images = []
        for m in old_images:
            newm = [reorder_dic[i] for i in m]
            self.molecule_images.append(newm)

    def add_atomic_node(self, **kwargs):
        """Insert nodes into the graph from the cif file"""
        #update keywords with more atom info
        # rename this to something more intuitive
        label="_atom_site_type_symbol"
        orig_keys = list(kwargs.keys())
        if(label not in kwargs):
            label = "_atom_site_label"
            if (label not in kwargs):
                print("ERROR: could not find the keyword for the element types in the cif file!"+
                        " Please use '_atom_site_type_symbol' or '_atom_site_label' for the element"+
                        " column.")
                sys.exit()

        charge_keywords = ["_atom_type_partial_charge",
                           "_atom_type_parital_charge",
                           "_atom_type_charge",
                           "_atom_site_charge" # RASPA cif file
                           ]
        element = kwargs.pop(label)

        # replacing Atom.__init__
        kwargs.update({'mass':MASS[element]})
        kwargs.update({'molid':self.molecule_id})
        kwargs.update({'element':element})
        kwargs.update({'cycle':False})
        kwargs.update({'rings':[]})
        kwargs.update({'atomic_number':ATOMIC_NUMBER.index(element)})
        kwargs.update({'pair_potential':None})
        kwargs.update({'h_bond_donor':False})
        kwargs.update({'h_bond_potential':None})
        kwargs.update({'tabulated_potential':False})
        kwargs.update({'table_potential':None})
        if set(orig_keys) & set(charge_keywords):
            key = list(set(orig_keys)&set(charge_keywords))[0]
            try:
                kwargs['charge'] = float(kwargs[key])
            except ValueError:
                print("Warning %s could not be converted "%(kwargs[key]) +
                      "to a charge value for atom %s"%(element) +
                      ", setting charge as 0.0 for this atom")
                kwargs['charge'] = 0.0
        else:
            kwargs['charge'] = 0.0
        try:
            fftype = kwargs.pop('_atom_site_description')
        except KeyError:
            fftype = None

        kwargs.update({'force_field_type':fftype})
        idx = self.number_of_nodes() + 1
        kwargs.update({'index':idx})
        #TODO(pboyd) should have some error checking here..
        try:
            n = kwargs.pop('_atom_site_label')
        except KeyError:
            n = label
        kwargs.update({'ciflabel':n})
        # to identify Cu paddlewheels, etc.
        #kwargs.update({'special_flag':None})
        self.add_node(idx, **kwargs)

    def compute_bonding(self, cell, scale_factor = 0.9):
        """Computes bonds between atoms based on covalent radii."""
        # here assume bonds exist, populate data with lengths and
        # symflags if needed.
        if (self.number_of_edges() > 0):
            # bonding found in cif file
            sf = []
            for n1, n2, data in self.edges_iter2(data=True):
                # get data['ciflabel'] for self.node[n1] and self.node[n2]
                # update the sorted_edge_dict with the indices, not the
                # cif labels
                n1data = self.node[n1]
                n2data = self.node[n2]
                n1label = n1data['ciflabel']
                n2label = n2data['ciflabel']
                try:
                    nn1, nn2 = self.sorted_edge_dict.pop((n1label, n2label))
                    if nn2 == n1label:
                        nn1 = n2
                        nn2 = n1
                    self.sorted_edge_dict.update({(n1, n2):(nn1, nn2)})
                    self.sorted_edge_dict.update({(n2, n1):(nn1, nn2)})
                except KeyError:
                    pass
                try:
                    nn1, nn2 = self.sorted_edge_dict.pop((n2label, n1label))
                    if nn2 == n1label:
                        nn1 = n2
                        nn2 = n1
                    self.sorted_edge_dict.update({(n2, n1):(nn1, nn2)})
                    self.sorted_edge_dict.update({(n1, n2):(nn1, nn2)})
                except KeyError:
                    pass

                sf.append(data['symflag'])
                bl = data['length']
                if bl <= 0.01:
                    id1, id2 = self.node[n1]['index']-1, self.node[n2]['index']-1
                    dist = self.distance_matrix[id1,id2]
                    data['length'] = dist

            if (set(sf) == set(['.'])):
                # compute sym flags
                for n1, n2, data in self.edges_iter2(data=True):
                    flag = self.compute_bond_image_flag(n1, n2, cell)
                    data['symflag'] = flag
            return

        # Here we will determine bonding from all atom pairs using
        # covalent radii.
        for n1, n2 in itertools.combinations(self.nodes(), 2):
            node1, node2 = self.node[n1], self.node[n2]
            e1, e2 = node1['element'],\
                    node2['element']
            elements = set([e1, e2])
            i1,i2 = node1['index']-1, node2['index']-1
            rad = (COVALENT_RADII[e1] + COVALENT_RADII[e2])
            dist = self.distance_matrix[i1,i2]
            tempsf = scale_factor
            # probably a better way to fix these kinds of issues..
            if (set("F") < elements) and  (elements & metals):
                tempsf = 0.8

            if (set("O") < elements) and (elements & metals):
                tempsf = 0.85
            # fix for water particle recognition.
            if(set(["O", "H"]) <= elements):
                tempsf = 0.8
            # fix for M-NDISA MOFs 
            if(set(["O", "C"]) <= elements):
                tempsf = 0.8
            if (set("O") < elements) and (elements & metals):
                tempsf = 0.82
                                                            
            # very specific fix for Michelle's amine appended MOF
            if(set(["N","H"]) <= elements):
                tempsf = 0.67
            if(set(["Mg","N"]) <= elements):
                tempsf = 0.80
            if(set(["C","H"]) <= elements):
                tempsf = 0.80
            if dist*tempsf < rad and not (alkali & elements):

                flag = self.compute_bond_image_flag(n1, n2, cell)
                self.sorted_edge_dict.update({(n1,n2): (n1, n2), (n2, n1):(n1, n2)})
                self.add_edge(n1, n2, key=self.number_of_edges() + 1,
                              order=1.0,
                              weight=1,
                              length=dist,
                              symflag = flag,
                              potential = None
                              )
    #TODO(pboyd) update this
    def compute_bond_image_flag(self, n1, n2, cell):
        """Update bonds to contain bond type, distances, and min img
        shift."""
        supercells = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        unit_repr = np.array([5,5,5], dtype=int)
        atom1 = self.node[n1]
        atom2 = self.node[n2]
        #coord1 = self.coordinates[atom1['index']-1]
        #coord2 = self.coordinates[atom2['index']-1]
        coord1 = self.node[n1]['cartesian_coordinates']
        coord2 = self.node[n2]['cartesian_coordinates']
        fcoords = np.dot(cell._inverse, coord2) + supercells

        coords = np.array([np.dot(j, cell.cell) for j in fcoords])

        dists = distance.cdist([coord1], coords)
        dists = dists[0].tolist()
        image = dists.index(min(dists))
        dist = min(dists)
        sym = '.' if all([i==0 for i in supercells[image]]) else \
                "1_%i%i%i"%(tuple(np.array(supercells[image],dtype=int) +
                                  unit_repr))
        if(dist > 7):
            #print("BEFORE: ", self[n1][n2]['symflag'])
            print("WARNING: bonded atoms %i and %i are %.3f Angstroms apart."%(n1,n2,dist) +
                    " This probably has something to do with the redefinition of the unitcell "+
                    "to a supercell. Please contact the developers!")
        return sym

    def compute_angle_between(self, l, m, r):
        coordl = self.node[l]['cartesian_coordinates']
        coordm = self.node[m]['cartesian_coordinates']
        coordr = self.node[r]['cartesian_coordinates']

        try:
            v1 = self.min_img(coordl - coordm)
            v2 = self.min_img(coordr - coordm)
        except AttributeError:
            v1 = coordl - coordm
            v2 = coordr - coordm

        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        a = np.arccos(np.dot(v1, v2))
        if np.isnan(a):
            if np.allclose((v1 + v2),np.zeros(3)):
                a = 180
            else:
                a = 0

        angle = a / DEG2RAD
        return angle

    def coplanar(self, node):
        """ Determine if this node, and it's neighbors are
        all co-planar.

        """
        coord = self.node[node]['cartesian_coordinates']
        vects = []
        for j in self.neighbors(node):
            vects.append(self.node[j]['cartesian_coordinates'] - coord)

        # use the first two vectors to define a plane
        v1 = vects[0]
        v2 = vects[1]
        n = np.cross(v1, v2)
        n /= np.linalg.norm(n)
        for v in vects[2:]:
            v /= np.linalg.norm(v)
            # what is a good tolerance for co-planarity in MOFs?
            # this is used solely to determine if a 4-coordinated metal atom
            # is square planar or tetrahedral..
            if not np.allclose(np.dot(v,n), 0., atol=0.02):
                return False
        return True

    def compute_dihedral_between(self, a, b, c, d):
        coorda = self.node[a]['cartesian_coordinates']
        coordb = self.node[b]['cartesian_coordinates']
        coordc = self.node[c]['cartesian_coordinates']
        coordd = self.node[d]['cartesian_coordinates']

        v1 = self.min_img(coorda - coordb)
        v2 = self.min_img(coordc - coordb)
        v3 = self.min_img(coordb - coordc)
        v4 = self.min_img(coordd - coordc)

        n1 = np.cross(v1, v2)
        n2 = np.cross(v3, v4)

        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)

        a = np.arccos(np.dot(n1, n2))
        if np.isnan(a):
            a = 0
        angle = a / DEG2RAD
        return angle

    def add_bond_edge(self, **kwargs):
        """Add bond edges (weight factor = 1)"""
        #TODO(pboyd) should figure out if there are other cif keywords to identify
        # atom types
        #TODO(pboyd) this is .cif specific and should be contained within the cif
        # file reading portion of the code. This is so that other file formats
        # can eventually be adopted if need be.

        n1 = kwargs.pop('_geom_bond_atom_site_label_1')
        n2 = kwargs.pop('_geom_bond_atom_site_label_2')
        try:
            length = float(del_parenth(kwargs.pop('_geom_bond_distance')))
        except KeyError:
            length = 0.0

        try:
            order = CCDC_BOND_ORDERS[kwargs['_ccdc_geom_bond_type']]
        except KeyError:
            order = 1.0

        try:
            flag = kwargs.pop('_geom_bond_site_symmetry_2')
        except KeyError:
            # assume bond does not straddle a periodic boundary
            flag = '.'
        kwargs.update({'length':length})
        kwargs.update({'weight': 1})
        kwargs.update({'order': order})
        kwargs.update({'symflag': flag})
        kwargs.update({'potential': None})
        # get the node index to avoid headaches
        for k,data in self.nodes_iter2(data=True):
            if data['ciflabel'] == n1:
                n1 = k
            elif data['ciflabel'] == n2:
                n2 =k

        self.sorted_edge_dict.update({(n1,n2): (n1, n2), (n2, n1):(n1, n2)})
        self.add_edge(n1, n2, key=self.number_of_edges()+1, **kwargs)

    def compute_cartesian_coordinates(self, cell):
        """Compute the cartesian coordinates for each atom node"""
        coord_keys = ['_atom_site_x', '_atom_site_y', '_atom_site_z']
        fcoord_keys = ['_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z']
        self.coordinates = np.empty((self.number_of_nodes(), 3))
        for node, data in self.nodes_iter2(data=True):
            #TODO(pboyd) probably need more error checking..
            try:
                coordinates = np.array([float(del_parenth(data[i])) for i in coord_keys])
            except KeyError:
                coordinates = np.array([float(del_parenth(data[i])) for i in fcoord_keys])
                coordinates = np.dot(coordinates, cell.cell)
            data.update({'cartesian_coordinates':coordinates})

            self.coordinates[data['index']-1] = coordinates

    def compute_min_img_distances(self, cell):
        self.distance_matrix = np.empty((self.number_of_nodes(), self.number_of_nodes()))
        for n1, n2 in itertools.combinations(self.nodes(), 2):
            id1, id2 = self.node[n1]['index']-1,\
                                self.node[n2]['index']-1
            #coords1, coords2 = self.coordinates[id1], self.coordinates[id2]
            coords1, coords2 = self.node[n1]['cartesian_coordinates'], self.node[n2]['cartesian_coordinates']
            try:
                dist = self.min_img_distance(coords1, coords2, cell)
            except TypeError:
                sys.exit()
            self.distance_matrix[id1][id2] = dist
            self.distance_matrix[id2][id1] = dist

    def min_img(self, coord):
        f = np.dot(self.cell.inverse, coord)
        f -= np.around(f)
        return np.dot(f, self.cell.cell)

    def in_cell(self, coord):
        f = np.dot(self.cell.inverse, coord) % 1
        return np.dot(f, self.cell.cell)

    def fractional(self, coord):
        f = np.dot(self.cell.inverse, coord)
        return f

    def min_img_distance(self, coords1, coords2, cell):
        one = np.dot(cell.inverse, coords1) % 1
        two = np.dot(cell.inverse, coords2) % 1
        three = np.around(one - two)
        four = np.dot(one - two - three, cell.cell)
        return np.linalg.norm(four)

    def compute_init_typing(self):
        """Find possible rings in the structure and
        initialize the hybridization for each atom.
        More refined determinations of atom and bond types
        is computed below in compute_bond_typing

        """
        #TODO(pboyd) return if atoms already 'typed' in the .cif file
        # compute and store cycles
        cycles = []
        for node, data in self.nodes_iter2(data=True):
            for n in self.neighbors(node):
                # fastest way I could think of..
                edge = self[node][n].copy()
                self.remove_edge(node, n)
                cycle = []
                try:
                    # cycle = list(nx.all_shortest_paths(self, node, n))
                    cycle = list(nx.all_simple_paths(self, source=node, target=n, cutoff=10))
                    if not cycle==[]: # storing all path that have the length of the shortest path of the graph
                        cycle_shortest_path = min(map(len, cycle))
                        cycle = [cyc for cyc in cycle if len(cyc) == cycle_shortest_path]
                except nx.exception.NetworkXNoPath:
                    pass
                self.add_edge(node, n, **edge)
                #FIXME MW edit to only store cycles < len(10)
                # should be a harmless edit but maybe need to test
                if(len(cycle) <= 10):
                    cycles += cycle
        for label, data in self.nodes_iter2(data=True):
            # N O C S
            neighbours = self.neighbors(label)
            element = data['element']
            if element == "C":
                if self.degree(label) >= 4:
                    self.node[label].update({'hybridization':'sp3'})
                elif self.degree(label) == 3:
                    self.node[label].update({'hybridization':'sp2'})
                elif self.degree(label) <= 2:
                    self.node[label].update({'hybridization':'sp'})
            elif element == "N":
                if self.degree(label) >= 3:
                    self.node[label].update({'hybridization':'sp3'})
                elif self.degree(label) == 2:
                    self.node[label].update({'hybridization':'sp2'})
                elif self.degree(label) == 1:
                    self.node[label].update({'hybridization':'sp'})
                else:
                    self.node[label].update({'hybridization':'sp3'})
            elif element == "O":
                n_elems = set([self.node[k]['element'] for k in neighbours])
                if self.degree(label) >= 2:
                    # if O is bonded to a metal, assume sp2 - like ...
                    # there's probably many cases where this fails,
                    # but carboxylate groups, bridging hydroxy groups
                    # make this true.
                    if (n_elems <= metals):
                        self.node[label].update({'hybridization': 'sp2'})
                    else:
                        self.node[label].update({'hybridization':'sp3'})
                elif self.degree(label) == 1:
                    self.node[label].update({'hybridization':'sp2'})
                else:
                    # If it has no neighbours, just give it SP3
                    self.node[label].update({'hybridization':'sp3'})
            elif element == "S":
                if self.degree(label) >= 2:
                    self.node[label].update({'hybridization':'sp3'})
                elif self.degree(label) == 1:
                    self.node[label].update({'hybridization':'sp2'})
                else:
                    self.node[label].update({'hybridization':'sp3'})

            else:
                #default sp3
                self.node[label].update({'hybridization':'sp3'})
        # convert to aromatic
        # probably not a good test for aromaticity..
        arom = set(["C", "N", "O", "S"])
        for cycle in cycles:
            elements = [self.node[k]['element'] for k in cycle]
            neigh = [self.degree(k) for k in cycle]
            if np.all(np.array(neigh) <= 3) and set(elements) <= arom:
                for a in cycle:
                    self.node[a]['hybridization'] = 'aromatic'
                    self.node[a]['cycle'] = True
                    self.node[a]['rings'].append(cycle)

    def compute_bond_typing(self):
        """ Compute bond types and atom types based on the local edge
        environment.
        Messy, loads of 'ifs'
        is there a better way to catch chemical features?
        """
        #TODO(pboyd) return if bonds already 'typed' in the .cif file
        double_check = []
        for n1, n2, data in self.edges_iter2(data=True):
            elements = [self.node[a]['element'] for a in (n1,n2)]
            hybridization = [self.node[a]['hybridization'] for a in (n1, n2)]
            rings = [self.node[a]['rings'] for a in (n1, n2)]
            samering = False

            if set(hybridization) == set(['aromatic']):
                for r in rings[0]:
                    if n2 in r:
                        samering = True
                if(samering):
                    data.update({"order" : 1.5})

            if set(elements) == set(["C", "O"]):
                car = n1 if self.node[n1]['element'] == "C" else n2
                car_data = self.node[car]
                oxy = n2 if self.node[n2]['element'] == "O" else n1
                oxy_data = self.node[oxy]

                carnn = [i for i in self.neighbors(car) if i != oxy]
                try:
                    carnelem = [self.node[j]['element'] for j in carnn]
                except:
                    carnelem = []

                oxynn = [i for i in self.neighbors(oxy) if i != car]
                try:
                    oxynelem = [self.node[j]['element'] for j in oxynn]
                except:
                    oxynelem = []
                if "O" in carnelem:
                    at = carnn[carnelem.index("O")]
                    at_data = self.node[at]
                    if self.degree(at) == 1:
                        if self.degree(oxy) == 1:
                            #CO2
                            car_data['hybridization'] = 'sp'
                            oxy_data['hybridization'] = 'sp2'
                            data['order'] = 2.
                        else:
                            # ester
                            if set(oxynelem) <= organic:
                                car_data['hybridization'] = 'sp2'
                                oxy_data['hybridization'] = 'sp2'
                                data['order'] = 1 # this is the ether part of an ester...
                            #carboxylate?
                            else:
                                car_data['hybridization'] = 'aromatic'
                                oxy_data['hybridization']= 'aromatic'
                                data['order'] = 1.5

                    else:
                        atnelem = [self.node[k]['element'] for k in self.neighbors(at)]
                        if (set(atnelem) <= organic):
                            # ester
                            if len(oxynn) == 0:
                                car_data['hybridization'] = 'sp2'
                                oxy_data['hybridization'] = 'sp2'
                                data['order'] = 2. # carbonyl part of ester
                            # some kind of resonance structure?
                            else:
                                car_data['hybridization'] = 'aromatic'
                                oxy_data['hybridization'] = 'aromatic'
                                data['order'] = 1.5
                        else:
                            car_data['hybridization'] = 'aromatic'
                            oxy_data['hybridization'] = 'aromatic'
                            data['order'] = 1.5

                if "N" in carnelem:
                    at = carnn[carnelem.index("N")]
                    # C=O of amide group
                    if self.degree(oxy) == 1:
                        data['order'] = 1.5
                        car_data['hybridization'] = 'aromatic'
                        oxy_data['hybridization'] = 'aromatic'
                # only one carbon oxygen connection.. could be C=O, R-C-O-R, R-C=O-R
                if (not "O" in carnelem) and (not "N" in carnelem):
                    if len(oxynn) > 0:
                        # ether
                        oxy_data['hybridization'] = 'sp3'
                        data['order'] = 1.0
                    else:
                        if car_data['cycle'] and car_data['hybridization'] == 'aromatic':
                            oxy_data['hybridization'] = 'aromatic'
                            data['order'] = 1.5
                        # carbonyl
                        else:
                            oxy_data['hybridization'] = 'sp2'
                            data['order'] = 2.0
            elif set(elements) == set(["C", "N"]) and not samering:
                car = n1 if self.node[n1]['element'] == "C" else n2
                car_data = self.node[car]
                nit = n2 if self.node[n2]['element'] == "N" else n1
                nit_data = self.node[nit]
                carnn = [j for j in self.neighbors(car) if j != nit]
                carnelem = [self.node[k]['element'] for k in carnn]
                nitnn = [j for j in self.neighbors(nit) if j != car]
                nitnelem = [self.node[k]['element'] for k in nitnn]
                # aromatic amine connected -- assume part of delocalized system
                if car_data['hybridization'] == 'aromatic' and set(['H']) == set(nitnelem):
                    data['order'] = 1.5
                    nit_data['hybridization'] = 'aromatic'
                # amide?
                elif self.degree(car) == 3 and len(nitnn) >=2:
                    if "O" in carnelem:
                        data['order'] = 1.5 # (amide)
                        nit_data['hybridization'] = 'aromatic'
                    # nitro
                    if set(nitnelem) == set(["O"]):
                        data['order'] = 1.
                        nit_data['hybridization'] = 'aromatic'
                        for oatom in nitnn:
                            nobond = self[nit][oatom]['order'] = 1.5
                            self.node[oatom]['hybridization'] = 'aromatic'

            elif (not self.node[n1]['cycle']) and (not self.node[n2]['cycle']) and (set(elements) <= organic):
                if set(hybridization) == set(['sp2']):
                    try:
                        cr1 = COVALENT_RADII['%s_2'%elements[0]]
                    except KeyError:
                        cr1 = COVALENT_RADII[elements[0]]
                    try:
                        cr2 = COVALENT_RADII['%s_2'%(elements[1])]
                    except KeyError:
                        cr2 = COVALENT_RADII[elements[1]]
                    covrad = cr1 + cr2
                    # first pass: assign all to 2.0 bond order
                    data['order'] = 2.0
                    double_check += [n1, n2]
                    #if (data['length'] <= covrad*.95):
                    #    data['order'] = 2.0
                elif set(hybridization) == set(['sp']):
                    try:
                        cr1 = COVALENT_RADII['%s_1'%elements[0]]
                    except KeyError:
                        cr1 = COVALENT_RADII[elements[0]]
                    try:
                        cr2 = COVALENT_RADII['%s_1'%elements[1]]
                    except KeyError:
                        cr2 = COVALENT_RADII[elements[1]]
                    # first pass: assign all to 3.0 bond order
                    double_check += [n1, n2]
                    data['order'] = 3.0
                    #covrad = cr1 + cr2
                    #if (data['length'] <= covrad*.95):
                    #    data['order'] = 3.0
        # second pass, check organic unsaturated bonds to make
        # sure alkyl chains are alternating etc.
        while double_check:
            n = double_check.pop()
            # rewind this atom to a 'terminal' connected atom
            for i in self.recurse_bonds_to_end(n, pool=[], visited=[]):
                start = i
                try:
                    idn = double_check.index(i)
                    del double_check[idn]
                except ValueError:
                    pass
            # iterate over all linear chains
            # BE CAREFUL about overwriting bond orders here, the recursion
            # can have duplicate bonds for each 'k' iteration since it iterates over
            # all possible linear chains. So if the molecule is branched, there
            # will be multiple recursions over the same set of bonds.
            for k in self.recurse_linear_chains(start, visited=[], excluded=[]):
                bond_orders = []
                # first pass, store all the bond orders
                for idx in range(len(k)-1):
                    n1 = k[idx]
                    n2 = k[idx+1]
                    bond = self[n1][n2]
                    bond_orders.append(bond['order'])
                # second pass, check continuity
                for idx in range(len(k)-1):
                    n1 = k[idx]
                    n2 = k[idx+1]
                    data1 = self.node[n1]
                    data2 = self.node[n2]

                    hyb1 = data1['hybridization']
                    hyb2 = data2['hybridization']

                    elem1 = data1['element']
                    elem2 = data2['element']
                    if(idx == 0):
                        order = bond_orders[idx]
                        if (len(bond_orders) > idx+1):
                            next_order = bond_orders[idx+1]
                        if (hyb1 == 'sp2') and (hyb2 == 'sp2'):
                            bond_orders[idx] = 2.
                    elif (idx < len(k)-2):
                        prev_order = bond_orders[idx-1]
                        next_order = bond_orders[idx+1]
                        order = bond_orders[idx]
                        if (hyb1 == 'sp2') and (hyb2 == 'sp2'):
                            if (prev_order == 2.) and (next_order == 1):
                                bond_orders[idx-1] = 1.5
                                bond_orders[idx] = 1.5
                            elif (prev_order == 2.) and (next_order == 2.):
                                bond_orders[idx] = 1.
                        elif (hyb1 == 'sp') and (hyb2 == 'sp'):
                            if (prev_order == 3.) and (next_order == 3.):
                                bond_orders[idx] = 1.
                    else:
                        prev_order = bond_orders[idx-1]
                        order = bond_orders[idx]
                        if (hyb1 == 'sp2') and (hyb2 == 'sp2'):
                            if (prev_order == 2.) and (order == 2):
                                if set([elem1, elem2]) == set(["C", "O"]):
                                    onode = n2 if self.node[n2]['element'] == "O" else n1
                                    # this very specific case is a enol
                                    if self.degree(onode) == 1:
                                        bond_orders[idx] = 1.5
                                        bond_orders[idx-1] = 1.5
                        elif (hyb1 == 'sp') and (hyb2 == 'sp'):
                            if (prev_order == 3.):
                                bond_orders[idx] = 1.

                for idx in range(len(k)-1):
                    n1 = k[idx]
                    n2 = k[idx+1]
                    bond = self[n1][n2]
                    # update bond orders.
                    bond['order'] = bond_orders[idx]

                #print([self.node[r]['element'] for r in k])

    def recurse_linear_chains(self, node, visited=[], excluded=[]):
        """Messy recursion function to return all unique chains from a set of atoms between two
        metals (or terminal atoms in the case of molecules)"""
        if self.node[node]['element'] == 'H':
            yield
        neighbors = [i for i in self.neighbors(node) if i not in excluded and self.node[i]['element'] != "H"]
        if (not neighbors) and (node in excluded) and (not visited):
            return
        elif (not neighbors) and (node in excluded):
            nde = visited.pop()
        elif (not neighbors) and (not (node in excluded)):
            excluded.append(node)
            visited.append(node)
            yield visited
            nde = visited.pop()
        else:
            excluded.append(node)
            visited.append(node)
            nde = neighbors[0]
        for x in self.recurse_linear_chains(nde, visited, excluded):
            yield x

    def recurse_bonds_to_end(self, node, pool=[], visited=[]):
        if self.node[node]['element'] == 'H':
            return
        visited.append(node)
        neighbors = [i for i in self.neighbors(node) if i not in visited and self.node[i]['element'] != "H"]
        pool += neighbors
        yield node
        if (not pool) or (self.node[node]['element'] in list(metals)):
            return
        for x in self.recurse_bonds_to_end(pool[0], pool[1:], visited):
            yield x

    def atomic_node_sanity_check(self):
        """Check for specific keyword/value pairs. Exit if non-existent"""

    def compute_angles(self):
        """angles are attached to specific nodes, this way
        if a node is cut out of a graph, the angle comes with it.


               b-----a
              /
             /
            c

        Must be updated with different adjacent nodes if a
        supercell is requested, and the angle crosses a
        periodic image.

        """
        for b, data in self.nodes_iter2(data=True):
            if self.degree(b) < 2:
                continue
            angles = itertools.combinations(self.neighbors(b), 2)
            for (a, c) in angles:
                data.setdefault('angles', {}).update({(a,c):{'potential':None}})

    def compute_dihedrals(self):
        """Dihedrals are attached to specific edges in the graph.
           a
            \
             b -- c
                   \
                    d

        the edge between b and c will contain all possible dihedral
        angles between the neighbours of b and c (this includes a
        and d and other possible bonded atoms)

        """
        for b, c, data in self.edges_iter2(data=True):
            b_neighbours = [k for k in self.neighbors(b) if k != c]
            c_neighbours = [k for k in self.neighbors(c) if k != b]
            for a in b_neighbours:
                for d in c_neighbours:
                    data.setdefault('dihedrals',{}).update({(a, d):{'potential':None}})

    def compute_improper_dihedrals(self):
        """Improper Dihedrals are attached to specific nodes in the graph.
           a
            \
             b -- c
             |
             d

        the node b will contain all possible improper dihedral
        angles between the neighbours of b

        """
        for b, data in self.nodes_iter2(data=True):
            if self.degree(b) != 3:
                continue
            # three improper torsion angles about each atom
            local_impropers = list(itertools.permutations(self.neighbors(b)))
            for idx in range(0, 6, 2):
                (a, c, d) = local_impropers[idx]
                data.setdefault('impropers',{}).update({(a,c,d):{'potential':None}})

    def compute_topology_information(self, cell, tol, num_neighbours):
        self.compute_cartesian_coordinates(cell)
        self.compute_min_img_distances(cell)
        self.compute_bonding(cell)
        self.compute_init_typing()
        self.compute_bond_typing()
        if (self.find_metal_sbus):
            self.detect_clusters(num_neighbours, tol) # num neighbors determines how many nodes from the metal element to cut out for comparison
        if (self.find_organic_sbus):
            self.detect_clusters(num_neighbours, tol,  type="Organic")
        self.compute_angles()
        self.compute_dihedrals()
        self.compute_improper_dihedrals()

    def sorted_node_list(self):
        return [n[1] for n in sorted([(data['index'], node) for node, data in self.nodes_iter2(data=True)])]

    def sorted_edge_list(self):
        return [e[1] for e in sorted([(data['index'], (n1, n2)) for n1, n2, data in self.edges_iter2(data=True)])]

    def show(self):
        nx.draw(self)

    def img_offset(self, cells, cell, maxcell, flag, redefine, n1=0):
        unit_repr = np.array([5, 5, 5], dtype=int)
        if(flag == '.'):
            return cells.index(tuple([tuple([i]) for i in cell]))
        translation = np.array([int(j) for j in flag[2:]]) - unit_repr
        ocell = np.array(cell + translation, dtype=np.float64)

        # have to find the off-diagonal values, and what their
        # multiples are to determine the image cell of this
        # bond.
        imgcell = ocell % maxcell
        if redefine is None:
            return cells.index(tuple([tuple([i]) for i in imgcell]))

        olde_imgcell = imgcell
        newcell = cell + np.dot(cell, redefine)%maxcell
        newocell = (newcell + translation) #% maxcell
        rd2 = redefine - np.identity(3)*maxcell
        # check if the newcell translation spans a periodic boundary
        if (np.any(newocell >= maxcell, axis=0)) or (np.any(newocell < 0.)):

            # get indices of where the newocell is spanning a pbc
            indexes = np.where(newocell - maxcell >= 0)[0].tolist() + np.where(newocell < 0)[0].tolist()

            # determine if the translation spans a lattice vector which is a linear combination
            # of the other vectors, in which case, make a correction in the image cell
            # this is checked by finding which lattice indices are spanning the cell, then
            # checking if these vectors are a linear combination of the other ones by
            # summing the columns together.
            # this ONLY works if b and c are lin combs of a and a is the unit vector
            newrd = np.sum(rd2[indexes], axis=0)%maxcell
            #newrd = np.dot(newocell, redefine)%maxcell
            if np.any(newrd != 0):
                imgcell = (newocell - np.dot(newocell, redefine)) % maxcell

        return cells.index(tuple([tuple([i]) for i in imgcell]))

    def update_symflag(self, cell, symflag, mincell, maxcell):
        unit_repr = np.array([5, 5, 5], dtype=int)
        ocell = cell + np.array([int(j) for j in symflag[2:]]) - unit_repr
        imgcell = ocell % maxcell
        if any(ocell < mincell) or any(ocell >= maxcell):
            newflaga = np.array([5,5,5])
            newflaga[np.where(ocell >= maxcell)] = 6
            newflaga[np.where(ocell < np.zeros(3))] = 4
            newflag = "1_%i%i%i"%(tuple(newflaga))

        else:
            newflag = '.'
        return newflag

    def correspondence_graph(self, graph, tol, general_metal=False, node_subset=None):
        """Generate a correspondence graph between the nodes
        and the SBU.
        tolerance is the distance tolerance for the edge generation
        in the correspondence graph.

        """
        if node_subset is None:
            node_subset = self.nodes()
        graph_nodes = graph.nodes()
        cg = nx.Graph()
        # add nodes to cg
        for (i, j) in itertools.product(node_subset, graph_nodes):
            # match element-wise
            elementi = self.node[i]['element']
            elementj = graph.node[j]['element']
            match = False
            if general_metal:
                if ATOMIC_NUMBER.index(elementi) in METALS and ATOMIC_NUMBER.index(elementj) in METALS:
                    match = True
                elif elementi == elementj:
                    match = True
            elif elementi == elementj:
                match = True
            if match:
                cg.add_node((i,j))
        # add edges to cg
        for (a1, b1), (a2, b2) in itertools.combinations(cg.nodes(), 2):
            if (a1 != a2) and (b1 != b2):
                da = self.distance_matrix[a1-1, a2-1]
                db = graph.distance_matrix[b1-1, b2-1]
                if np.allclose(da, db, atol=tol):
                    cg.add_edge((a1,b1), (a2,b2))
        return cg

    def detect_clusters(self, num_neighbors, tol, type='Inorganic', general_metal=False):
        """Detect clusters such as the copper paddlewheel using
        maximum clique detection. This will assign specific atoms
        with a special flag for use when building their force field.

        setting general_metal to True will allow for cluster recognition of
        inorganic SBUs while ignoring the specific element type of the metal,
        so long as it is a metal, it will be paired with other metals.
        This may increase the time for SBU recognition.

        """
        print("Detecting %s clusters"%type)

        reference_nodes = []

        if type=="Inorganic":
            types = InorganicCluster.keys()
            ref_sbus = InorganicCluster
            store_sbus = self.inorganic_sbus
        elif type == "Organic":
            types = OrganicCluster.keys()
            ref_sbus = OrganicCluster
            store_sbus = self.organic_sbus

        for node, data in self.nodes_iter2(data=True):
            if (type=='Inorganic') and (general_metal) and (data['atomic_number'] in METALS)\
                    and ('special' not in data.keys()): # special means that this atom has already been found in a previous clique detection
                reference_nodes.append(node)

            elif (data['element'] in types) and ('special' not in data.keys()):
                reference_nodes.append(node)

        no_cluster = []
        #FIXME(pboyd): This routine doesn't work for finding 1-D rod SBUs.
        # At each step the atoms found that belong to an SBU are deleted
        # to improve the search efficiency. In 1-D rod SBUs there are
        # repeating elements in the 'discretized' version used to discover
        # the rod in a MOF, so in some cases only a fragment of the rod
        # is found. This results in the wrong force field types being
        # assigned to these atoms (UFF4MOF).
        for node in reference_nodes:
        #while reference_nodes:
            #node = reference_nodes.pop()
            data = self.node[node]
            possible_clusters = {}
            toln = tol
            if type=="Inorganic" and general_metal:
                for j in ref_sbus.keys():
                    possible_clusters.update(ref_sbus[j])
            else:
                possible_clusters.update(ref_sbus[data['element']])
            try:
                neighbour_nodes = []
                instanced_neighbours = self.neighbors(node)
                if (data['element'] == "C"):
                    chk_neighbors = num_neighbors # organic clusters can be much bigger.
                else:
                    chk_neighbors = num_neighbors
                # tree-like spanning of original node
                for j in range(chk_neighbors):
                    temp_neighbours = []
                    for n in instanced_neighbours:
                        if('special' not in self.node[n].keys()):
                            neighbour_nodes.append(n)
                            temp_neighbours += [j for j in self.neighbors(n) if j not in neighbour_nodes]
                    instanced_neighbours = temp_neighbours
                cluster_found = False
                # sort by descending number of nodes, this will ensure the largest SBU will be found
                # instead of a collection of smaller ones (e.g. multiple aromatic rings).
                clusters = [(cluster.number_of_nodes(), name, cluster) for name,cluster in possible_clusters.items()]
                for count, name, cluster in list(reversed(sorted(clusters))):
                    # ignore if it is impossible to find a clique with the current cluster.
                    if (len(neighbour_nodes)+1) < cluster.number_of_nodes():
                        continue
                    cg = self.correspondence_graph(cluster, toln, general_metal=general_metal, node_subset=neighbour_nodes + [node])
                    cliques = nx.find_cliques(cg)
                    #cliques = approximation.clique.max_clique(cg)
                    for clique in cliques:
                        #print(len(clique), cluster.number_of_nodes())
                        if len(clique) == cluster.number_of_nodes():
                            # found cluster
                            # update the 'hybridization' data
                            for i,j in clique:
                                self.node[i]['special_flag'] = cluster.node[j]['special_flag']
                            cluster_found = True
                            print("Found %s"%(name))
                            store_sbus.setdefault(name, []).append([i for (i,j) in clique])
                            #break

                    #if(cluster_found):
                    #    for n in neighbour_nodes:
                    #        try:
                    #            reference_nodes.pop(reference_nodes.index(n))
                    #        except:
                    #            pass
                    #    break
                    #else:
                    #    # put node back into the pool
                    #    reference_nodes.append(node)
                if not (cluster_found):
                    no_cluster.append(data['element'])
            except KeyError:
                # no recognizable clusters for element
                no_cluster.append(data['element'])
        for j in set(no_cluster):
            print ("No recognizable %s clusters for %i elements %s"%(type.lower(), no_cluster.count(j),  j))

    def redefine_lattice(self, redefinition, lattice):
        """Redefines the lattice based on the old lattice vectors. This was designed to convert
        non-orthogonal cells to orthogonal boxes, but it could in principle be used to
        convert any cell to any other cell. (As long as the redefined lattice
        are integer multiples of the old vectors)

        """
        #print(redefinition)
        #redefinition = np.array([[1., 0., 0.], [ -2., 2.,0.], [ -1., 3., 2.]])
        # determine how many replicas of the atoms is necessary to produce the supercell.
        vol_change = np.prod(np.diag(redefinition))
        if vol_change > 20:
            print("ERROR: The volume change is %i times greater than the unit cell. "%(vol_change) +
                    "I cannot process structures of this size!")
            sys.exit()

        print("The redefined cell will be %i times larger than the original."%(int(vol_change)))

        # replicate supercell
        sc = (tuple([int(i) for i in np.diag(redefinition)]))
        self.build_supercell(sc, lattice, redefine=redefinition)
        # re-define the cell
        old_cell = np.multiply(self.cell._cell.T, sc).T
        self.cell.set_cell(np.dot(redefinition, self.cell._cell))
        # the node cartesian_coordinates must be shifted by the periodic boundaries.
        for node, data in self.nodes_iter2(data=True):
            coord = data['cartesian_coordinates']
            data['cartesian_coordinates'] = self.in_cell(coord)

        # the bonds which span a periodic boundary will change
        for n1, n2, data in self.edges_iter2(data=True):
            flag = self.compute_bond_image_flag(n1, n2, self.cell)
            data['symflag'] = flag #'.'

        # not sure what this may break, but have to assume this new cell is the 'original'
        self.store_original_size()
    
    def subgraph(self,nodelist):
        """ Have to override subgraph from networkx. No idea why, but when Graph.subgraph is
        called in nx v >= 2.0, it returns a Graph object, not a MolecularGraph.

        """
        gg = deepcopy(self)
        delete_nodes = []
        for g in gg.nodes():
            if (not g in nodelist):
                delete_nodes.append(g)

        gg.remove_nodes_from(delete_nodes)
        return gg


    def build_supercell(self, sc, lattice, track_molecule=False, molecule_len=0, redefine=None):
        """Construct a graph with nodes supporting the size of the
        supercell (sc)
        Oh man.. so ugly.
        NB: this replaces and overwrites the original unit cell data
            with a supercell. There may be a better way to do this
            if one needs to keep both the super- and unit cells.
        """
        # preserve indices across molecules.
        unitatomlen = self.original_size
        totatomlen = nx.number_of_nodes(self)
        # keep a numerical index of the nodes.. this is to make sure that the molecules
        # are kept in their positions in the supercell (if replicated)
        unit_node_ids = sorted(self.nodes())
        origincell = np.array([0., 0., 0.])
        cells = list(itertools.product(*[itertools.product(range(j)) for j in sc]))
        maxcell = np.array(sc)
        rem_edges = []
        add_edges = []
        union_graphs = []
        orig_copy = deepcopy(self)
        for count, cell in enumerate(cells):
            newcell = np.array(cell).flatten()
            offset = count * unitatomlen
            mol_offset = count * molecule_len

            cartesian_offset = np.dot(newcell, lattice.cell)
            # Initial setup of new image in the supercell.
            if (count == 0):
                graph_image = self
            else:
                # rename nodes
                graph_image = nx.relabel_nodes(deepcopy(orig_copy), {unit_node_ids[i-1]: offset+unit_node_ids[i-1] for i in range(1, totatomlen+1)})
                graph_image.sorted_edge_dict = self.sorted_edge_dict.copy()
                # rename the edges with the offset associated with this supercell.
                for k,v in list(graph_image.sorted_edge_dict.items()):
                    newkey = (k[0] + offset, k[1] + offset)
                    newval = (v[0] + offset, v[1] + offset)
                    del graph_image.sorted_edge_dict[k]
                    graph_image.sorted_edge_dict.update({newkey:newval})

            # keep track of original index value from the unit cell.
            for i in range(1, totatomlen+1):
                graph_image.node[unit_node_ids[i-1]+offset]['image'] = unit_node_ids[i-1]
            if track_molecule:
                self.molecule_images.append(graph_image.nodes())
                graph_image.molecule_id = orig_copy.molecule_id + mol_offset
            # update cartesian coordinates for each node in the image
            for node in graph_image.nodes():
                data = graph_image.node[node] 
                n_orig = data['image']
                if track_molecule:
                    data['molid'] = graph_image.molecule_id
                data['cartesian_coordinates'] = data['cartesian_coordinates'] + cartesian_offset
                # update all angle and improper terms to the curent image indices. Dihedrals will be done in the edge loop
                # angle check
                try:
                    for (a, c), val in list(data['angles'].items()):
                        aid, cid = offset + a, offset + c
                        e_ba = graph_image[node][aid]
                        e_bc = graph_image[node][cid]
                        ba_symflag = e_ba['symflag']
                        order_ba = graph_image.sorted_edge_dict[(aid, node)]
                        if order_ba != (node, aid) and e_ba['symflag'] != '.':
                            ba_symflag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in e_ba['symflag'][2:]])))
                        bc_symflag = e_bc['symflag']
                        order_bc = graph_image.sorted_edge_dict[(node, cid)]
                        if order_bc != (node, cid) and e_bc['symflag'] != '.':
                            bc_symflag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in e_bc['symflag'][2:]])))
                        os_a = self.img_offset(cells, newcell, maxcell, ba_symflag, redefine) * unitatomlen
                        os_c = self.img_offset(cells, newcell, maxcell, bc_symflag, redefine) * unitatomlen
                        data['angles'].pop((a,c))
                        data['angles'][(a + os_a, c + os_c)] = val

                except KeyError:
                    # no angles for n1
                    pass
                # improper check
                try:
                    for (a, c, d), val in list(data['impropers'].items()):
                        aid, cid, did = offset + a, offset + c, offset + d
                        e_ba = graph_image[node][aid]
                        order_ba = graph_image.sorted_edge_dict[(node, aid)]
                        ba_symflag = e_ba['symflag']
                        e_bc = graph_image[node][cid]
                        order_bc = graph_image.sorted_edge_dict[(node, cid)]
                        bc_symflag = e_bc['symflag']
                        e_bd = graph_image[node][did]
                        order_bd = graph_image.sorted_edge_dict[(node, did)]
                        bd_symflag = e_bd['symflag']
                        if order_ba != (node, aid) and e_ba['symflag'] != '.':
                            ba_symflag = "1_%i%i%i"%(tuple([10 - int(j) for j in e_ba['symflag'][2:]]))
                        if order_bc != (node, cid) and e_bc['symflag'] != '.':
                            bc_symflag = "1_%i%i%i"%(tuple([10 - int(j) for j in e_bc['symflag'][2:]]))
                        if order_bd != (node, did) and e_bd['symflag'] != '.':
                            bd_symflag = "1_%i%i%i"%(tuple([10 - int(j) for j in e_bd['symflag'][2:]]))

                        os_a = self.img_offset(cells, newcell, maxcell, ba_symflag, redefine) * unitatomlen
                        os_c = self.img_offset(cells, newcell, maxcell, bc_symflag, redefine) * unitatomlen
                        os_d = self.img_offset(cells, newcell, maxcell, bd_symflag, redefine) * unitatomlen
                        data['impropers'].pop((a,c,d))
                        data['impropers'][(a + os_a, c + os_c, d + os_d)] = val

                except KeyError:
                    # no impropers for n1
                    pass

            # update nodes and edges to account for bonding to periodic images.
            #unique_translations = {}
            for v1, v2 in graph_image.edges():
                #data = graph_image.edges[(v1,v2)]
                data = graph_image[v1][v2]
                n1,n2 = graph_image.sorted_edge_dict[(v1,v2)]
                # flag boundary crossings, and determine updated nodes.
                # check symmetry flags if they need to be updated,
                n1_data = graph_image.node[n1]
                n2_data = graph_image.node[n2]
                try:
                    n1_orig = n1_data['image']
                    n2_orig = n2_data['image']
                except KeyError:
                    n1_orig = n1
                    n2_orig = n2
                # TODO(pboyd) the data of 'rings' for each node is not updated, do so if needed..
                # update angle, dihedral, improper indices.
                # dihedrals are difficult if the edge spans one of the terminal atoms..

                if (data['symflag'] != '.'):
                    # DEBUGGING
                    unit_repr = np.array([5, 5, 5], dtype=int)
                    translation = tuple(np.array([int(j) for j in data['symflag'][2:]]) - unit_repr)
                    #unique_translations.setdefault(translation,0)
                    #unique_translations[translation] += 1
                    # DEBUGGING
                    os_id = self.img_offset(cells, newcell, maxcell, data['symflag'], redefine, n1)
                    offset_c = os_id * unitatomlen
                    img_n2 = offset_c + n2_orig
                    #if (n1 == 1712):
                    #    print(os_id)
                    #    print(newcell)
                    #    print(redefine)
                    # pain...
                    opposite_flag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in data['symflag'][2:]])))
                    rev_n1_img = self.img_offset(cells, newcell, maxcell, opposite_flag, redefine) * unitatomlen + n1_orig
                    # dihedral check
                    try:
                        for (a, d), val in list(data['dihedrals'].items()):
                            # check to make sure edge between a, n1 is not crossing an image
                            edge_n1_a = orig_copy[n1_orig][a]
                            order_n1_a = graph_image.sorted_edge_dict[(n1, a+offset)]
                            n1a_symflag = edge_n1_a['symflag']

                            edge_n2_d = orig_copy[n2_orig][d]
                            order_n2_d = graph_image.sorted_edge_dict[(n2, d+offset)]
                            n2d_symflag = edge_n2_d['symflag']

                            offset_a = offset
                            if order_n1_a != (n1, a+offset) and edge_n1_a['symflag'] != '.':
                                n1a_symflag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in edge_n1_a['symflag'][2:]])))
                            if (edge_n1_a['symflag'] != '.'):
                                offset_a = self.img_offset(cells, newcell, maxcell, n1a_symflag, redefine) * unitatomlen
                            # check to make sure edge between n2, c is not crossing an image
                            offset_d = offset_c

                            if order_n2_d != (n2, d+offset) and edge_n2_d['symflag'] != '.':
                                n2d_symflag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in edge_n2_d['symflag'][2:]])))
                            if (edge_n2_d['symflag'] != '.'):
                                offset_d = self.img_offset(cells, np.array(cells[os_id]).flatten(), maxcell, n2d_symflag, redefine) * unitatomlen

                            aid, did = offset_a + a, offset_d + d
                            copyover = data['dihedrals'].pop((a,d))
                            data['dihedrals'][(aid, did)] = copyover
                    except KeyError:
                        # no dihedrals here.
                        pass

                    # Update symmetry flag of bond
                    data['symflag'] = self.update_symflag(newcell, data['symflag'], origincell, maxcell)
                    add_edges += [((n1, img_n2),data)]
                    rem_edges += [(n1, n2)]
                else:
                    # dihedral check
                    try:
                        for (a, d), val in list(data['dihedrals'].items()):
                            # check to make sure edge between a, n1 is not crossing an image
                            edge_n1_a = orig_copy[n1_orig][a]
                            order_n1_a = graph_image.sorted_edge_dict[(n1, a+offset)]
                            n1a_symflag = edge_n1_a['symflag']

                            edge_n2_d = orig_copy[n2_orig][d]
                            order_n2_d = graph_image.sorted_edge_dict[(n2, d+offset)]
                            n2d_symflag = edge_n2_d['symflag']

                            offset_a = offset
                            if order_n1_a != (n1, a+offset) and edge_n1_a['symflag'] != '.':
                                n1a_symflag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in edge_n1_a['symflag'][2:]])))
                            if (edge_n1_a['symflag'] != '.'):
                                offset_a = self.img_offset(cells, newcell, maxcell, n1a_symflag, redefine) * unitatomlen
                            # check to make sure edge between n2, c is not crossing an image
                            offset_d = offset

                            if order_n2_d != (n2, d+offset) and edge_n2_d['symflag'] != '.':
                                n2d_symflag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in edge_n2_d['symflag'][2:]])))
                            if (edge_n2_d['symflag'] != '.'):
                                offset_d = self.img_offset(cells, newcell, maxcell, n2d_symflag, redefine) * unitatomlen

                            aid, did = offset_a + a, offset_d + d
                            copyover = data['dihedrals'].pop((a,d))
                            data['dihedrals'][(aid, did)] = copyover
                    except KeyError:
                        # no dihedrals here.
                        pass
            if (count > 0):
                union_graphs.append(graph_image)
        for G in union_graphs:
            for node in G.nodes():
                data = G.node[node]
                self.add_node(node, **data)
           #once nodes are added, add edges.
        for G in union_graphs:
            self.sorted_edge_dict.update(G.sorted_edge_dict)
            for v1, v2 in G.edges():
                #d=G.edges[(v1,v2)]
                d=G[v1][v2]
                n1,n2 = G.sorted_edge_dict[(v1,v2)]
                self.add_edge(n1, n2, **d)

        for (n1, n2) in rem_edges:
            self.remove_edge(n1, n2)
            self.sorted_edge_dict.pop((n1,n2))
            self.sorted_edge_dict.pop((n2,n1))
        for (n1, n2), data in add_edges:
            self.add_edge(n1, n2, **data)
            self.sorted_edge_dict.update({(n1,n2):(n1,n2)})
            self.sorted_edge_dict.update({(n2,n1):(n1,n2)})
        #print(list(unique_translations.keys()))
    def unwrap_node_coordinates(self, cell):
        """Must be done before supercell generation.
        This is a recursive method iterating over all edges.
        The design is totally unpythonic and
        written in about 5 mins.. so be nice (PB)

        """
        supercells = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        # just use the first node as the unwrapping point..
        # probably a better way to do this to keep most atoms in the unit cell,
        # but I don't think it matters too much.
        nodelist = list(self.nodes())
        n1 = nodelist[0]
        queue = []
        while (nodelist or queue):
            for n2, data in self[n1].items():
                if n2 not in queue and n2 in nodelist:
                    queue.append(n2)
                    coord1 = self.node[n1]['cartesian_coordinates']
                    coord2 = self.node[n2]['cartesian_coordinates']
                    fcoords = np.dot(cell.inverse, coord2) + supercells

                    coords = np.array([np.dot(j, cell.cell) for j in fcoords])

                    dists = distance.cdist([coord1], coords)
                    dists = dists[0].tolist()
                    image = dists.index(min(dists))
                    self.node[n2]['cartesian_coordinates'] += np.dot(supercells[image], cell.cell)
                    data['symflag'] = '.'
            del nodelist[nodelist.index(n1)]
            try:
                n1 = queue[0]
                queue = queue[1:]

            except IndexError:
                pass

    def store_original_size(self):
        self.original_size = self.number_of_nodes()

    def __iadd__(self, newgraph):
        self.sorted_edge_dict.update(newgraph.sorted_edge_dict)
        for n, data in newgraph.nodes_iter2(data=True):
            self.add_node(n, **data)
        for n1,n2, data in newgraph.edges_iter2(data=True):
            self.add_edge(n1,n2, **data)
        return self

    def __or__(self, graph):
        if (len(graph.nodes()) == 1) and len(self.nodes()) == 1:
            return list([0])
        cg = self.correspondence_graph(graph, 0.4)
        cliques = list(nx.find_cliques(cg))
        cliques.sort(key=len)
        return cliques[-1]

def del_parenth(string):
    return re.sub(r'\([^)]*\)', '' , string)

def from_CIF(cifname):
    """Reads the structure data from the CIF
    - currently does not read the symmetry of the cell
    - does not unpack the assymetric unit (assumes P1)
    - assumes that the appropriate keys are in the cifobj (no error checking)
    """

    cifobj = CIF()
    cifobj.read(cifname)

    data = cifobj._data
    # obtain atoms and cell
    cell = Cell()
    # add data to molecular graph (to be parsed later..)
    mg = MolecularGraph(name=clean(cifname))
    cellparams = [float(del_parenth(i)) for i in [data['_cell_length_a'],
                                     data['_cell_length_b'],
                                     data['_cell_length_c'],
                                     data['_cell_angle_alpha'],
                                     data['_cell_angle_beta'],
                                     data['_cell_angle_gamma']]]
    cell.set_params(cellparams)

    #add atom nodes
    id = cifobj.block_order.index('atoms')
    atheads = cifobj._headings[id]
    for atom_data in zip(*[data[i] for i in atheads]):
        kwargs = {a:j.strip() for a, j in zip(atheads, atom_data)}
        mg.add_atomic_node(**kwargs)

    # add bond edges, if they exist
    try:
        id = cifobj.block_order.index('bonds')
        bondheads = cifobj._headings[id]
        for bond_data in zip(*[data[i] for i in bondheads]):
            kwargs = {a:j.strip() for a, j in zip(bondheads, bond_data)}
            mg.add_bond_edge(**kwargs)
    except:
        # catch no bonds
        print("No bonds reported in cif file - computing bonding..")
    mg.store_original_size()
    mg.cell = cell
    return cell, mg

def write_CIF(graph, cell):
    """Currently used for debugging purposes"""
    c = CIF(name="%s.debug"%graph.name)
    # data block
    c.add_data("data", data_=graph.name)
    c.add_data("data", _audit_creation_date=
                        CIF.label(c.get_time()))
    c.add_data("data", _audit_creation_method=
                        CIF.label("Lammps Interface v.%s"%(str(0))))

    # sym block
    c.add_data("sym", _symmetry_space_group_name_H_M=
                        CIF.label("P1"))
    c.add_data("sym", _symmetry_Int_Tables_number=
                        CIF.label("1"))
    c.add_data("sym", _symmetry_cell_setting=
                        CIF.label("triclinic"))

    # sym loop block
    c.add_data("sym_loop", _symmetry_equiv_pos_as_xyz=
                        CIF.label("'x, y, z'"))

    # cell block
    c.add_data("cell", _cell_length_a=CIF.cell_length_a(cell.a))
    c.add_data("cell", _cell_length_b=CIF.cell_length_b(cell.b))
    c.add_data("cell", _cell_length_c=CIF.cell_length_c(cell.c))
    c.add_data("cell", _cell_angle_alpha=CIF.cell_angle_alpha(cell.alpha))
    c.add_data("cell", _cell_angle_beta=CIF.cell_angle_beta(cell.beta))
    c.add_data("cell", _cell_angle_gamma=CIF.cell_angle_gamma(cell.gamma))
    # atom block
    element_counter = {}
    carts = []
    for node, data in graph.nodes_iter2(data=True):
        label = "%s%i"%(data['element'], node)
        c.add_data("atoms", _atom_site_label=
                                CIF.atom_site_label(label))
        c.add_data("atoms", _atom_site_type_symbol=
                                CIF.atom_site_type_symbol(data['element']))
        try:
            c.add_data("atoms", _atom_site_description=
                                CIF.atom_site_description(data['force_field_type']))
        except KeyError:
            c.add_data("atoms", _atom_site_description=
                                CIF.atom_site_description(data['element']))

        coords = data['cartesian_coordinates']
        carts.append(coords)
        fc = np.dot(cell.inverse, coords)
        c.add_data("atoms", _atom_site_fract_x=
                                CIF.atom_site_fract_x(fc[0]))
        c.add_data("atoms", _atom_site_fract_y=
                                CIF.atom_site_fract_y(fc[1]))
        c.add_data("atoms", _atom_site_fract_z=
                                CIF.atom_site_fract_z(fc[2]))
        try:
            c.add_data("atoms", _atom_type_partial_charge=
                                CIF.atom_type_partial_charge(data['charge']))
        except KeyError:
            c.add_data("atoms", _atom_type_partial_charge="0.0")
    # bond block
    # must re-sort them based on bond type (Mat Sudio)
    tosort = [(data['order'], (n1, n2, data)) for n1, n2, data in graph.edges_iter2(data=True)]
    for ord, (n1, n2, data) in sorted(tosort, key=lambda tup: tup[0]):
        try:
            type = CCDC_BOND_ORDERS[data['order']]
        except KeyError:
            # just assume single bond if not in CCDC_BOND_ORDERS dic
            type = "S" 
        dist = data['length']
        sym = data['symflag']


        label1 = "%s%i"%(graph.node[n1]['element'], n1)
        label2 = "%s%i"%(graph.node[n2]['element'], n2)
        c.add_data("bonds", _geom_bond_atom_site_label_1=
                                    CIF.geom_bond_atom_site_label_1(label1))
        c.add_data("bonds", _geom_bond_atom_site_label_2=
                                    CIF.geom_bond_atom_site_label_2(label2))
        c.add_data("bonds", _geom_bond_distance=
                                    CIF.geom_bond_distance(dist))
        c.add_data("bonds", _geom_bond_site_symmetry_2=
                                    CIF.geom_bond_site_symmetry_2(sym))
        c.add_data("bonds", _ccdc_geom_bond_type=
                                    CIF.ccdc_geom_bond_type(type))

    print('Output cif file written to %s.cif'%c.name)
    file = open("%s.cif"%c.name, "w")
    file.writelines(str(c))
    file.close()

def write_PDB(graph, cell):
    """
    Program to write pdb file from structure graph. Doing this because MS v. 6 doesn't do the CONECT lines
    correctly.

    """
    pdbfile=open("%s.debug.pdb"%(graph.name), "w")

    # remark section
    pdbfile.writelines("%6s %3i PDB file created by Lammps intervace v.%i\n"%("REMARK", 1, 0))
    pdbfile.writelines("%6s %3i Created on %s\n"%("REMARK",2,get_time()))
    # write crystallographic data 
    pdbfile.writelines("%6s%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f %11s%4i\n"%(
        "CRYST1",cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma,"P1",1))
    conect_string = ""
    for node, d in graph.nodes(data=True):  # nx2.1
        # 55 - 60 - occupancy
        # 61 - 66 temperature factor
        # 77 - 78 element symbol (left justified)
        # 79 - 80 charge on atom.
        try:
            label = "%s"%(d['ciflabel'])
        except:
            label = "%i%s"%(node,d['element'])
        cart = d['cartesian_coordinates']
        pdbfile.writelines("%-6s%5i %4s%1s%3s% 1s%3i%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n"
                               %("ATOM",        # 1 - 6: ATOM
                                  node,         # 7 - 11: atom number
                                  label,        # 13 - 16: atom name
                                  " ",          # 17: alternate location id. ????
                                  "MOL",        # 18 - 20: residue name
                                  " ",          # 22: chainId?
                                  2,            # 23 - 26: residue sequence number
                                  " ",          # 27: code for insertion of residues
                                  cart[0],      # 31 - 38: real 8.3 x
                                  cart[1],      # 39 - 46: real 8.3 y
                                  cart[2],      # 47 - 54: real 8.3 z
                                  1.0,          # 55 - 60: occupancy
                                  0.0,          # 61 - 66: temperature factor
                                  d['element'], # 77 - 78: element symbol (left justified)
                                  "  ",         # 79 - 80: charge on atom.
                                  ))
        # list bonded atoms, 4 per line.

        neighbs = sorted(graph.neighbors(node), reverse=True)
        remove = []
        # delete connected atoms that cross periodic boundary, cause vis software for 
        # pdb files suck.
        for idx,n in enumerate(neighbs):
            bonddata = graph.get_edge_data(node,n)
            if bonddata['symflag'] != '.':
                remove.append(idx)
        for i in sorted(remove, reverse=True):
            del neighbs[i]

        num_lines = math.ceil(len(neighbs)/4.)

        for i in range(num_lines):
            try:
                neighbour1 = "%5i"%(neighbs.pop())
            except IndexError:
                neighbour1 = "%5s"%(" ")
            try:
                neighbour2 = "%5i"%(neighbs.pop())
            except IndexError:
                neighbour2 = "%5s"%(" ")
            try:
                neighbour3 = "%5i"%(neighbs.pop())
            except IndexError:
                neighbour3 = "%5s"%(" ")
            try:
                neighbour4 = "%5i"%(neighbs.pop())
            except IndexError:
                neighbour4 = "%5s"%(" ")

            
            conect_string += "%6s%5s%5s%5s%5s%5s\n"%(
              "CONECT",                         # 1 - 6: CONECT
              node,                             # 7 - 11: atom serial number
              neighbour1,                       # 12 - 16: serial number of bonded atom
              neighbour2,                       # 17 - 21: serial number of bonded atom
              neighbour3,                       # 22 - 26: serial number of bonded atom
              neighbour4,                       # 27 - 31: serial number of bonded atom
              )
    pdbfile.writelines("%6s%5i\n"%("TER", graph.number_of_nodes() + 1))
    pdbfile.writelines(conect_string)
    pdbfile.writelines("END")
    pdbfile.close()
    print('Output file written to %s.debug.pdb'%graph.name)

def write_RASPA_CIF(graph, cell,classifier=0):
    """
    Same as debugging cif write routine
        except _atom_site_label is now equal to _atom_site_description
        b/c RASPA uses _atom_site_label as the type for assigning FF params
    """
    c = CIF(name="%s_raspa"%graph.name)
    # data block
    c.add_data("data", data_=graph.name)
    c.add_data("data", _audit_creation_date=
                        CIF.label(c.get_time()))
    c.add_data("data", _audit_creation_method=
                        CIF.label("Lammps Interface v.%s"%(str(0))))

    # sym block
    c.add_data("sym", _symmetry_space_group_name_H_M=
                        CIF.label("P1"))
    c.add_data("sym", _symmetry_Int_Tables_number=
                        CIF.label("1"))
    c.add_data("sym", _symmetry_cell_setting=
                        CIF.label("triclinic"))

    # sym loop block
    c.add_data("sym_loop", _symmetry_equiv_pos_as_xyz=
                        CIF.label("'x, y, z'"))

    # cell block
    c.add_data("cell", _cell_length_a=CIF.cell_length_a(cell.a))
    c.add_data("cell", _cell_length_b=CIF.cell_length_b(cell.b))
    c.add_data("cell", _cell_length_c=CIF.cell_length_c(cell.c))
    c.add_data("cell", _cell_angle_alpha=CIF.cell_angle_alpha(cell.alpha))
    c.add_data("cell", _cell_angle_beta=CIF.cell_angle_beta(cell.beta))
    c.add_data("cell", _cell_angle_gamma=CIF.cell_angle_gamma(cell.gamma))
    # atom block
    element_counter = {}
    carts = []

    # slight modification to make sure atoms printed out in same order as in data and original cif
    for node, data in sorted(graph.nodes_iter2(data=True)):
        label = "%s%i"%(data['element'], node)

        # sometimes we need to keep the original CIF label in the case
        # of slab geometries where charge symmetry is very reduced
        if(classifier==0):
            c.add_data("atoms", _atom_site_label=
                              CIF.atom_site_label(data['force_field_type']))
        elif(classifier==1):
            c.add_data("atoms", _atom_site_label=
                              CIF.atom_site_label(data['ciflabel']))

        c.add_data("atoms", _atom_site_type_symbol=
                                CIF.atom_site_type_symbol(data['element']))
        #c.add_data("atoms", _atom_site_description=
        #                        CIF.atom_site_description(data['force_field_type']))
        coords = data['cartesian_coordinates']
        carts.append(coords)
        fc = np.dot(cell.inverse, coords)
        c.add_data("atoms", _atom_site_fract_x=
                                CIF.atom_site_fract_x(fc[0]))
        c.add_data("atoms", _atom_site_fract_y=
                                CIF.atom_site_fract_y(fc[1]))
        c.add_data("atoms", _atom_site_fract_z=
                                CIF.atom_site_fract_z(fc[2]))
        c.add_data("atoms", _atom_site_charge=
                                CIF.atom_type_partial_charge(data['charge']))
    # bond block
    # must re-sort them based on bond type (Mat Sudio)
    #tosort = [(data['order'], (n1, n2, data)) for n1, n2, data in graph.edges_iter2(data=True)]
    #for ord, (n1, n2, data) in sorted(tosort, key=lambda tup: tup[0]):
    #    type = CCDC_BOND_ORDERS[data['order']]
    #    dist = data['length']
    #    sym = data['symflag']


    #    label1 = "%s%i"%(graph.node[n1]['element'], n1)
    #    label2 = "%s%i"%(graph.node[n2]['element'], n2)
    #    c.add_data("bonds", _geom_bond_atom_site_label_1=
    #                                CIF.geom_bond_atom_site_label_1(label1))
    #    c.add_data("bonds", _geom_bond_atom_site_label_2=
    #                                CIF.geom_bond_atom_site_label_2(label2))
    #    c.add_data("bonds", _geom_bond_distance=
    #                                CIF.geom_bond_distance(dist))
    #    c.add_data("bonds", _geom_bond_site_symmetry_2=
    #                                CIF.geom_bond_site_symmetry_2(sym))
    #    c.add_data("bonds", _ccdc_geom_bond_type=
    #                                CIF.ccdc_geom_bond_type(type))

    print('Output cif file written to %s.cif'%c.name)
    file = open("%s.cif"%c.name, "w")
    file.writelines(str(c))
    file.close()



def write_RASPA_sim_files(lammps_sim,classifier=0):
    """
    Write the RASPA pseudo_atoms.def file for this MOF
    All generic adsorbates info automatically included
    Additional framework atoms taken from lammps_interface anaylsis


    classifier = 0 -> UFF atom label is used for pseudo atoms
    classifier = 1 -> cif label is used for pseudo atoms (useful for
                      interface/defect systems where symmetry of charge density
                      is very reduced)

    """

    MOF_PSEUDO_ATOMS = []
    MOF_FF_MIXING = []

    data_list = []
    max_image = 0
    if(classifier == 0):
        for node, data in sorted(lammps_sim.unique_atom_types.items()):
            data['node']=node
            print(key)
            print(data)
            data_list.append(data)

            if(int(data['image']) > max_image):
                max_image = int(data['image'])
            #keys.append(lammps_sim.graph.node[n]['force_field_type'])

    elif(classifier == 1):
        for node, data in sorted(lammps_sim.graph.nodes_iter2(data=True)):
            data['node']=node
            print(node)
            print(data)
            data_list.append(data)

            if(int(data['image']) > max_image):
                max_image = int(data['image'])
            #keys.append(lammps_sim.graph.node[n]['force_field_type'])

    print(max_image)
    for i in range(len(data_list)):

        #ind = 0
        ##for char in key:
        #    # identify first non alphabetic character in string
        #    if((ord(char) >= 65 and ord(char) <= 90) or (ord(char)>=97 and ord(char) <= 122)):
        #        ind += 1
        #    else:
        #        break
        #atmtype_ = key[:ind]
        #print(key,atmtype_)
        #print(atmtype_ in MASS)
        #print(atmtype_ in COVALENT_RADII)
        #print(key in UFF_DATA)

        if(data_list[i]['node']<max_image):
            element_ = data_list[i]['element']
            fftype_ = data_list[i]['force_field_type']

            try:
                if(classifier==0):
                    type_spec_ = data_list[i]['force_field_type']
                elif(classifier==1):
                    type_spec_ = data_list[i]['ciflabel']

                print_ = 'yes'
                as_ = element_
                chem_ = element_
                oxidation_ = '0'
                mass_ = str(MASS[element_])
                charge_ = str(data_list[i]['charge'])
                polarization_ = '0.0'
                B_factor_ = '1.0'
                radii_ = str(COVALENT_RADII[element_])
                connectivity_ = '0'
                anisotropic_ = '0'
                anisotropic_type_ = 'relative'
                tinker_type_ = '0'

                potential_ = 'lennard-jones'
                eps_ = str(UFF_DATA[fftype_][3]*500)
                sig_ = str(UFF_DATA[fftype_][2]*(2**(-1./6.)))
            except:
                print("%s %s not found!\n"%(element_, fftype_))
                continue

            # finally, add strings to list for each RASPA file
            MOF_PSEUDO_ATOMS.append([type_spec_, print_, as_,chem_, oxidation_,mass_, charge_, \
                                     polarization_, B_factor_, radii_, connectivity_, \
                                     anisotropic_, anisotropic_type_, tinker_type_])

            MOF_FF_MIXING.append([type_spec_, potential_, eps_, sig_])

    if(len(MOF_PSEUDO_ATOMS) == 0):
        print("Error! No MOF atoms found. Exiting...")
        sys.exit()

    # Determine final column widths
    col_widths = [0 for i in range(len(MOF_PSEUDO_ATOMS[0]))]
    for i in range(len(MOF_PSEUDO_ATOMS[0])):
        # num columns
        max_width = 0
        for j in range(len(MOF_PSEUDO_ATOMS)):
            # num rows
            if(len(MOF_PSEUDO_ATOMS[j][i]) > max_width):
                max_width = len(MOF_PSEUDO_ATOMS[j][i])
        col_widths[i] = max_width + 2

    col_widths1 = [0 for i in range(len(GENERIC_PSEUDO_ATOMS[0]))]
    for i in range(len(GENERIC_PSEUDO_ATOMS[0])):
        # num columns
        max_width = 0
        for j in range(len(GENERIC_PSEUDO_ATOMS)):
            # num rows
            if(len(GENERIC_PSEUDO_ATOMS[j][i]) > max_width):
                max_width = len(GENERIC_PSEUDO_ATOMS[j][i])
        col_widths1[i] = max_width + 2

    col_widths_final = [max(col_widths[i], col_widths1[i]) for i in range(len(col_widths))]


    # Begin file writing
    f = open('pseudo_atoms.def','w')

    # write header
    num_psuedo_atoms =len(GENERIC_PSEUDO_ATOMS) + len(MOF_PSEUDO_ATOMS)
    GENERIC_PSEUDO_ATOMS_HEADER[1] = str(num_psuedo_atoms)
    for line in GENERIC_PSEUDO_ATOMS_HEADER:
        f.write("".join(word for word in line) +'\n')

    # write this MOFs pseudo atoms
    for i in range(len(MOF_PSEUDO_ATOMS)):
        base_string = ""
        for j in range(len(MOF_PSEUDO_ATOMS[0])):
            buff = "".join(" " for i in range(col_widths_final[j] - len(MOF_PSEUDO_ATOMS[i][j])))

            base_string += MOF_PSEUDO_ATOMS[i][j]
            base_string += buff
        f.write(base_string + '\n')

    # write the generic adsorbates
    for i in range(len(GENERIC_PSEUDO_ATOMS)):
        base_string = ""
        for j in range(len(GENERIC_PSEUDO_ATOMS[0])):
            buff = "".join(" " for i in range(col_widths_final[j] - len(GENERIC_PSEUDO_ATOMS[i][j])))

            base_string += GENERIC_PSEUDO_ATOMS[i][j]
            base_string += buff
        f.write(base_string + '\n')

    f.close()


    # Determine column widths for FF MIXING
    col_widths = [0 for i in range(len(MOF_FF_MIXING[0]))]
    for i in range(len(MOF_FF_MIXING[0])):
        # num columns
        max_width = 0
        for j in range(len(MOF_FF_MIXING)):
            # num rows
            if(len(MOF_FF_MIXING[j][i]) > max_width):
                max_width = len(MOF_FF_MIXING[j][i])
        col_widths[i] = max_width + 2

    col_widths1 = [0 for i in range(len(GENERIC_FF_MIXING[0]))]
    for i in range(len(GENERIC_FF_MIXING[0])):
        # num columns
        max_width = 0
        for j in range(len(GENERIC_FF_MIXING)):
            # num rows
            if(len(GENERIC_FF_MIXING[j][i]) > max_width):
                max_width = len(GENERIC_FF_MIXING[j][i])
        col_widths1[i] = max_width + 2

    col_widths_final = [max(col_widths[i], col_widths1[i]) for i in range(len(col_widths))]

    # write ff mixing file
    f = open('force_field_mixing_rules.def','w')

    # write header
    num_interactions = len(MOF_FF_MIXING) + len(GENERIC_FF_MIXING)
    GENERIC_FF_MIXING_HEADER[5] = str(num_interactions)
    for line in GENERIC_FF_MIXING_HEADER:
        f.write("".join(word for word in line) +'\n')


    # write this MOFs pseudo atoms
    for i in range(len(MOF_FF_MIXING)):
        base_string = ""
        for j in range(len(MOF_FF_MIXING[0])):
            buff = "".join(" " for i in range(col_widths_final[j] - len(MOF_FF_MIXING[i][j])))

            base_string += MOF_FF_MIXING[i][j]
            base_string += buff
        f.write(base_string + '\n')

    # write the generic adsorbates
    for i in range(len(GENERIC_FF_MIXING)):
        base_string = ""
        for j in range(len(GENERIC_FF_MIXING[0])):
            buff = "".join(" " for i in range(col_widths_final[j] - len(GENERIC_FF_MIXING[i][j])))

            base_string += GENERIC_FF_MIXING[i][j]
            base_string += buff
        f.write(base_string + '\n')

    for line in GENERIC_FF_MIXING_FOOTER:
        f.write("".join(word for word in line) +'\n')

    f.close()


class MDMC_config(object):
    """
    Very sloppy for now but just doing the bare minimum to get this up and running
    for methane in flexible materials
    """

    def __init__(self, lammps_sim):

        try:
            f = open("MDMC.config","r")
        except:
            self.initialized = False
            print("Warning! No MDMC.config file found.  LAMMPS sim files will not have guest molecule info")
            return

        lines = f.readlines()

        outlines = ""
        for line in lines:
            parsed = line.strip().split()

            if(parsed[0] == "num_framework"):
                outlines += "num_framework\t%d\n"%(nx.number_of_nodes(lammps_sim.graph))
            if(parsed[0] == "type_framework"):
                outlines += "type_framework\t%d\n"%(len(lammps_sim.unique_atom_types.keys()))
            if(parsed[0] == "type_guest"):
                self.type_guest = int(parsed[1])
                outlines += "type_guest\t%d\n"%(self.type_guest)
            if(parsed[0] == "pair_coeff"):
                parsed[1] = str(len(lammps_sim.unique_atom_types.keys()) + self.type_guest)
                for word in parsed:
                    outlines += word + " "
                outlines += "\n"
            if(parsed[0] == "mass_guest"):
                parsed[1] = str(len(lammps_sim.unique_atom_types.keys()) + self.type_guest)
                for word in parsed:
                    outlines += word + " "
                outlines += "\n"

        f.close()

        f = open("MDMC.config", "w")
        f.write(outlines)
        f.close()

        return


class Cell(object):
    def __init__(self):
        self._cell = np.identity(3, dtype=np.float64)
        # cell parameters (a, b, c, alpha, beta, gamma)
        self._params = (1., 1., 1., 90., 90., 90.)
        self._inverse = None

    @property
    def volume(self):
        """Calculate cell volume a.bxc."""
        b_cross_c = np.cross(self.cell[1], self.cell[2])
        return np.dot(self.cell[0], b_cross_c)

    def get_cell(self):
        """Get the 3x3 vector cell representation."""
        return self._cell

    def get_cell_inverse(self):
        """Get the 3x3 vector cell representation."""
        return self._inverse

    def mod_to_UC(self, num):
        """
        Retrun any fractional coordinate back into the unit cell
        """
        if(hasattr(num,'__iter__')):
            for i in range(len(num)):
                if(num[i] < 0.0):
                    num[i] = 1+math.fmod(num[i], 1.0)
                else:
                    num[i] = math.fmod(num[i], 1.0)

            return num
        else:
            if(num < 0.0):
                num = math.fmod((num*(-1)), 1.0)
            else:
                num = math.fmod(num, 1.0)

    def set_cell(self, value):
        """Set cell and params from the cell representation."""
        # Class internally expects an array
        self._cell = np.array(value).reshape((3,3))
        self.__mkparam()
        self.__mklammps()
        # remake cell so a in x, b in xy and c in xyz
        self.__mkcell()
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
        """Calculate the smallest supercell with a half-cell width cutoff.

        Increment from smallest cell vector to largest. So the supercell
        is not considering the 'unit cell' for each cell dimension.

        """

        a_cross_b = np.cross(self.cell[0], self.cell[1])
        b_cross_c = np.cross(self.cell[1], self.cell[2])
        c_cross_a = np.cross(self.cell[2], self.cell[0])

        #volume = np.dot(self.cell[0], b_cross_c)

        widths = [np.dot(self.cell[0], b_cross_c) / np.linalg.norm(b_cross_c),
                  np.dot(self.cell[1], c_cross_a) / np.linalg.norm(c_cross_a),
                  np.dot(self.cell[2], a_cross_b) / np.linalg.norm(a_cross_b)]

        return tuple(int(math.ceil(2*cutoff/x)) for x in widths)
        #diag = np.diag(self.cell)
        ##print(np.ceil(cutoff/diag*2.))
        #return tuple(int(i) for i in np.ceil(cutoff/diag*2.))

    def orthogonal_transformation(self):
        """Compute the transformation from the original unit cell to a supercell which
        has 90 degree angles between it's basis vectors. This is somewhat approximate,
        and the angles will not be EXACTLY 90 deg.

        """
        absmat = np.abs(self._inverse.T)
        zero_tol = 0.01
        # round all near - zero values to zero
        absmat[np.where(np.allclose(absmat, 0., atol=zero_tol))] = 0.
        divs = np.array([np.min(absmat[i, np.nonzero(absmat[i])]) for i in range(3)])

        M = np.around(self._inverse.T / divs[:,None])
        MN = M.copy()
        return MN

    def update_supercell(self, tuple):
        self._cell = np.multiply(self._cell.T, tuple).T
        self.__mkparam()
        self.__mklammps()
        self._inverse = np.linalg.inv(self._cell.T)

    @property
    def minimum_width(self):
        """The shortest perpendicular distance within the cell."""
        a_cross_b = np.cross(self.cell[0], self.cell[1])
        b_cross_c = np.cross(self.cell[1], self.cell[2])
        c_cross_a = np.cross(self.cell[2], self.cell[0])

        volume = np.dot(self.cell[0], b_cross_c)

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
        cell_a = np.sqrt(sum(x**2 for x in self.cell[0]))
        cell_b = np.sqrt(sum(x**2 for x in self.cell[1]))
        cell_c = np.sqrt(sum(x**2 for x in self.cell[2]))
        alpha = np.arccos(sum(self.cell[1, :] * self.cell[2, :]) /
                       (cell_b * cell_c)) * 180 / np.pi
        beta = np.arccos(sum(self.cell[0, :] * self.cell[2, :]) /
                      (cell_a * cell_c)) * 180 / np.pi
        gamma = np.arccos(sum(self.cell[0, :] * self.cell[1, :]) /
                       (cell_a * cell_b)) * 180 / np.pi
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

def clean(name):
    name = os.path.split(name)[-1]
    if name.endswith('.cif'):
        name = name[:-4]
    return name
