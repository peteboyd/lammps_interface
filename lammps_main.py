#!/usr/bin/env python
"""
main.py

the program starts here.

"""

import sys
import os
import math
import ForceFields
import itertools
import operator
from structure_data import CIF, Structure
from datetime import datetime
from InputHandler import Options


def construct_data_file(ff):

    t = datetime.today()
    string = "Created on %s\n\n"%t.strftime("%a %b %d %H:%M:%S %Y %Z")

    if(len(ff.unique_atom_types.keys()) > 0):
        string += "%12i atoms\n"%(len(ff.structure.atoms))
    if(len(ff.unique_bond_types.keys()) > 0):
        string += "%12i bonds\n"%(len(ff.structure.bonds))
    if(len(ff.unique_angle_types.keys()) > 0):
        string += "%12i angles\n"%(len(ff.structure.angles))
    if(len(ff.unique_dihedral_types.keys()) > 0):
        string += "%12i dihedrals\n"%(len(ff.structure.dihedrals))
    if (len(ff.unique_improper_types.keys()) > 0):
        string += "%12i impropers\n\n"%(len(ff.structure.impropers))

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

    cell = ff.structure.cell
    string += "%19.6f %10.6f %s %s\n"%(0., cell.lx, "xlo", "xhi")
    string += "%19.6f %10.6f %s %s\n"%(0., cell.ly, "ylo", "yhi")
    string += "%19.6f %10.6f %s %s\n"%(0., cell.lz, "zlo", "zhi")
    string += "%19.6f %10.6f %10.6f %s %s %s\n"%(cell.xy, cell.xz, cell.yz, "xy", "xz", "yz")


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
                string += "%5i %s "%(key, angle.potential)
                string += "# %s %s %s\n"%(atom_a.force_field_type, 
                                          atom_b.force_field_type, 
                                          atom_c.force_field_type)
    
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
                string += "%5i %s "%(key, dihedral.potential)
                string += "# %s %s %s %s\n"%(atom_a.force_field_type, 
                                             atom_b.force_field_type, 
                                             atom_c.force_field_type, 
                                             atom_d.force_field_type)

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
                string += "%5i %s "%(key, improper.potential)
                string += "# %s %s %s %s\n"%(atom_a.force_field_type, 
                                             atom_b.force_field_type, 
                                             atom_c.force_field_type, 
                                             atom_d.force_field_type)

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

def construct_input_file(ff):
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
    inp_str += "%-15s %s\n"%("minimize","1.0e-4 1.0e-6 10000 100000")
    inp_str += "%-15s %s\n"%("fix","1 all box/relax tri 0.0 vmax 0.01")
    inp_str += "%-15s %s\n"%("minimize","1.0e-4 1.0e-6 10000 100000")
    inp_str += "%-15s %s\n"%("unfix", "1")
    inp_str += "%-15s %s\n"%("minimize","1.0e-4 1.0e-6 10000 100000")

    return inp_str

def clean(name):
    name = os.path.split(name)[-1]
    if name.endswith('.cif'):
        name = name[:-4]
    return name

def groups(ints):
    ints = sorted(ints)
    for k, g in itertools.groupby(enumerate(ints), lambda ix : ix[0]-ix[1]):
        yield list(map(operator.itemgetter(1), g))

def main():

    # command line parsing
    options = Options()

    mofname = clean(options.cif_file)
    struct = Structure(name=mofname)
    struct.from_CIF(options.cif_file)
    # compute minimum supercell
    # NB: half box width should be a user-defined command,
    # or default to 2.5*sigma_max of the requested force field
    # currently defaults to 12.5 anstroms
    #obtain desired force field class from command line, parse 
    # string to real class value
    try:
        ff = getattr(ForceFields, options.force_field)(struct)
    except AttributeError:
        print("Error: could not find the force field: %s"%options.force_field)
        sys.exit()
    ff.detect_ff_terms() 
    ff.cutoff = options.cutoff
    ff.structure.minimum_cell(cutoff=options.cutoff)
    ff.structure.compute_molecules()
    if options.output_cif:
        print("output of .cif file requested. Exiting.")
        struct.write_cif()
        sys.exit()

    struct.compute_angles()
    struct.compute_dihedrals()
    struct.compute_improper_dihedrals()

    
#    if (options.force_field == "BTW_FF"):
#        struct.assign_ff_charges() # if the force field assings charges for different atoms

    # doesn't really follow the logic of the program, but can only be done 
    # when the atoms have been assigned to a force field.

    ff.compute_force_field_terms()
    data_str = construct_data_file(ff) 
    inp_str = construct_input_file(ff)
   
    datafile = open("data.%s"%struct.name, 'w')
    datafile.writelines(data_str)
    datafile.close()

    inpfile = open("in.%s"%struct.name, 'w')
    inpfile.writelines(inp_str)
    inpfile.close()
    print("files created!")

if __name__ == "__main__": 
    main()

