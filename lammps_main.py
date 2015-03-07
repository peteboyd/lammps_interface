#!/usr/bin/env python
"""
main.py

the program starts here.

"""

# Turn on keyword expansion to get revision numbers in version strings
# in .hg/hgrc put
# [extensions]
# keyword =
#
# [keyword]
# lammps_main.py =
#
# [keywordmaps]
# Revision = {rev}


try:
    __version_info__ = (0, 0, 0, int("$Revision$".strip("$Revision: ")))
except ValueError:
    __version_info__ = (0, 0, 0, 0)
__version__ = "%i.%i.%i.%i"%__version_info__

import sys
from structure_data import CIF, Structure
from ForceFields import UFF

def write_data_file(ff):

    string = "Created on %s\n\n"%t.strftime("%a %b %d %H:%M:%S %Y %Z")

    string += "%12i atoms\n"%(len(ff.structure.atoms))
    string += "%12i bonds\n"%(len(ff.structure.bonds))
    string += "%12i angles\n"%(len(ff.structure.angles))
    string += "%12i dihedrals\n"%(len(ff.structure.dihedrals))
    string += "%12i impropers\n\n"%(len(ff.structure.impropers))

    string += "%12i atom types\n"%(len(ff.unique_atom_types.keys()))
    string += "%12i bond types\n"%(len(ff.unique_bond_types.keys()))
    string += "%12i angle types\n"%(len(ff.unique_angle_types.keys()))
    string += "%12i dihedral types\n"%(len(ff.unique_dihedral_types.keys()))
    string += "%12i improper types\n"%(len(ff.unique_improper_types.keys()))

    cell = ff.structure.cell
    string += "%19.6f %10.6f %s %s\n"%(0., cell.lx, "xlo", "xhi")
    string += "%19.6f %10.6f %s %s\n"%(0., cell.ly, "ylo", "yhi")
    string += "%19.6f %10.6f %s %s\n"%(0., cell.lz, "zlo", "zhi")
    string += "%19.6f %10.6f %10.6f %s %s %s\n"%(cell.xy, cell.xz, cell.yz, "xy", "xz", "yz")

    string += "\nMasses\n\n"
    for key in sorted(ff.unique_atom_types.keys()):
        unq_atom = ff.unique_atom_types[key]
        mass, type = unq_atom.mass, unq_atom.force_field_type
        string += "%5i %8.4f # %s\n"%(key, mass, type)

    string += "\nBond Coeffs\n\n"
    for key in sorted(ff.unique_bond_types.keys()):
        bond = ff.unique_bond_types[key] 
        uff1, uff2 = bond.atoms[0].force_field_type, bond.atoms[1].force_field_type
        K = bond.parameters[0]
        R = bond.parameters[1]
        string += "%5i %15.3f %15.2f # %s %s\n"%(key, K, R, uff1, uff2)

    string += "\nAngle Coeffs\n\n"
    for key in sorted(ff.unique_angle_types.keys()):
        angle = ff.unique_angle_types[key]
        atom_a, atom_b, atom_c = angle.atoms
        #type_a, type_b, type_c = atom_a.ff_type_index, atom_b.ff_type_index, atom_c.ff_type_index

        unq_l, unq_c, unq_r, function, kappa, c0, c1, c2 = unq_angles[key]
        if angle.function == 'fourier':
            kappa, c0, c1, c2 = angle.parameters     
            string += "%5i %s %15.3f %15.3f %15.3f %15.3f # %s %s %s\n"%(key, angle.function, kappa, c0, c1, c2,
                                                          atom_a.force_field_type,
                                                          atom_b.force_field_type,
                                                          atom_c.force_field_type)

        elif angle.function == 'fourier/simple':
            kappa, c0, c1 = angle.parameters     
            string += "%5i %s %15.3f %15.3f %15.3f # %s %s %s\n"%(key, angle.function, kappa, c0, c1,
                                                          atom_a.force_field_type,
                                                          atom_b.force_field_type,
                                                          atom_c.force_field_type)

    string +=  "\nDihedral Coeffs\n\n"
    for key in sorted(ff.unique_dihedral_types.keys()):
        dihedral = ff.unique_dihedral_types[key]
        atom_a, atom_b, atom_c, atom_d = dihedral.atoms
        V, d, n = dihedral.parameters
        dihedral.parameters = (0.5*V, math.cos(nphi0*DEG2RAD), n)
        string += "%5i %15.6f %5i %15i # %s %s %s %s\n"%(key, V, d, n,
                                              atom_a.force_field_type,
                                              atom_b.force_field_type,
                                              atom_c.force_field_type,
                                              atom_d.force_field_type)

    string += "\nImproper Coeffs\n\n"
    for key in sorted(ff.unique_improper_terms.keys()):
        improper = ff.unique_improper_terms[key]
        atom_a, atom_b, atom_c, atom_d = improper.atoms  
        k, c0, c1, c2 = improper.parameters 
        unq_a, unq_b, unq_c, unq_d, csi0, kcsi = unq_impropers[key]
        string += "%5i %15.6f %15.6f %15.6f %15.6f 1. # %s %s %s %s\n"%(key, k, c0, c1, c2
                                              atom_a.force_field_type,
                                              atom_b.force_field_type,
                                              atom_c.force_field_type,
                                              atom_d.force_field_type)

def main():
    print __version__

    cif = CIF(name="test")
    # NB can add the filename as the second argument of the class instance,
    # or from a separate function

    # set as the first argument for testing
    cif.read(sys.argv[1])

    struct = Structure(name='test')
    struct.from_CIF(cif)
    struct.compute_angles()
    struct.compute_dihedrals()
    struct.compute_improper_dihedrals()
    ff = UFF(struct)
    ff.compute_force_field_terms()


if __name__ == "__main__": 
    main()

