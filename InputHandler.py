#!/usr/bin/env python

from argparse import ArgumentParser
import os
import subprocess
def git_revision_hash():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    wrk_dir = os.getcwd()
    os.chdir(src_dir)
    rev_no = len(subprocess.check_output(['git', 'rev-list', 'HEAD'], universal_newlines=True).strip().split("\n")) 
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], universal_newlines=True).strip()
    os.chdir(wrk_dir)
    return (rev_no, commit)
rev_no, commit = git_revision_hash()
__version_info__ = (0, 0, rev_no, "%s"%commit)
__version__ = "%i.%i.%i.%s"%__version_info__

class Options(object):

    def __init__(self):
        #print("Lammps_interface version: %s"%__version__)
        self.run_command_line_options()

    def run_command_line_options(self):
        parser = ArgumentParser(description="LAMMPS interface :D", prog="lammps_interface")
        parser.add_argument("-V", "--version",
                            action="version",
                            version="%(prog)s version "+__version__)
        parser.add_argument("-o", "--outputcif",
                            action="store_true",
                            dest="output_cif",
                            help="Write a .cif file for visualization."+
                                 " Necessary for debugging purposes, this"+
                                 " file can show the user how the structure "+
                                 "has been interpreted by the program.")
        parser.add_argument("-t", "--tolerance",
                            action="store",
                            type=float,
                            default=0.4,
                            dest="tol",
                            help="Tolerance in angstroms to determine "+
                                 "detection of inorganic clusters.")

        #split the command line options into separate groups for nicer
        #visualization.
        force_field_group = parser.add_argument_group("Force Field options")
        force_field_group.add_argument("-ff", "--force_field", action="store", 
                                       type=str, dest="force_field",
                                       default="UFF",
                                       help="Enter the requested force "+
                                          "field to describe the system."+
                                          " The default is the Universal "+
                                          "Force Field [UFF].")
        force_field_group.add_argument("--h-bonding", action="store_true", 
                                       dest="h_bonding",
                                       default=False,
                                       help="Add hydrogen bonding potentials "+
                                          "to the force field characterization."+
                                          " Currently only applies to Dreiding. "+
                                          "Default is off.")

        simulation_group = parser.add_argument_group("Simulation options")
        simulation_group.add_argument("--minimize", action="store_true",
                                      dest="minimize",
                                      default=True,
                                      help="Request input files necessary for"
                                      + " a geometry optimization.")
        simulation_group.add_argument("--cutoff", action="store",
                                      type=float, dest="cutoff",
                                      default=12.5,
                                      help="Set the long-range cutoff "+
                                      "to this value in Angstroms." + 
                                      " This will determine the size of "+
                                      "the supercell computed for the simulation.")


        parser.add_argument(metavar="CIF", dest="cif_file",
                            help="path to cif file to interpret")

        args = vars(parser.parse_args())
        self._set_attr(args)

    def _set_attr(self, args):
        for key, value in args.items():
            setattr(self, key, value)
