#!/usr/bin/env python
import sys
from lammps_interface.lammps_main import LammpsSimulation
from lammps_interface.structure_data import from_CIF, write_CIF, write_PDB, write_RASPA_CIF, write_RASPA_sim_files, MDMC_config
from lammps_interface.InputHandler import Options

# command line parsing
options = Options()
sim = LammpsSimulation(options)
cell, graph = from_CIF(options.cif_file)
sim.set_cell(cell)
sim.set_graph(graph)
sim.split_graph()
sim.assign_force_fields()
sim.compute_simulation_size()
sim.merge_graphs()
if options.output_cif:
    print("CIF file requested. Exiting...")
    write_CIF(graph, cell)
    sys.exit()
if options.output_pdb:
    print("PDB file requested. Exiting...")
    write_PDB(graph, cell)
    sys.exit()
sim.write_lammps_files()

# Additional capability to write RASPA files if requested
if options.output_raspa:
    print("Writing RASPA files to current WD")
    classifier = 1
    write_RASPA_CIF(graph, cell, classifier)
    write_RASPA_sim_files(sim, classifier)
    this_config = MDMC_config(sim)
    sim.set_MDMC_config(this_config)
