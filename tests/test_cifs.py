import glob
import os
import sys
import pytest
import subprocess

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
CIFS_DIR = os.path.join( os.path.join(TEST_DIR, 'cifs') )
CIF_FILES = glob.glob(os.path.join(CIFS_DIR, '*.cif'))

def run_lammps_interface(args):
    return subprocess.check_output(['lammps-interface'] + args, stderr=subprocess.STDOUT).decode('utf8')

@pytest.mark.parametrize("cif_file", CIF_FILES)
def test_parse(cif_file):
    """Test running lammps_interface on each CIF file in /cifs."""
    result = run_lammps_interface([cif_file])

    #assert not "Error" in result
