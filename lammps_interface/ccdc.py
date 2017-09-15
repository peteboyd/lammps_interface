"""
Bond order information.
"""
CCDC_BOND_ORDERS = {
    # http://cima.chem.usyd.edu.au:8080/cif/skunkworks/html/ddl1/mif/bond.html
    'S': 1.0,  # single (two-electron) bond or sigma bond to metal
    'D': 2.0,  # double (four-electron) bond
    'T': 3.0,  # triple (six-electron) bond
    'Q': 4.0,  # quadruple (eight-electron, metal-metal) bond
    'A': 1.5,  # alternating normalized ring bond (aromatic)
    'C': 1.0,  # catena-forming bond in crystal structure
    'E': 1.5,  # equivalent (delocalized double) bond
    'P': 1.0,   # pi bond (metal-ligand pi interaction)
    'Am': 1.41,  # Amide bond (non standard)
    1.0: 'S',  # single (two-electron) bond or sigma bond to metal
    2.0: 'D',  # double (four-electron) bond
    3.0: 'T',  # triple (six-electron) bond
    4.0: 'Q',  # quadruple (eight-electron, metal-metal) bond
    1.5: 'A',  # alternating normalized ring bond (aromatic)
    1.41: 'Am'  # Amide bond (non standard)
}
