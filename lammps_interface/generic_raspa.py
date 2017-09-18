#! /usr/bin/env python
"""
RASPA file format and default parameters.
"""
GENERIC_PSEUDO_ATOMS_HEADER = [
    ['# of pseudo atoms'],
    ['29'],
    ['#type ', 'print ', 'as ', 'chem ', 'oxidation ', 'mass ', 'charge ', 'polarization ', 'B-factor radii ', 'connectivity ', 'anisotropic ', 'anisotropic-type ', 'tinker-type ']
]

GENERIC_PSEUDO_ATOMS = [
    ['He     '    ,'yes'     ,'He'    ,'He'    ,'0'           ,'4.002602'   ,' 0.0   '   ,'0.0'          ,'1.0'      ,'1.0  '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['CH4_sp3'    ,'yes'     ,'C '    ,'C '    ,'0'           ,'16.04246'   ,' 0.0   '   ,'0.0'          ,'1.0'      ,'1.00 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['CH3_sp3'    ,'yes'     ,'C '    ,'C '    ,'0'           ,'15.03452'   ,' 0.0   '   ,'0.0'          ,'1.0'      ,'1.00 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['CH2_sp3'    ,'yes'     ,'C '    ,'C '    ,'0'           ,'14.02658'   ,' 0.0   '   ,'0.0'          ,'1.0'      ,'1.00 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['CH_sp3 '    ,'yes'     ,'C '    ,'C '    ,'0'           ,'13.01864'   ,' 0.0   '   ,'0.0'          ,'1.0'      ,'1.00 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['C_sp3  '    ,'yes'     ,'C '    ,'C '    ,'0'           ,'12.0    '   ,' 0.0   '   ,'0.0'          ,'1.0'      ,'1.00 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['H_h2   '    ,'yes'     ,'H '    ,'H '    ,'0'           ,'1.00794 '   ,' 0.468 '   ,'0.0'          ,'1.0'      ,'0.7  '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['H_com  '    ,'no '     ,'H '    ,'H '    ,'0'           ,'0.0     '   ,'-0.936 '   ,'0.0'          ,'1.0'      ,'0.7  '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['C_co2  '    ,'yes'     ,'C '    ,'C '    ,'0'           ,'12.0    '   ,' 0.70  '   ,'0.0'          ,'1.0'      ,'0.720'  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['O_co2  '    ,'yes'     ,'O '    ,'O '    ,'0'           ,'15.9994 '   ,'-0.35  '   ,'0.0'          ,'1.0'      ,'0.68 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['O_o2   '    ,'yes'     ,'O '    ,'O '    ,'0'           ,'15.9994 '   ,'-0.112 '   ,'0.0'          ,'1.0'      ,'0.7  '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['O_com  '    ,'no '     ,'O '    ,'- '    ,'0'           ,'0.0     '   ,' 0.224 '   ,'0.0'          ,'1.0'      ,'0.7  '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['N_n2   '    ,'yes'     ,'N '    ,'N '    ,'0'           ,'14.00674'   ,'-0.4048'   ,'0.0'          ,'1.0'      ,'0.7  '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['N_com  '    ,'no '     ,'N '    ,'- '    ,'0'           ,'0.0     '   ,' 0.8096'   ,'0.0'          ,'1.0'      ,'0.7  '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['Ar     '    ,'yes'     ,'Ar'    ,'Ar'    ,'0'           ,'39.948  '   ,' 0.0   '   ,'0.0'          ,'1.0'      ,'0.7  '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['Ow     '    ,'yes'     ,'O '    ,'O '    ,'0'           ,'15.9994 '   ,' 0.0   '   ,'0.0'          ,'1.0'      ,'0.5  '  ,'2'            ,'0'           ,'relative'           ,'0'],
    ['Hw     '    ,'yes'     ,'H '    ,'H '    ,'0'           ,'1.00794 '   ,' 0.241 '   ,'0.0'          ,'1.0'      ,'1.00 '  ,'1'            ,'0'           ,'relative'           ,'0'],
    ['Lw     '    ,'no '     ,'L '    ,'H '    ,'0'           ,'0.0     '   ,'-0.241 '   ,'0.0'          ,'1.0'      ,'1.00 '  ,'1'            ,'0'           ,'relative'           ,'0'],
    ['C_benz '    ,'yes'     ,'C '    ,'C '    ,'0'           ,'12.0    '   ,'-0.095 '   ,'0.0'          ,'1.0'      ,'0.70 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['H_benz '    ,'yes'     ,'H '    ,'H '    ,'0'           ,'1.00794 '   ,' 0.095 '   ,'0.0'          ,'1.0'      ,'0.320'  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['N_dmf  '    ,'yes'     ,'N '    ,'N '    ,'0'           ,'14.00674'   ,'-0.57  '   ,'0.0'          ,'1.0'      ,'0.50 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['Co_dmf '    ,'yes'     ,'C '    ,'C '    ,'0'           ,'12.0    '   ,' 0.45  '   ,'0.0'          ,'1.0'      ,'0.52 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['Cm_dmf '    ,'yes'     ,'C '    ,'C '    ,'0'           ,'12.0    '   ,' 0.28  '   ,'0.0'          ,'1.0'      ,'0.52 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['O_dmf  '    ,'yes'     ,'O '    ,'O '    ,'0'           ,'15.9994 '   ,'-0.50  '   ,'0.0'          ,'1.0'      ,'0.78 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['H_dmf  '    ,'yes'     ,'H '    ,'H '    ,'0'           ,'1.00794 '   ,' 0.06  '   ,'0.0'          ,'1.0'      ,'0.22 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['Na     '    ,'yes'     ,'Na'    ,'Na'    ,'0'           ,'22.98977'   ,' 1.0   '   ,'0.0'          ,'1.0'      ,'1.00 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['Cl     '    ,'yes'     ,'Cl'    ,'Cl'    ,'0'           ,'35.453  '   ,'-1.0   '   ,'0.0'          ,'1.0'      ,'1.00 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['Kr     '    ,'yes'     ,'Kr'    ,'Kr'    ,'0'           ,'83.798  '   ,' 0.0   '   ,'0.0'          ,'1.0'      ,'1.00 '  ,'0'            ,'0'           ,'relative'           ,'0'],
    ['Xe     '    ,'yes'     ,'Xe'    ,'Xe'    ,'0'           ,'131.293 '   ,' 0.0   '   ,'0.0'          ,'1.0'      ,'1.00 '  ,'0'            ,'0'           ,'relative'           ,'0'],
]

GENERIC_FF_MIXING_HEADER = [
    ['# general rule for shifted vs truncated                                                                                    '],
    ['shifted                                                                                                                    '],
    ['# general rule tailcorrections                                                                                             '],
    ['no                                                                                                                         '],
    ['# number of defined interactions                                                                                           '],
    ['55                                                                                                                         '],
    ['# type interaction, parameters.    IMPORTANT: define shortest matches first, so that more specific ones overwrites these   '],
]

GENERIC_FF_MIXING = [
    ['He      '      , 'lennard-jones'    ,'10.9  '    ,'2.64  '],
    ['CH4_sp3 '      , 'lennard-jones'    ,'158.5 '    ,'3.72  '],
    ['CH3_sp3 '      , 'lennard-jones'    ,'108.0 '    ,'3.76  '],
    ['CH2_sp3 '      , 'lennard-jones'    ,'56.0  '    ,'3.96  '],
    ['CH_sp3  '      , 'lennard-jones'    ,'17.0  '    ,'4.67  '],
    ['C_sp3   '      , 'lennard-jones'    ,' 0.8  '    ,'6.38  '],
    ['H_com   '      , 'lennard-jones'    ,'36.7  '    ,'2.958 '],
    ['H_h2    '      , 'none         '    ,'      '    ,'      '],
    ['O_co2   '      , 'lennard-jones'    ,'79.0  '    ,'3.05  '],
    ['C_co2   '      , 'lennard-jones'    ,'27.0  '    ,'2.80  '],
    ['C_benz  '      , 'lennard-jones'    ,'30.70 '    ,'3.60  '],
    ['H_benz  '      , 'lennard-jones'    ,'25.45 '    ,'2.36  '],
    ['N_n2    '      , 'lennard-jones'    ,'36.0  '    ,'3.31  '],
    ['N_com   '      , 'none         '    ,'      '    ,'      '],
    ['Ow      '      , 'lennard-jones'    ,'89.633'    ,'3.097 '],
    ['N_dmf   '      , 'lennard-jones'    ,'80.0  '    ,'3.2   '],
    ['Co_dmf  '      , 'lennard-jones'    ,'50.0  '    ,'3.7   '],
    ['Cm_dmf  '      , 'lennard-jones'    ,'80.0  '    ,'3.8   '],
    ['O_dmf   '      , 'lennard-jones'    ,'100.0 '    ,'2.96  '],
    ['H_dmf   '      , 'lennard-jones'    ,'8.0   '    ,'2.2   '],
    ['Ar      '      , 'lennard-jones'    ,'119.8 '    ,'3.34  '],
    ['Kr      '      , 'lennard-jones'    ,'166.4 '    ,'3.636 '],
    ['Xe      '      , 'lennard-jones'    ,'221.0 '    ,'4.1   '],
]

GENERIC_FF_MIXING_FOOTER = [
    ['# general mixing rule for Lennard-Jones '],
    ['Lorentz-Berthelot                       '],
]
