"""
MOF sbus.
"""
import networkx as nx
import numpy as np
from scipy.spatial import distance


def add_distance_matrix(graph):
    carts = []
    if(float('.'.join(nx.__version__.split('.')[:2])) >= 2.0):
        for j, data in sorted(list(graph.nodes(data=True))):
            carts.append(data['cartesian_coordinates'])
    else:
        for j, data in sorted(graph.nodes_iter(data=True)):
            carts.append(data['cartesian_coordinates'])
    carts = np.array(carts)
    graph.distance_matrix = distance.cdist(carts, carts)


InorganicCluster = {
        'Cu':{'Cu Paddlewheel': nx.Graph(name='Cu Paddlewheel') # taken from doi: 10.1126/science.283.5405.1148
              },
        'Zn':{'Zn4O': nx.Graph(name='Zn4O'), # taken from doi:
              'Zn Paddlewheel': nx.Graph(name='Zn Paddlewheel'), # taken from doi:
              'Kuratowski': nx.Graph(name='Kuratowski')},
        'Zr':{'Zr_UiO': nx.Graph(name='Zr_UiO') # taken from doi:
              },
        'Cr':{'Cr_tri': nx.Graph(name='Cr_tri') # taken from doi:
              },
        'V':{'V_pillar': nx.Graph(name='V_pillar') # taken from doi:
              },
        'Al':{'Al_pillar': nx.Graph(name='Al_pillar') # taken from doi:
              }
        }


OrganicCluster = {
        'N':{'Thymine': nx.Graph(name='Thymine'),
             'Adenine': nx.Graph(name='Adenine'),
             'CarboxylateImidazolate': nx.Graph(name='CarboxylateImidazolate')},
        'C':{'Benzene-2C': nx.Graph(name='Benzene-2C'),
             'Biphenyl-2C': nx.Graph(name='Biphenyl-2C'),
             'Triphenyl-2C': nx.Graph(name='Triphenyl-2C')
            }
        }

# add entry
InorganicCluster['Cu']['Cu Paddlewheel'].add_nodes_from([
    (1, {'element':'O',
         'special_flag': 'O1_Cu_pdw',
         'cartesian_coordinates':np.array([1.755, -0.181, -1.376])
         }
        ),
    (2, {'element':'O',
         'special_flag': 'O2_Cu_pdw',
         'cartesian_coordinates':np.array([-1.755,  0.181, -1.376])
         }
        ),
    (3, {'element':'O',
         'special_flag': 'O1_Cu_pdw',
         'cartesian_coordinates':np.array([-0.181,  1.755,  1.376])
         }
        ),
    (4, {'element':'O',
         'special_flag':'O2_Cu_pdw',
         'cartesian_coordinates':np.array([0.181, -1.755,  1.376])
         }
        ),
    (5, {'element':'O',
         'special_flag':'O1_Cu_pdw',
         'cartesian_coordinates':np.array([-1.755,  0.181,  1.376])
         }
        ),
    (6, {'element':'O',
         'special_flag':'O2_Cu_pdw',
         'cartesian_coordinates':np.array([1.755, -0.181,  1.376])
         }
        ),
    (7, {'element':'O',
         'special_flag':'O1_Cu_pdw',
         'cartesian_coordinates':np.array([0.181, -1.755, -1.376])
         }
        ),
    (8, {'element':'O',
         'special_flag':'O2_Cu_pdw',
         'cartesian_coordinates':np.array([-0.181,  1.755, -1.376])
         }
        ),
    (9, {'element':'Cu',
         'special_flag':'Cu_pdw',
         'cartesian_coordinates':np.array([0.929,  0.929,  0.000])
         }
        ),
    (10, {'element':'Cu',
          'special_flag':'Cu_pdw',
          'cartesian_coordinates':np.array([-0.929, -0.929,  0.000])
          }
        ),
    (11, {'element':'C',
          'special_flag':'C_Cu_pdw',
          'cartesian_coordinates':np.array([1.233, -1.233, -1.810])
          }
        ),
    (12, {'element':'C',
          'special_flag':'C_Cu_pdw',
          'cartesian_coordinates':np.array([-1.233, 1.233, -1.810])
          }
        ),
    (13, {'element':'C',
          'special_flag':'C_Cu_pdw',
          'cartesian_coordinates':np.array([-1.233, 1.233, 1.810])
          }
        ),
    (14, {'element':'C',
          'special_flag':'C_Cu_pdw',
          'cartesian_coordinates':np.array([1.233, -1.233, 1.810])
          }
        )
    ])

InorganicCluster['Zn']['Zn Paddlewheel'].add_nodes_from([
    (1, {'element':'O',
         'special_flag': 'O1_Zn_pdw',
         'cartesian_coordinates':np.array([-1.398, -1.339, 1.417])
         }
        ),
    (2, {'element':'O',
         'special_flag': 'O2_Zn_pdw',
         'cartesian_coordinates':np.array([-1.398, 0.853, -1.417])
         }
        ),
    (3, {'element':'O',
         'special_flag': 'O1_Zn_pdw',
         'cartesian_coordinates':np.array([-1.398, 0.853, 1.417])
         }
        ),
    (4, {'element':'O',
         'special_flag':'O2_Zn_pdw',
         'cartesian_coordinates':np.array([-1.398, -1.339, -1.417])
         }
        ),
    (5, {'element':'O',
         'special_flag':'O1_Zn_pdw',
         'cartesian_coordinates':np.array([1.398, -1.339, -1.417])
         }
        ),
    (6, {'element':'O',
         'special_flag':'O2_Zn_pdw',
         'cartesian_coordinates':np.array([1.398, 0.853, 1.417])
         }
        ),
    (7, {'element':'O',
         'special_flag':'O1_Zn_pdw',
         'cartesian_coordinates':np.array([1.398, 0.853, -1.417])
         }
        ),
    (8, {'element':'O',
         'special_flag':'O2_Zn_pdw',
         'cartesian_coordinates':np.array([1.398, -1.339, 1.417])
         }
        ),
    (9, {'element':'Zn',
         'special_flag':'Zn_pdw',
         'cartesian_coordinates':np.array([0.000, -1.717, 0.000])
         }
        ),
    (10, {'element':'Zn',
          'special_flag':'Zn_pdw',
          'cartesian_coordinates':np.array([0.000, 1.230, 0.000])
          }
        ),
    (11, {'element':'C',
          'special_flag':'C_Zn_pdw',
          'cartesian_coordinates':np.array([-1.761, -0.243, 1.837])
          }
        ),
    (12, {'element':'C',
          'special_flag':'C_Zn_pdw',
          'cartesian_coordinates':np.array([-1.761, -0.243, -1.837])
          }
        ),
    (13, {'element':'C',
          'special_flag':'C_Zn_pdw',
          'cartesian_coordinates':np.array([1.761, -0.243, 1.837])
          }
        ),
    (14, {'element':'C',
          'special_flag':'C_Zn_pdw',
          'cartesian_coordinates':np.array([1.761, -0.243, -1.837])
          }
        )
    ])


InorganicCluster['Zn']['Zn4O'].add_nodes_from([
    (1, {'element':'Zn',
         'special_flag':'Zn4O',
         'cartesian_coordinates':np.array([-1.063000,-1.063000,-1.174000])
         }
       ),
    (2, {'element':'Zn',
         'special_flag':'Zn4O',
         'cartesian_coordinates':np.array([-1.062000,1.179000,1.067000])
         }
       ),
    (3, {'element':'Zn',
         'special_flag':'Zn4O',
         'cartesian_coordinates':np.array([1.179000,-1.063000,1.067000])
         }
       ),
    (4, {'element':'Zn',
         'special_flag':'Zn4O',
         'cartesian_coordinates':np.array([1.179000,1.178000,-1.175000])
         }
       ),
    (5, {'element':'O',
         'special_flag':'O_z_Zn4O',
         'cartesian_coordinates':np.array([0.058000,0.058000,-0.054000])
         }
       ),
    (6, {'element':'O',
         'special_flag':'O_c_Zn4O',
         'cartesian_coordinates':np.array([-2.939000,-0.765000,-0.876000])
         }
       ),
    (7, {'element':'O',
         'special_flag':'O_c_Zn4O',
         'cartesian_coordinates':np.array([-0.764000,0.883000,2.943000])
         }
       ),
    (8, {'element':'O',
         'special_flag':'O_c_Zn4O',
         'cartesian_coordinates':np.array([0.881000,-2.938000,0.770000])
         }
       ),
    (9, {'element':'O',
         'special_flag':'O_c_Zn4O',
         'cartesian_coordinates':np.array([-2.938000,0.883000,0.770000])
         }
       ),
    (10, {'element':'O',
          'special_flag':'O_c_Zn4O',
          'cartesian_coordinates':np.array([-0.767000,-2.938000,-0.876000])
          }
        ),
    (11, {'element':'O',
          'special_flag':'O_c_Zn4O',
          'cartesian_coordinates':np.array([0.882000,-0.764000,2.943000])
          }
        ),
    (12, {'element':'O',
          'special_flag':'O_c_Zn4O',
          'cartesian_coordinates':np.array([3.055000,-0.766000,0.769000])
          }
        ),
    (13, {'element':'O',
          'special_flag':'O_c_Zn4O',
          'cartesian_coordinates':np.array([0.881000,0.880000,-3.051000])
          }
        ),
    (14, {'element':'O',
          'special_flag':'O_c_Zn4O',
          'cartesian_coordinates':np.array([3.055000,0.880000,-0.878000])
          }
        ),
    (15, {'element':'O',
          'special_flag':'O_c_Zn4O',
          'cartesian_coordinates':np.array([-0.766000,-0.766000,-3.050000])
          }
        ),
    (16, {'element':'O',
          'special_flag':'O_c_Zn4O',
          'cartesian_coordinates':np.array([-0.764000,3.055000,0.769000])
          }
        ),
    (17, {'element':'O',
          'special_flag':'O_c_Zn4O',
          'cartesian_coordinates':np.array([0.882000,3.054000,-0.879000])
          }
        ),
    (18, {'element':'C',
          'special_flag':'C_Zn4O',
          'cartesian_coordinates':np.array([3.541000,0.057000,-0.055000])
          }
        ),
    (19, {'element':'C',
          'special_flag':'C_Zn4O',
          'cartesian_coordinates':np.array([0.059000,3.541000,-0.055000])
          }
        ),
    (20, {'element':'C',
          'special_flag':'C_Zn4O',
          'cartesian_coordinates':np.array([0.057000,0.057000,-3.550000])
          }
        ),
    (21, {'element':'C',
          'special_flag':'C_Zn4O',
          'cartesian_coordinates':np.array([-3.438000,0.059000,-0.053000])
          }
        ),
    (22, {'element':'C',
          'special_flag':'C_Zn4O',
          'cartesian_coordinates':np.array([0.057000,-3.438000,-0.053000])
          }
        ),
    (23, {'element':'C',
          'special_flag':'C_Zn4O',
          'cartesian_coordinates':np.array([0.058000,0.058000,3.429000])
          }
        )
    ])

InorganicCluster['Zn']['Kuratowski'].add_nodes_from([
    (1, {'element':'Zn',
         'special_flag':'Zn_tet',
         'cartesian_coordinates':np.array([2.079000,2.079000,-2.079000])
         }
       ),
    (2, {'element':'Cl',
         'special_flag':'Cl_kuratowski',
         'cartesian_coordinates':np.array([3.295000,3.295000,-3.295000])
         }
       ),
    (3, {'element':'Zn',
         'special_flag':'Zn_tet',
         'cartesian_coordinates':np.array([-2.079000,2.079000,2.079000])
         }
       ),
    (4, {'element':'Cl',
         'special_flag':'Cl_kuratowski',
         'cartesian_coordinates':np.array([-3.295000,3.295000,3.295000])
         }
       ),
    (5, {'element':'Zn',
         'special_flag':'Zn_tet',
         'cartesian_coordinates':np.array([2.079000,-2.079000,2.079000])
         }
       ),
    (6, {'element':'Cl',
         'special_flag':'Cl_kuratowski',
         'cartesian_coordinates':np.array([3.295000,-3.295000,3.295000])
         }
       ),
    (7, {'element':'Zn',
         'special_flag':'Zn_tet',
         'cartesian_coordinates':np.array([-2.079000,-2.079000,-2.079000])
         }
       ),
    (8, {'element':'Cl',
         'special_flag':'Cl_kuratowski',
         'cartesian_coordinates':np.array([-3.295000,-3.295000,-3.295000])
         }
       ),
    (9, {'element':'Zn',
         'special_flag':'Zn_oct',
         'cartesian_coordinates':np.array([-0.000000,-0.000000,-0.000000])
         }
       ),
    (10, {'element':'N',
          'special_flag':'N_tet',
          'cartesian_coordinates':np.array([2.946000,0.770000,-0.770000])
          }
        ),
    (11, {'element':'C',
          'special_flag':'C_kuratowski',
          'cartesian_coordinates':np.array([4.261000,-0.493000,0.493000])
          }
        ),
    (12, {'element':'N',
          'special_flag':'N_tet',
          'cartesian_coordinates':np.array([-0.770000,2.946000,0.770000])
          }
        ),
    (13, {'element':'C',
          'special_flag':'C_kuratowski',
          'cartesian_coordinates':np.array([0.493000,4.261000,-0.493000])
          }
        ),
    (14, {'element':'N',
          'special_flag':'N_tet',
          'cartesian_coordinates':np.array([0.770000,-0.770000,2.946000])
          }
        ),
    (15, {'element':'C',
          'special_flag':'C_kuratowski',
          'cartesian_coordinates':np.array([-0.493000,0.493000,4.261000])
          }
        ),
    (16, {'element':'N',
          'special_flag':'N_tet',
          'cartesian_coordinates':np.array([0.770000,2.946000,-0.770000])
          }
        ),
    (17, {'element':'C',
          'special_flag':'C_kuratowski',
          'cartesian_coordinates':np.array([-0.493000,4.261000,0.493000])
          }
        ),
    (18, {'element':'N',
          'special_flag':'N_tet',
          'cartesian_coordinates':np.array([2.946000,-0.770000,0.770000])
          }
        ),
    (19, {'element':'C',
          'special_flag':'C_kuratowski',
          'cartesian_coordinates':np.array([4.261000,0.493000,-0.493000])
          }
        ),
    (20, {'element':'N',
          'special_flag':'N_tet',
          'cartesian_coordinates':np.array([-0.770000,0.770000,2.946000])
          }
        ),
    (21, {'element':'C',
          'special_flag':'C_kuratowski',
          'cartesian_coordinates':np.array([0.493000,-0.493000,4.261000])
          }
        ),
    (22, {'element':'N',
          'special_flag':'N_tet',
          'cartesian_coordinates':np.array([-0.770000,-2.946000,-0.770000])
          }
        ),
    (23, {'element':'C',
          'special_flag':'C_kuratowski',
          'cartesian_coordinates':np.array([0.493000,-4.261000,0.493000])
          }
        ),
    (24, {'element':'N',
          'special_flag':'N_tet',
          'cartesian_coordinates':np.array([0.770000,0.770000,-2.946000])
          }
        ),
    (25, {'element':'C',
          'special_flag':'C_kuratowski',
          'cartesian_coordinates':np.array([-0.493000,-0.493000,-4.261000])
          }
        ),
    (26, {'element':'N',
          'special_flag':'N_tet',
          'cartesian_coordinates':np.array([0.770000,-2.946000,0.770000])
          }
        ),
    (27, {'element':'C',
          'special_flag':'C_kuratowski',
          'cartesian_coordinates':np.array([-0.493000,-4.261000,-0.493000])
          }
        ),
    (28, {'element':'N',
          'special_flag':'N_tet',
          'cartesian_coordinates':np.array([-0.770000,-0.770000,-2.946000])
          }
        ),
    (29, {'element':'C',
          'special_flag':'C_kuratowski',
          'cartesian_coordinates':np.array([0.493000,0.493000,-4.261000])
          }
        ),
    (30, {'element':'N',
          'special_flag':'N_tet',
          'cartesian_coordinates':np.array([-2.946000,0.770000,0.770000])
          }
        ),
    (31, {'element':'C',
          'special_flag':'C_kuratowski',
          'cartesian_coordinates':np.array([-4.261000,-0.493000,-0.493000])
          }
        ),
    (32, {'element':'N',
          'special_flag':'N_tet',
          'cartesian_coordinates':np.array([-2.946000,-0.770000,-0.770000])
          }
        ),
    (33, {'element':'C',
          'special_flag':'C_kuratowski',
          'cartesian_coordinates':np.array([-4.261000,0.493000,0.493000])
          }
        ),
    (34, {'element':'N',
          'special_flag':'N_oct',
          'cartesian_coordinates':np.array([2.211000,-0.000000,-0.000000])
          }
        ),
    (35, {'element':'N',
          'special_flag':'N_oct',
          'cartesian_coordinates':np.array([-0.000000,2.211000,-0.000000])
          }
        ),
    (36, {'element':'N',
          'special_flag':'N_oct',
          'cartesian_coordinates':np.array([-0.000000,-0.000000,2.211000])
          }
        ),
    (37, {'element':'N',
          'special_flag':'N_oct',
          'cartesian_coordinates':np.array([-0.000000,-2.211000,-0.000000])
          }
        ),
    (38, {'element':'N',
          'special_flag':'N_oct',
          'cartesian_coordinates':np.array([-0.000000,-0.000000,-2.211000])
          }
        ),
    (39, {'element':'N',
          'special_flag':'N_oct',
          'cartesian_coordinates':np.array([-2.211000,-0.000000,-0.000000])
          }
        )
    ])


InorganicCluster['Zr']['Zr_UiO'].add_nodes_from([
    (1, {'element':'Zr',
         'special_flag':'Zr_UiO',
         'cartesian_coordinates':np.array([-0.000000,-2.521000,0.000000])
         }
       ),
    (2, {'element':'O',
         'special_flag':'O_c_Zr_UiO',
         'cartesian_coordinates':np.array([1.973000,-3.568000,0.000000])
         }
       ),
    (3, {'element':'O',
         'special_flag':'O_c_Zr_UiO',
         'cartesian_coordinates':np.array([-1.973000,-3.568000,0.000000])
         }
       ),
    (4, {'element':'O',
         'special_flag':'O_c_Zr_UiO',
         'cartesian_coordinates':np.array([-0.000000,-2.012000,-3.529000])
         }
       ),
    (5, {'element':'O',
         'special_flag':'O_c_Zr_UiO',
         'cartesian_coordinates':np.array([-0.000000,-2.012000,3.529000])
         }
       ),
    (6, {'element':'O',
         'special_flag':'O_c_Zr_UiO',
         'cartesian_coordinates':np.array([-0.000000,-3.568000,-1.973000])
         }
       ),
    (7, {'element':'O',
         'special_flag':'O_c_Zr_UiO',
         'cartesian_coordinates':np.array([-0.000000,-3.568000,1.973000])
         }
       ),
    (8, {'element':'O',
         'special_flag':'O_c_Zr_UiO',
         'cartesian_coordinates':np.array([-3.529000,-2.012000,0.000000])
         }
       ),
    (9, {'element':'O',
         'special_flag':'O_c_Zr_UiO',
         'cartesian_coordinates':np.array([3.529000,-2.012000,0.000000])
         }
       ),
    (10, {'element':'O',
          'special_flag':'O_h_Zr_UiO',
          'cartesian_coordinates':np.array([1.161000,-1.200000,-1.161000])
          }
        ),
    (11, {'element':'O',
          'special_flag':'O_h_Zr_UiO',
          'cartesian_coordinates':np.array([-1.161000,-1.200000,1.161000])
          }
        ),
    (12, {'element':'O',
          'special_flag':'O_z_Zr_UiO',
          'cartesian_coordinates':np.array([1.161000,-1.200000,1.161000])
          }
        ),
    (13, {'element':'O',
          'special_flag':'O_z_Zr_UiO',
          'cartesian_coordinates':np.array([-1.161000,-1.200000,-1.161000])
          }
        ),
    (14, {'element':'C',
          'special_flag':'C_Zr_UiO',
          'cartesian_coordinates':np.array([-3.180000,-3.219000,0.000000])
          }
        ),
    (15, {'element':'C',
          'special_flag':'C_Zr_UiO',
          'cartesian_coordinates':np.array([3.180000,-3.219000,0.000000])
          }
        ),
    (16, {'element':'C',
          'special_flag':'C_Zr_UiO',
          'cartesian_coordinates':np.array([-0.000000,-3.219000,3.180000])
          }
        ),
    (17, {'element':'C',
          'special_flag':'C_Zr_UiO',
          'cartesian_coordinates':np.array([-0.000000,-3.219000,-3.180000])
          }
        ),
    (18, {'element':'Zr',
          'special_flag':'Zr_UiO',
          'cartesian_coordinates':np.array([2.482000,-0.039000,0.000000])
          }
        ),
    (19, {'element':'Zr',
          'special_flag':'Zr_UiO',
          'cartesian_coordinates':np.array([-2.482000,-0.039000,0.000000])
          }
        ),
    (20, {'element':'Zr',
          'special_flag':'Zr_UiO',
          'cartesian_coordinates':np.array([-0.000000,2.443000,0.000000])
          }
        ),
    (21, {'element':'Zr',
          'special_flag':'Zr_UiO',
          'cartesian_coordinates':np.array([-0.000000,-0.039000,2.482000])
          }
        ),
    (22, {'element':'Zr',
          'special_flag':'Zr_UiO',
          'cartesian_coordinates':np.array([-0.000000,-0.039000,-2.482000])
          }
        ),
    (23, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([3.529000,-0.039000,1.973000])
          }
        ),
    (24, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([-3.529000,-0.039000,1.973000])
          }
        ),
    (25, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([-3.529000,-0.039000,-1.973000])
          }
        ),
    (26, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([3.529000,-0.039000,-1.973000])
          }
        ),
    (27, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([1.973000,3.490000,0.000000])
          }
        ),
    (28, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([-1.973000,3.490000,0.000000])
          }
        ),
    (29, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([-0.000000,1.934000,3.529000])
          }
        ),
    (30, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([-0.000000,1.934000,-3.529000])
          }
        ),
    (31, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([-0.000000,3.490000,-1.973000])
          }
        ),
    (32, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([-0.000000,3.490000,1.973000])
          }
        ),
    (33, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([3.529000,1.934000,0.000000])
          }
        ),
    (34, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([-3.529000,1.934000,0.000000])
          }
        ),
    (35, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([1.973000,-0.039000,-3.529000])
          }
        ),
    (36, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([1.973000,-0.039000,3.529000])
          }
        ),
    (37, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([-1.973000,-0.039000,3.529000])
          }
        ),
    (38, {'element':'O',
          'special_flag':'O_c_Zr_UiO',
          'cartesian_coordinates':np.array([-1.973000,-0.039000,-3.529000])
          }
        ),
    (39, {'element':'O',
          'special_flag':'O_h_Zr_UiO',
          'cartesian_coordinates':np.array([-1.161000,1.122000,-1.161000])
          }
        ),
    (40, {'element':'O',
          'special_flag':'O_h_Zr_UiO',
          'cartesian_coordinates':np.array([1.161000,1.122000,1.161000])
          }
        ),
    (41, {'element':'O',
          'special_flag':'O_z_Zr_UiO',
          'cartesian_coordinates':np.array([-1.161000,1.122000,1.161000])
          }
        ),
    (42, {'element':'O',
          'special_flag':'O_z_Zr_UiO',
          'cartesian_coordinates':np.array([1.161000,1.122000,-1.161000])
          }
        ),
    (43, {'element':'C',
          'special_flag':'C_Zr_UiO',
          'cartesian_coordinates':np.array([3.180000,-0.039000,-3.180000])
          }
        ),
    (44, {'element':'C',
          'special_flag':'C_Zr_UiO',
          'cartesian_coordinates':np.array([-3.180000,-0.039000,-3.180000])
          }
        ),
    (45, {'element':'C',
          'special_flag':'C_Zr_UiO',
          'cartesian_coordinates':np.array([-3.180000,-0.039000,3.180000])
          }
        ),
    (46, {'element':'C',
          'special_flag':'C_Zr_UiO',
          'cartesian_coordinates':np.array([3.180000,-0.039000,3.180000])
          }
        ),
    (47, {'element':'C',
          'special_flag':'C_Zr_UiO',
          'cartesian_coordinates':np.array([-3.180000,3.141000,0.000000])
          }
        ),
    (48, {'element':'C',
          'special_flag':'C_Zr_UiO',
          'cartesian_coordinates':np.array([3.180000,3.141000,0.000000])
          }
        ),
    (49, {'element':'C',
          'special_flag':'C_Zr_UiO',
          'cartesian_coordinates':np.array([-0.000000,3.141000,-3.180000])
          }
        ),
    (50, {'element':'C',
          'special_flag':'C_Zr_UiO',
          'cartesian_coordinates':np.array([-0.000000,3.141000,3.180000])
          }
        ),
    (51, {'element':'H',
          'special_flag':'H_o_Zr_UiO',
          'cartesian_coordinates':np.array([1.881000,1.801000,1.666000])
          }
        ),
    (52, {'element':'H',
          'special_flag':'H_o_Zr_UiO',
          'cartesian_coordinates':np.array([-1.832000,-1.884000,1.722000])
          }
        ),
    (53, {'element':'H',
          'special_flag':'H_o_Zr_UiO',
          'cartesian_coordinates':np.array([-1.838000,1.795000,-1.728000])
          }
        ),
    (54, {'element':'H',
          'special_flag':'H_o_Zr_UiO',
          'cartesian_coordinates':np.array([1.871000,-1.866000,-1.695000])
          }
        )
    ])


InorganicCluster['Cr']['Cr_tri'].add_nodes_from([
    (1, {'element':'O',
         'special_flag':'O',
         'cartesian_coordinates':np.array([2.267000,-1.345000,1.482000])
         }
       ),
    (2, {'element':'O',
         'special_flag':'O',
         'cartesian_coordinates':np.array([-0.321000,-2.272000,1.374000])
         }
       ),
    (3, {'element':'C',
         'special_flag':'C',
         'cartesian_coordinates':np.array([-1.353000,-2.006000,2.059000])
         }
       ),
    (4, {'element':'O',
         'special_flag':'O',
         'cartesian_coordinates':np.array([-2.299000,-1.290000,1.482000])
         }
       ),
    (5, {'element':'O',
         'special_flag':'O',
         'cartesian_coordinates':np.array([-1.808000,1.414000,1.374000])
         }
       ),
    (6, {'element':'C',
         'special_flag':'C',
         'cartesian_coordinates':np.array([-1.061000,2.175000,2.059000])
         }
       ),
    (7, {'element':'O',
         'special_flag':'O',
         'cartesian_coordinates':np.array([0.032000,2.636000,1.482000])
         }
       ),
    (8, {'element':'O',
         'special_flag':'O',
         'cartesian_coordinates':np.array([2.128000,0.859000,1.374000])
         }
       ),
    (9, {'element':'C',
         'special_flag':'C',
         'cartesian_coordinates':np.array([2.414000,-0.169000,2.059000])
         }
       ),
    (10, {'element':'O',
          'special_flag':'O',
          'cartesian_coordinates':np.array([2.267000,-1.345000,-1.477000])
          }
        ),
    (11, {'element':'O',
          'special_flag':'O',
          'cartesian_coordinates':np.array([-0.321000,-2.272000,-1.369000])
          }
        ),
    (12, {'element':'C',
          'special_flag':'C',
          'cartesian_coordinates':np.array([-1.353000,-2.006000,-2.054000])
          }
        ),
    (13, {'element':'O',
          'special_flag':'O',
          'cartesian_coordinates':np.array([-2.299000,-1.290000,-1.477000])
          }
        ),
    (14, {'element':'O',
          'special_flag':'O',
          'cartesian_coordinates':np.array([-1.808000,1.414000,-1.369000])
          }
        ),
    (15, {'element':'C',
          'special_flag':'C',
          'cartesian_coordinates':np.array([-1.061000,2.175000,-2.054000])
          }
        ),
    (16, {'element':'O',
          'special_flag':'O',
          'cartesian_coordinates':np.array([0.032000,2.636000,-1.477000])
          }
        ),
    (17, {'element':'O',
          'special_flag':'O',
          'cartesian_coordinates':np.array([2.128000,0.859000,-1.369000])
          }
        ),
    (18, {'element':'C',
          'special_flag':'C',
          'cartesian_coordinates':np.array([2.414000,-0.169000,-2.054000])
          }
        ),
    (19, {'element':'Cr',
          'special_flag':'Cr_tri',
          'cartesian_coordinates':np.array([0.918000,-1.740000,0.002000])
          }
        ),
    (20, {'element':'Cr',
          'special_flag':'Cr_tri',
          'cartesian_coordinates':np.array([-1.966000,0.075000,0.002000])
          }
        ),
    (21, {'element':'Cr',
          'special_flag':'Cr_tri',
          'cartesian_coordinates':np.array([1.048000,1.665000,0.002000])
          }
        ),
    (22, {'element':'O',
          'special_flag':'O_z_Cr_tri',
          'cartesian_coordinates':np.array([0.000000,0.000000,0.002000])
          }
        )
    ])

InorganicCluster['V']['V_pillar'].add_nodes_from([
    (1, {'element':'O',
         'special_flag':'O_c_V_pillar',
         'cartesian_coordinates':np.array([-3.335000,1.411000,1.192000])
         }
       ),
    (2, {'element':'O',
         'special_flag':'O_c_V_pillar',
         'cartesian_coordinates':np.array([-1.088000,-1.401000,1.345000])
         }
       ),
    (3, {'element':'O',
         'special_flag':'O_c_V_pillar',
         'cartesian_coordinates':np.array([0.073000,-1.411000,-1.136000])
         }
       ),
    (4, {'element':'C',
         'special_flag':'C_V_pillar',
         'cartesian_coordinates':np.array([-2.221000,-1.831000,1.655000])
         }
       ),
    (5, {'element':'O',
         'special_flag':'O_c_V_pillar',
         'cartesian_coordinates':np.array([-1.088000,1.401000,1.345000])
         }
       ),
    (6, {'element':'O',
         'special_flag':'O_c_V_pillar',
         'cartesian_coordinates':np.array([0.073000,1.411000,-1.136000])
         }
       ),
    (7, {'element':'C',
         'special_flag':'C_V_pillar',
         'cartesian_coordinates':np.array([-2.221000,1.831000,1.655000])
         }
       ),
    (8, {'element':'O',
         'special_flag':'O_c_V_pillar',
         'cartesian_coordinates':np.array([-3.335000,-1.411000,1.192000])
         }
       ),
    (9, {'element':'O',
         'special_flag':'O_z_V_pillar',
         'cartesian_coordinates':np.array([-2.201000,0.000000,-0.786000])
         }
       ),
    (10, {'element':'V',
          'special_flag':'V_pillar',
          'cartesian_coordinates':np.array([-0.327000,0.000000,0.179000])
          }
        ),
    (11, {'element':'O',
          'special_flag':'O_c_V_pillar',
          'cartesian_coordinates':np.array([2.321000,1.401000,-1.289000])
          }
        ),
    (12, {'element':'C',
          'special_flag':'C_V_pillar',
          'cartesian_coordinates':np.array([1.187000,1.831000,-1.599000])
          }
        ),
    (13, {'element':'O',
          'special_flag':'O_c_V_pillar',
          'cartesian_coordinates':np.array([2.321000,-1.401000,-1.289000])
          }
        ),
    (14, {'element':'C',
          'special_flag':'C_V_pillar',
          'cartesian_coordinates':np.array([1.187000,-1.831000,-1.599000])
          }
        ),
    (15, {'element':'V',
          'special_flag':'V_pillar',
          'cartesian_coordinates':np.array([3.082000,0.000000,-0.123000])
          }
        ),
    (16, {'element':'O',
          'special_flag':'O_z_V_pillar',
          'cartesian_coordinates':np.array([1.208000,0.000000,0.842000])
          }
        )
    ])


InorganicCluster['Al']['Al_pillar'].add_nodes_from([
    (1, {'element':'O',
         'special_flag':'O_c_Al_pillar',
         'cartesian_coordinates':np.array([-1.215000,1.107000,-0.732000])
         }
       ),
    (2, {'element':'O',
         'special_flag':'O_c_Al_pillar',
         'cartesian_coordinates':np.array([1.383000,-1.106000,-0.464000])
         }
       ),
    (3, {'element':'O',
         'special_flag':'O_c_Al_pillar',
         'cartesian_coordinates':np.array([1.383000,1.107000,-0.464000])
         }
       ),
    (4, {'element':'O',
         'special_flag':'O_c_Al_pillar',
         'cartesian_coordinates':np.array([-1.215000,-1.106000,-0.732000])
         }
       ),
    (5, {'element':'Al',
         'special_flag':'Al_pillar',
         'cartesian_coordinates':np.array([-0.102000,-1.657000,0.608000])
         }
       ),
    (6, {'element':'O',
         'special_flag':'O_z_Al_pillar',
         'cartesian_coordinates':np.array([-0.102000,0.000000,1.473000])
         }
       ),
    (7, {'element':'C',
         'special_flag':'C_Al_pillar',
         'cartesian_coordinates':np.array([2.005000,0.000000,-0.744000])
         }
       ),
    (8, {'element':'C',
         'special_flag':'C_Al_pillar',
         'cartesian_coordinates':np.array([-1.849000,0.000000,-0.976000])
         }
       ),
    (9, {'element':'H',
         'special_flag':'H_Al_pillar',
         'cartesian_coordinates':np.array([-0.121000,-0.071000,2.580000])
         }
       )#,
    #(10, {'element':'Al',
    #      'special_flag':'Al_pillar',
    #      'cartesian_coordinates':np.array([-0.102000,1.658000,0.608000])
    #      }
    #    )
    ])

OrganicCluster['N']['Adenine'].add_nodes_from([
    (1, {'element':'C',
         'special_flag':'C',
         'cartesian_coordinates':np.array([-0.108000,-0.237000,0.527000])
         }
       ),
    (2, {'element':'C',
         'special_flag':'C',
         'cartesian_coordinates':np.array([0.853000,-2.150000,0.700000])
         }
       ),
    (3, {'element':'C',
         'special_flag':'C',
         'cartesian_coordinates':np.array([0.550000,-0.540000,-0.675000])
         }
       ),
    (4, {'element':'C',
         'special_flag':'C',
         'cartesian_coordinates':np.array([-0.074000,1.419000,-1.600000])
         }
       ),
    (5, {'element':'C',
         'special_flag':'C',
         'cartesian_coordinates':np.array([-0.796000,0.992000,0.603000])
         }
       ),
    (6, {'element':'H',
         'special_flag':'Hd',
         'cartesian_coordinates':np.array([-1.914000,2.348000,1.629000])
         }
       ),
    (7, {'element':'H',
         'special_flag':'H',
         'cartesian_coordinates':np.array([-1.599000,0.804000,2.476000])
         }
       ),
    (8, {'element':'H',
         'special_flag':'H',
         'cartesian_coordinates':np.array([1.193000,-3.098000,1.104000])
         }
       ),
    (9, {'element':'H',
         'special_flag':'H',
         'cartesian_coordinates':np.array([-0.080000,2.127000,-2.431000])
         }
       ),
    (10, {'element':'N',
          'special_flag':'N',
          'cartesian_coordinates':np.array([0.121000,-1.283000,1.403000])
          }
        ),
    (11, {'element':'N',
          'special_flag':'N',
          'cartesian_coordinates':np.array([1.133000,-1.761000,-0.560000])
          }
        ),
    (12, {'element':'N',
          'special_flag':'N',
          'cartesian_coordinates':np.array([0.617000,0.283000,-1.751000])
          }
        ),
    (13, {'element':'N',
          'special_flag':'Na',
          'cartesian_coordinates':np.array([-0.763000,1.773000,-0.514000])
          }
        ),
    (14, {'element':'N',
          'special_flag':'Nd',
          'cartesian_coordinates':np.array([-1.424000,1.447000,1.691000])
          }
        )
    ])

OrganicCluster['N']['Thymine'].add_nodes_from([
    (1, {'element':'C',
         'special_flag':'C',
         'cartesian_coordinates':np.array([13.966000,16.972000,12.145000])
         }
       ),
    (2, {'element':'C',
         'special_flag':'C',
         'cartesian_coordinates':np.array([12.549000,18.380000,13.950000])
         }
       ),
    (3, {'element':'C',
         'special_flag':'C',
         'cartesian_coordinates':np.array([11.714000,19.119000,14.888000])
         }
       ),
    (4, {'element':'C',
         'special_flag':'C',
         'cartesian_coordinates':np.array([13.016000,17.103000,14.220000])
         }
       ),
    (5, {'element':'N',
         'special_flag':'Ndw',
         'cartesian_coordinates':np.array([13.714000,16.442000,13.316000])
         }
       ),
    (6, {'element':'O',
         'special_flag':'Oa2',
         'cartesian_coordinates':np.array([14.542000,16.323000,11.289000])
         }
       ),
    (7, {'element':'O',
         'special_flag':'Oaw',
         'cartesian_coordinates':np.array([12.755000,16.528000,15.269000])
         }
       ),
    (8, {'element':'H',
         'special_flag':'H',
         'cartesian_coordinates':np.array([10.864000,18.500000,15.184000])
         }
       ),
    (9, {'element':'H',
         'special_flag':'Hdw',
         'cartesian_coordinates':np.array([14.003000,15.581000,13.493000])
         }
       ),
    (10, {'element':'C',
          'special_flag':'C',
          'cartesian_coordinates':np.array([12.877000,18.890000,12.738000])
          }
        ),
    (11, {'element':'N',
          'special_flag':'Nd2',
          'cartesian_coordinates':np.array([13.557000,18.186000,11.867000])
          }
        ),
    (12, {'element':'H',
          'special_flag':'H',
          'cartesian_coordinates':np.array([12.293000,19.381000,15.776000])
          }
        ),
    (13, {'element':'H',
          'special_flag':'H',
          'cartesian_coordinates':np.array([11.316000,20.039000,14.453000])
          }
        ),
    (14, {'element':'H',
          'special_flag':'H',
          'cartesian_coordinates':np.array([12.585000,19.801000,12.470000])
          }
        ),
    (15, {'element':'H',
          'special_flag':'Hd2',
          'cartesian_coordinates':np.array([13.727000,18.544000,11.021000])
          }
        )
    ])
OrganicCluster['N']['CarboxylateImidazolate'].add_nodes_from([
    (1, {'element':'C',
         'special_flag':'C13',
         'cartesian_coordinates':np.array([-0.325000,-0.797000,0.755000])
         }
       ),
    (2, {'element':'N',
         'special_flag':'N20',
         'cartesian_coordinates':np.array([-0.712000,0.499000,0.760000])
         }
       ),
    (3, {'element':'C',
         'special_flag':'N20',
         'cartesian_coordinates':np.array([-0.133000,1.108000,-0.263000])
         }
       ),
    (4, {'element':'C',
         'special_flag':'C13',
         'cartesian_coordinates':np.array([0.616000,0.148000,-0.885000])
         }
       ),
    (5, {'element':'N',
         'special_flag':'N20',
         'cartesian_coordinates':np.array([0.512000,-1.071000,-0.265000])
         }
       ),
    (6, {'element':'H',
         'special_flag':'8H13',
         'cartesian_coordinates':np.array([1.218000,0.325000,-1.764000])
         }
       ),
    (7, {'element':'H',
         'special_flag':'H',
         'cartesian_coordinates':np.array([-0.314000,2.158000,-0.439000])
         }
       ),
    (8, {'element':'C',
         'special_flag':'C1',
         'cartesian_coordinates':np.array([-0.843000,-1.760000,1.840000])
         }
       ),
    (9, {'element':'O',
         'special_flag':'O2',
         'cartesian_coordinates':np.array([-0.453000,-3.062000,1.835000])
         }
       ),
    (10, {'element':'O',
          'special_flag':'O3',
          'cartesian_coordinates':np.array([-1.690000,-1.307000,2.803000])
          }
        )
    ])

# Note, the special_flags for the organic linkers below are designed to be compatible
# with the Dubbeldam force field, so changing these values will break if one requests
# the Dubbeldam FF.
OrganicCluster['C']['Benzene-2C'].add_nodes_from([
    (1, {'element':'C',
         'special_flag':'Cb',
         'cartesian_coordinates':np.array([-0.000000,-0.000000,-1.401000])
         }
       ),
    (2, {'element':'C',
         'special_flag':'Cb',
         'cartesian_coordinates':np.array([-0.000000,-0.000000,1.399000])
         }
       ),
    (3, {'element':'C',
         'special_flag':'Cc',
         'cartesian_coordinates':np.array([-0.858000,0.858000,-0.700000])
         }
       ),
    (4, {'element':'H',
         'special_flag':'Ha',
         'cartesian_coordinates':np.array([-1.519000,1.519000,-1.239000])
         }
       ),
    (5, {'element':'C',
         'special_flag':'Cc',
         'cartesian_coordinates':np.array([-0.857000,0.857000,0.700000])
         }
       ),
    (6, {'element':'H',
         'special_flag':'Ha',
         'cartesian_coordinates':np.array([-1.519000,1.519000,1.241000])
         }
       ),
    (7, {'element':'C',
         'special_flag':'Cc',
         'cartesian_coordinates':np.array([0.858000,-0.858000,-0.700000])
         }
       ),
    (8, {'element':'H',
         'special_flag':'Ha',
         'cartesian_coordinates':np.array([1.519000,-1.519000,-1.239000])
         }
       ),
    (9, {'element':'C',
         'special_flag':'Cc',
         'cartesian_coordinates':np.array([0.857000,-0.857000,0.700000])
         }
       ),
    (10, {'element':'H',
          'special_flag':'Ha',
          'cartesian_coordinates':np.array([1.519000,-1.519000,1.241000])
          }
        )
    ])

OrganicCluster['C']['Biphenyl-2C'].add_nodes_from([
    (1, {'element':'C',
         'special_flag':'Cb',
         'cartesian_coordinates':np.array([0.000000,0.000000,-3.571000])
         }
       ),
    (2, {'element':'C',
         'special_flag':'Ce',
         'cartesian_coordinates':np.array([0.000000,0.000000,-0.771000])
         }
       ),
    (3, {'element':'C',
         'special_flag':'Cb',
         'cartesian_coordinates':np.array([0.000000,0.000000,3.569000])
         }
       ),
    (4, {'element':'C',
         'special_flag':'Ce',
         'cartesian_coordinates':np.array([0.000000,0.000000,0.771000])
         }
       ),
    (5, {'element':'H',
         'special_flag':'Hb',
         'cartesian_coordinates':np.array([1.519000,-1.519000,0.928000])
         }
       ),
    (6, {'element':'C',
         'special_flag':'Cd',
         'cartesian_coordinates':np.array([0.858000,-0.858000,1.469000])
         }
       ),
    (7, {'element':'C',
         'special_flag':'Cc',
         'cartesian_coordinates':np.array([-0.858000,0.858000,-2.870000])
         }
       ),
    (8, {'element':'H',
         'special_flag':'Ha',
         'cartesian_coordinates':np.array([-1.519000,1.519000,-3.409000])
         }
       ),
    (9, {'element':'C',
         'special_flag':'Cd',
         'cartesian_coordinates':np.array([-0.857000,0.857000,-1.470000])
         }
       ),
    (10, {'element':'H',
          'special_flag':'Hb',
          'cartesian_coordinates':np.array([-1.519000,1.519000,-0.929000])
          }
        ),
    (11, {'element':'H',
          'special_flag':'Ha',
          'cartesian_coordinates':np.array([-1.519000,1.519000,3.412000])
          }
        ),
    (12, {'element':'C',
          'special_flag':'Cc',
          'cartesian_coordinates':np.array([-0.858000,0.858000,2.872000])
          }
        ),
    (13, {'element':'H',
          'special_flag':'Ha',
          'cartesian_coordinates':np.array([1.519000,-1.519000,3.412000])
          }
        ),
    (14, {'element':'C',
          'special_flag':'Cc',
          'cartesian_coordinates':np.array([0.858000,-0.858000,2.872000])
          }
        ),
    (15, {'element':'H',
          'special_flag':'Hb',
          'cartesian_coordinates':np.array([-1.519000,1.519000,0.928000])
          }
        ),
    (16, {'element':'C',
          'special_flag':'Cd',
          'cartesian_coordinates':np.array([-0.858000,0.858000,1.469000])
          }
        ),
    (17, {'element':'C',
          'special_flag':'Cc',
          'cartesian_coordinates':np.array([0.858000,-0.858000,-2.870000])
          }
        ),
    (18, {'element':'H',
          'special_flag':'Ha',
          'cartesian_coordinates':np.array([1.519000,-1.519000,-3.409000])
          }
        ),
    (19, {'element':'C',
          'special_flag':'Cd',
          'cartesian_coordinates':np.array([0.857000,-0.857000,-1.470000])
          }
        ),
    (20, {'element':'H',
          'special_flag':'Hb',
          'cartesian_coordinates':np.array([1.519000,-1.519000,-0.929000])
          }
        )
    ])

OrganicCluster['C']['Triphenyl-2C'].add_nodes_from([
    (1, {'element':'C',
         'special_flag':'Cb',
         'cartesian_coordinates':np.array([-0.000000,-0.000000,-5.741000])
         }
       ),
    (2, {'element':'C',
         'special_flag':'Ce',
         'cartesian_coordinates':np.array([-0.000000,-0.000000,-2.941000])
         }
       ),
    (3, {'element':'C',
         'special_flag':'Cf',
         'cartesian_coordinates':np.array([-0.000000,-0.000000,1.399000])
         }
       ),
    (4, {'element':'C',
         'special_flag':'Cb',
         'cartesian_coordinates':np.array([-0.000000,-0.000000,5.741000])
         }
       ),
    (5, {'element':'C',
         'special_flag':'Ce',
         'cartesian_coordinates':np.array([-0.000000,-0.000000,2.941000])
         }
       ),
    (6, {'element':'C',
         'special_flag':'Cf',
         'cartesian_coordinates':np.array([-0.000000,-0.000000,-1.399000])
         }
       ),
    (7, {'element':'H',
         'special_flag':'Hc',
         'cartesian_coordinates':np.array([1.519000,-1.519000,-1.242000])
         }
       ),
    (8, {'element':'C',
         'special_flag':'Cg',
         'cartesian_coordinates':np.array([0.858000,-0.858000,-0.701000])
         }
       ),
    (9, {'element':'C',
         'special_flag':'Cc',
         'cartesian_coordinates':np.array([-0.858000,0.858000,-5.040000])
         }
       ),
    (10, {'element':'H',
          'special_flag':'Ha',
          'cartesian_coordinates':np.array([-1.519000,1.519000,-5.579000])
          }
        ),
    (11, {'element':'C',
          'special_flag':'Cd',
          'cartesian_coordinates':np.array([-0.857000,0.857000,-3.640000])
          }
        ),
    (12, {'element':'H',
          'special_flag':'Hb',
          'cartesian_coordinates':np.array([-1.519000,1.519000,-3.099000])
          }
        ),
    (13, {'element':'H',
          'special_flag':'Hc',
          'cartesian_coordinates':np.array([-1.519000,1.519000,1.242000])
          }
        ),
    (14, {'element':'C',
          'special_flag':'Cg',
          'cartesian_coordinates':np.array([-0.858000,0.858000,0.701000])
          }
        ),
    (15, {'element':'C',
          'special_flag':'Cc',
          'cartesian_coordinates':np.array([0.858000,-0.858000,5.040000])
          }
        ),
    (16, {'element':'H',
          'special_flag':'Ha',
          'cartesian_coordinates':np.array([1.519000,-1.519000,5.579000])
          }
        ),
    (17, {'element':'C',
          'special_flag':'Cd',
          'cartesian_coordinates':np.array([0.857000,-0.857000,3.640000])
          }
        ),
    (18, {'element':'H',
          'special_flag':'Hb',
          'cartesian_coordinates':np.array([1.519000,-1.519000,3.099000])
          }
        ),
    (19, {'element':'H',
          'special_flag':'Hc',
          'cartesian_coordinates':np.array([1.519000,-1.519000,1.242000])
          }
        ),
    (20, {'element':'C',
          'special_flag':'Cg',
          'cartesian_coordinates':np.array([0.858000,-0.858000,0.701000])
          }
        ),
    (21, {'element':'C',
          'special_flag':'Cc',
          'cartesian_coordinates':np.array([-0.858000,0.858000,5.040000])
          }
        ),
    (22, {'element':'H',
          'special_flag':'Ha',
          'cartesian_coordinates':np.array([-1.519000,1.519000,5.579000])
          }
        ),
    (23, {'element':'C',
          'special_flag':'Cd',
          'cartesian_coordinates':np.array([-0.857000,0.857000,3.640000])
          }
        ),
    (24, {'element':'H',
          'special_flag':'Hb',
          'cartesian_coordinates':np.array([-1.519000,1.519000,3.099000])
          }
        ),
    (25, {'element':'H',
          'special_flag':'Hc',
          'cartesian_coordinates':np.array([-1.519000,1.519000,-1.242000])
          }
        ),
    (26, {'element':'C',
          'special_flag':'Cg',
          'cartesian_coordinates':np.array([-0.858000,0.858000,-0.701000])
          }
        ),
    (27, {'element':'C',
          'special_flag':'Cc',
          'cartesian_coordinates':np.array([0.858000,-0.858000,-5.040000])
          }
        ),
    (28, {'element':'H',
          'special_flag':'Ha',
          'cartesian_coordinates':np.array([1.519000,-1.519000,-5.579000])
          }
        ),
    (29, {'element':'C',
          'special_flag':'Cd',
          'cartesian_coordinates':np.array([0.857000,-0.857000,-3.640000])
          }
        ),
    (30, {'element':'H',
          'special_flag':'Hb',
          'cartesian_coordinates':np.array([1.519000,-1.519000,-3.099000])
          }
        )
    ])

# compute the distance matrix
add_distance_matrix(InorganicCluster['Cu']['Cu Paddlewheel'])
add_distance_matrix(InorganicCluster['Zn']['Zn Paddlewheel'])
add_distance_matrix(InorganicCluster['Zn']['Zn4O'])
add_distance_matrix(InorganicCluster['Zn']['Kuratowski'])
add_distance_matrix(InorganicCluster['Zr']['Zr_UiO'])
add_distance_matrix(InorganicCluster['Cr']['Cr_tri'])
add_distance_matrix(InorganicCluster['V']['V_pillar'])
add_distance_matrix(InorganicCluster['Al']['Al_pillar'])
add_distance_matrix(OrganicCluster['N']['Adenine'])
add_distance_matrix(OrganicCluster['N']['Thymine'])
add_distance_matrix(OrganicCluster['N']['CarboxylateImidazolate'])
add_distance_matrix(OrganicCluster['C']['Benzene-2C'])
add_distance_matrix(OrganicCluster['C']['Biphenyl-2C'])
add_distance_matrix(OrganicCluster['C']['Triphenyl-2C'])
