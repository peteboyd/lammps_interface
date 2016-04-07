import networkx as nx
import numpy as np
from scipy.spatial import distance

def add_distance_matrix(graph):
    carts = []
    for j, data in sorted(graph.nodes_iter(data=True)):
        carts.append(data['cartesian_coordinates'])
    
    carts = np.array(carts)
    graph.distance_matrix = distance.cdist(carts, carts)

InorganicCluster = {
        'Cu':{'Cu Paddlewheel': nx.Graph(name='Cu Paddlewheel') # taken from doi: 10.1126/science.283.5405.1148
              },
        'Zn':{'Zn4O': nx.Graph(name='Zn4O'), # taken from doi:
              'Zn Paddlewheel': nx.Graph(name='Zn Paddlewheel') # taken from doi:
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

# compute the distance matrix
add_distance_matrix(InorganicCluster['Cu']['Cu Paddlewheel'])
add_distance_matrix(InorganicCluster['Zn']['Zn Paddlewheel'])
add_distance_matrix(InorganicCluster['Zn']['Zn4O'])
add_distance_matrix(InorganicCluster['Zr']['Zr_UiO'])
