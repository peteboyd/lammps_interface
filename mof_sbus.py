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
         'special_flag': 'O1',
         'cartesian_coordinates':np.array([1.755, -0.181, -1.376])
         }
        ),
    (2, {'element':'O',
         'special_flag': 'O2',
         'cartesian_coordinates':np.array([-1.755,  0.181, -1.376])
         }
        ),
    (3, {'element':'O',
         'special_flag': 'O1',
         'cartesian_coordinates':np.array([-0.181,  1.755,  1.376])
         }
        ),
    (4, {'element':'O', 
         'special_flag':'O2',
         'cartesian_coordinates':np.array([0.181, -1.755,  1.376])
         }
        ),
    (5, {'element':'O',
         'special_flag':'O1', 
         'cartesian_coordinates':np.array([-1.755,  0.181,  1.376])
         }
        ),
    (6, {'element':'O',
         'special_flag':'O2', 
         'cartesian_coordinates':np.array([1.755, -0.181,  1.376])
         }
        ),
    (7, {'element':'O',
         'special_flag':'O1',
         'cartesian_coordinates':np.array([0.181, -1.755, -1.376])
         }
        ),
    (8, {'element':'O',
         'special_flag':'O2',
         'cartesian_coordinates':np.array([-0.181,  1.755, -1.376])
         }
        ),
    (9, {'element':'Cu',
         'special_flag':'Cu2+', 
         'cartesian_coordinates':np.array([0.929,  0.929,  0.000])
         }
        ),
    (10, {'element':'Cu',
          'special_flag':'Cu2+',
          'cartesian_coordinates':np.array([-0.929, -0.929,  0.000])
          }
        ),
    (11, {'element':'C', 
          'special_flag':'sp2',
          'cartesian_coordinates':np.array([1.233, -1.233, -1.810])
          }
        ),
    (12, {'element':'C', 
          'special_flag':'sp2',
          'cartesian_coordinates':np.array([-1.233, 1.233, -1.810])
          }
        ),
    (13, {'element':'C', 
          'special_flag':'sp2',
          'cartesian_coordinates':np.array([-1.233, 1.233, 1.810])
          }
        ),
    (14, {'element':'C', 
          'special_flag':'sp2',
          'cartesian_coordinates':np.array([1.233, -1.233, 1.810])
          }
        )
    ])

InorganicCluster['Zn']['Zn Paddlewheel'].add_nodes_from([
    (1, {'element':'O',
         'special_flag': 'O1',
         'cartesian_coordinates':np.array([-1.398, -1.339, 1.417])
         }
        ),
    (2, {'element':'O',
         'special_flag': 'O2',
         'cartesian_coordinates':np.array([-1.398, 0.853, -1.417])
         }
        ),
    (3, {'element':'O',
         'special_flag': 'O1',
         'cartesian_coordinates':np.array([-1.398, 0.853, 1.417])
         }
        ),
    (4, {'element':'O', 
         'special_flag':'O2',
         'cartesian_coordinates':np.array([-1.398, -1.339, -1.417])
         }
        ),
    (5, {'element':'O',
         'special_flag':'O1', 
         'cartesian_coordinates':np.array([1.398, -1.339, -1.417])
         }
        ),
    (6, {'element':'O',
         'special_flag':'O2', 
         'cartesian_coordinates':np.array([1.398, 0.853, 1.417])
         }
        ),
    (7, {'element':'O',
         'special_flag':'O1',
         'cartesian_coordinates':np.array([1.398, 0.853, -1.417])
         }
        ),
    (8, {'element':'O',
         'special_flag':'O2',
         'cartesian_coordinates':np.array([1.398, -1.339, 1.417])
         }
        ),
    (9, {'element':'Zn',
         'special_flag':'Zn2+', 
         'cartesian_coordinates':np.array([0.000, -1.717, 0.000])
         }
        ),
    (10, {'element':'Zn',
          'special_flag':'Zn2+',
          'cartesian_coordinates':np.array([0.000, 1.230, 0.000])
          }
        ),
    (11, {'element':'C', 
          'special_flag':'sp2',
          'cartesian_coordinates':np.array([-1.761, -0.243, 1.837])
          }
        ),
    (12, {'element':'C', 
          'special_flag':'sp2',
          'cartesian_coordinates':np.array([-1.761, -0.243, -1.837])
          }
        ),
    (13, {'element':'C', 
          'special_flag':'sp2',
          'cartesian_coordinates':np.array([1.761, -0.243, 1.837])
          }
        ),
    (14, {'element':'C', 
          'special_flag':'sp2',
          'cartesian_coordinates':np.array([1.761, -0.243, -1.837])
          }
        )
    ])
# compute the distance matrix
add_distance_matrix(InorganicCluster['Cu']['Cu Paddlewheel'])
add_distance_matrix(InorganicCluster['Zn']['Zn Paddlewheel'])
