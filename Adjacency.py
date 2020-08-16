"""
Wrapper of scikit-learn NN methods
"""
import numpy as np
from sklearn import neighbors
from scipy import sparse

class Ajacency():
     def __init__(
          self, 
          n_neighbors=None, 
          radius=None,
          mode='distance'
          ):

          self.n_neighbors = n_neighbors
          self.mode = mode
          if n_neighbors is not None:
               self.method = 'kneighbors'
               self.n_neighbors = n_neighbors
               self.radius = None
          elif radius is not None:
               self.method = 'radius_neighbors'
               self.radius = radius
               self.n_neighbors = None
          else:
               raise ValueError("Must have either 'n_neighbors' or 'radius'.")
     
     @staticmethod
     def _check_algo(X):
     """
     Determine the algorithm of NN from the dimension of data matrix, 
     if input is 'auto'.
     X: data matrix, nxd np array or sparse matrix. 
     """
     if X.shape[1]>6:
          method = 'ball_tree'
     else: 
          method = 'kd_tree'

     return method

     def compute_adjacency(
          self, 
          X, 
          algorithm='auto'
          # leaf_size=30, 
          # metric='minkowski', 
          # p=2, 
          # metric_params=None, 
          # n_jobs=None
          ):
          if algorithm is 'auto':
               algorithm = _check_algo(X)
          
          nn = neighbors.NearestNeighbors(algorithm=algorithm).fit(X)

          if self.radius is not None:
               return nn.radius_neighbors_graph(X, radius=self.radius,
                                            mode=self.mode)
          else:
               return nn.kneighbors_graph(X, n_neighbors=self.n_neighbors,
                                      mode=self.mode)
