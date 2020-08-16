"""
find affinity matrix of data matrix X with methods {'auto', 'gaussian'}
credit to megaman/geometry/affinity.py
"""
from __future__ import division
import numpy as np
from scipy.sparse import isspmatrix
from sklearn.utils.validation import check_array

class Affinity():
     def __init__(self, radius=None, symmetrize=True):
          if radius is None:
               raise ValueError("must specify radius for affinity matrix")
          self.radius = radius
          self.symmetrize = symmetrize


     @staticmethod
     def _symmetrize(A):
          B = 0.5 * (A + A.T)
          # for sparse, manual symmetrization 
          if isspmatrix(B):
               B.setdiag(1)
          return B

     def compute_affinity(self, adjacency_matrix, method='auto'):
          if method is 'auto':
               method = 'gaussian'
          
          adj_M = check_array(adjacency_matrix, dtype=float, copy=True,
                         accept_sparse=['csr', 'csc', 'coo'])

          if isspmatrix(adj_M):
               data = adj_M.data
          else:
               data = adj_M

          # in-place computation of
          # data = np.exp(-(data / radius) ** 2)
          data **= 2
          data /= -self.radius ** 2
          np.exp(data, out=data)

          if self.symmetrize:
               adj_M = self._symmetrize(adj_M)

          return adj_M