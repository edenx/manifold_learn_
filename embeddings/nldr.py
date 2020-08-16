"""
Use ARPACK or LOBCG in scipy and amg to solve the eigen problem
"""
from abc import ABCMeta, abstractmethod
from scipy import sparse
import warnings

class NLDR(object):
     """
     base class for NLDR methods
     """
     __metaclass__ = ABCMeta

     def __init__(
          self
          , adjacency_dic # {'obj': adjacency_obj, 'algorithm': 'auto'}
          , affinity_dic # {'obj': affinity_obj, 'algorithm': 'auto'}
          , n_components=2
          , eigen_solver='amg'
          , solver_kwds=None
          , seed=100
          ):
          self.adjacency = adjacency_dic['obj']
          self.affinity = affinity_dic['obj']

          if 'algorithm' in adjacency_dic:
               self.nn_algorithm = adjacency_dic['algorithm']
          else:
               self.nn_algorithm = 'auto'
          self.adjacency_mat = None

          if 'algorithm' in affinity_dic:
               self.aff_method = affinity_dic['algorithm']
          else:
               self.aff_method = 'auto'
          self.affinity_mat = None

          self.eigen_solver = eigen_solver
          self.solver_kwds = solver_kwds
          self.seed = seed

     
     def _get_Adjacency_matrix(self, X):
          """
          Build Adjacency matrix with KNN (discrete) or r-NN (continuous) 
          using scikit-learn. 
          """
          self.adjacency_mat = self.adjacency.compute_adjacency(X, algorithm=self.nn_algorithm)
          
     
     def get_Adjacency_matrix(self, X):
          if self.adjacency_mat is None:
               self._get_Adjacency_matrix(X)
          return self.adjacency_mat


     def _get_Affinity_matrix(self, X):
          """
          Build Affinity matrix with KNN (discrete) or r-NN (continuous) 
          using scikit-learn. 
          """
          if self.adjacency_mat is None:
               self._get_Adjacency_matrix(X)
          self.affinity_mat = self.affinity.compute_affinity(self.adjacency_mat, algorithm=self.aff_method)
     

     def get_Affinity_matrix(self, X):
          if self.affinity_mat is None:
               self._get_Affinity_matrix(X)
          return self.affinity_mat


     @abstractmethod
     def _get_Laplacian_matrix(self):
          """
          Build graph Laplacian, with {}
          """
          raise NotImplementedError
     

     @abstractmethod
     def get_Laplacian_matrix(self):
          raise NotImplementedError

     def fit(self):
          """
          Compute the embedding vectors for data X (and transform X with scipy.sparse.linalg.eigs)
          """
          raise NotImplementedError

     def transform(self):
          raise NotImplementedError
     
     def fit_transform(self):
          raise NotImplementedError

     


class LTSA(NLDR):


class LLE(NLDR):
     


     