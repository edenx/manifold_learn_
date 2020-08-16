import warnings
from __future__ import division
from nldr import NLDR
from scipy import sparse
from scipy.sparse.csgraph import shortest_path as graph_shortest_path
from .utils import eigen_decomposition

class Isomap(NLDR):
     def __init__(
          self
          , path_method='auto'
          , adjacency_dic # {'obj': adjacency_obj, 'algorithm': 'auto'}
          , affinity_dic # {'obj': affinity_obj, 'algorithm': 'auto'}
          , n_components=2
          , eigen_solver='amg'
          , solver_kwds=None
          , seed=100
          ):
          super().__init__(
               , adjacency_dic 
               , affinity_dic 
               , n_components=2
               , eigen_solver='amg'
               , solver_kwds=None
               , seed=100
               )
          self.path_method = path_method

     def _isomap(self, X):
          # get adjacency (distance) matrix first, then find shortest path 
          graph_distance_mat = graph_shortest_path(
               self.get_Adjacency_matrix(X), 
               method=self.path_method,
               directed=False)

          # center matrix
          centered_mat = centeralise_matrix(graph_distance_mat)

          # eigen decomp
          # get non-one vectors?
          eigenvalues, eigenvectors = eigen_decomposition(
               centered_mat, self.n_components, eigen_solver=eigen_solver,
               seed=self.seed, largest=True,
               solver_kwds=self.solver_kwds)
          
          # return Y = [sqrt(lambda_1)*V_1, ..., sqrt(lambda_d)*V_d]
          # ind = np.argsort(lambdas); ind = ind[::-1] # sort largest
          # lambdas = lambdas[ind];

          """ERROR: the eigenvalues should already been sorted using the spectral_embedding!!!"""

          embedding = eigenvectors[:, 0:n_components] * np.sqrt(eigenvalues[0:n_components])

          return embedding

     @staticmethod
     def _centeralise_matrix(G):
          """
          For S = -1/2* G^2 and  N_1 = np.ones([N, N])/N
          compute S - N_1*S - S*N_1  + N_1*S*N_1
          """
          S = -0.5 * G ** 2

          K = S.copy()
          K -= np.mean(S, axis=0)
          K -= np.mean(S, axis=1)[:, np.newaxis]
          K += np.mean(S)

          return K

     def fit_transform(self, X):
          self.embeddings = self._isomap(X)

     return self.embeddings