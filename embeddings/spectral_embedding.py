"""
Compute spectral embedding with graph Laplacian, where Eigenmap and Diffusion map
are within the category
USe 'amg' and 'lobpcg' 
"""
import warnings
from nldr import NLDR
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from utils import eigen_decomposition

class Spectral_embedding(NLDR):

     def __init__(
          self
          , laplacian_dic #{'obj': laplacian_obj, 'algorithm':'symmetricnormalized'}
          , diffusion=False
          , time=0 # diffusion time
          , adjacency_dic # {'obj': adjacency_obj, 'algorithm': 'auto'}
          , affinity_dic # {'obj': affinity_obj, 'algorithm': 'auto'}
          , n_components=2
          , eigen_solver='amg'
          , solver_kwds=None
          , seed=100
          ):
          super().__init__(
               adjacency_dic 
               , affinity_dic 
               , n_components=2
               , eigen_solver='amg'
               , solver_kwds=None
               , seed=100
               )
          
          self.laplacian = laplacian_dic['obj']

          self.lap_method = laplacian_dic['algorithm']
          self.laplacian_mat = None 
          self.laplacian_sym_mat = None
          self.laplacian_weights = None

          self.lap_method = lap_method
          self.diffusion = diffusion
          self.time = time


     def _get_Laplacian_matrix(self, X):
          """
          Build graph Laplacian, with {unnormalised, symmetricnormalized, renormalised}
          """
          self.laplacian_mat, self.laplacian_sym_mat, self.laplacian_weights = self.laplacian.compute_laplacian(
                    self.get_Affinity_matrix(X)
               )
          
     def get_Laplacian_matrix(self, X):
          if self.laplacian_mat is None:
               self._get_Laplacian_matrix(X)
          return self.laplacian_mat, self.laplacian_sym_mat, self.laplacian_weights


     @staticmethod
     def _graph_connected_component(G, node_id):
          """
          Find the largest graph connected components the contains one
          given node

          Parameters
          ----------
          G : array-like, shape: (n_samples, n_samples)
               adjacency matrix of the graph G, non-zero weight means an edge
               between the nodes

          node_id : int
               The index of the query node of the graph

          Returns
          -------
          connected_components : array-like, shape: (n_samples,)
               An array of bool value indicates the indexes of the nodes
               belong to the largest connected components of the given query
               node
          """
          connected_components = np.zeros(shape=(G.shape[0]), dtype=np.bool)
          connected_components[node_id] = True
          n_node = G.shape[0]
          for i in range(n_node):
               last_num_component = connected_components.sum()
               _, node_to_add = np.where(G[connected_components] != 0)
               connected_components[node_to_add] = True
               if last_num_component >= connected_components.sum():
                    break
          return connected_components

     @staticmethod
     def _check_connected(G):
          if sparse.isspmatrix(G):
               # sparse graph, find all the connected components
               n_connected_components, _ = connected_components(G)
               return n_connected_components == 1
          else:
               # dense graph, find all connected components start from node 0
               return _graph_connected_component(G, 0).sum() == G.shape[0]


     def _spectral_embedding(self, X):
          if not _graph_is_connected(geom.affinity_matrix):
               warnings.warn("Graph is not fully connected: "
                         "spectral embedding may not work as expected.")
          _, _, w = self.get_Laplacian_matrix(X)
          sym_lap = self.laplacian_sym_mat.copy()
          n_nodes = lap.shape[0]

          # use {'amg', 'lobpcg'} for the eigen-solver
          # using a symmetric laplacian but adjust to avoid positive definite errors
          epsilon = 2
          if sparse.isspmatrix(sym_lap):
               sym_lap = (1+epsilon) * sparse.identity(n_nodes) - sym_lap
          else:
               symmetrixed_laplacian = (1+epsilon) * np.identity(n_nodes) - sym_lap

          eigenvalues, eigenvectors = eigen_decomposition(
               symmetrized_laplacian, self.n_components+1, eigen_solver=eigen_solver,
               seed=self.seed, largest=False,
               solver_kwds=self.solver_kwds)
          eigenvalues = -eigenvalues + epsilon

          # get back to the space of the original non-sym Laplacian
          if self.lap_method is 'renormalised':
               eigenvectors /= np.sqrt(w[:, np.newaxis]) 
               eigenvectors /= np.linalg.norm(eigenvectors, axis = 0) # norm 1 vectors

          # sort the eigenvalues 
          ind = np.argsort(eigenvalues); ind = ind[::-1]
          eigenvalues = eigenvalues[ind]; eigenvalues[0] = 0
          eigenvectors = eigenvectors[:, ind]

          # if diffusion map
          if self.diffusion:
               embeddings = self._diffusion_map(eigenvecs, eigenvalues)[:, 1:(n_components+1)]
          else:
               embeddings = eigenvectors.copy()[:, 1:(n_components+1)]

          # TODO: check null space and remove zero evals and evecs
          return embedding, eigenvalues[:, 1:(n_components+1)], eigenvectors[:, 1:(n_components+1)]

     def _diffusion_map(self, eigenvectors, eigenvalues):
          if self.lap_method is not 'renormalized':
                    warnings.warn("for correct diffusion maps embedding use laplacian type 'renormalized'.")
          # Step 5 of diffusion maps:
          vectors = eigenvectors.copy()
          values = eigenvalues.copy()
          psi = vectors/vectors[:,[0]]

          if self.time == 0:
               values = np.abs(values)
               self.time = np.exp(1. -  np.log(1 - values[1:])/np.log(values[1:]))
               values = values / (1 - values)
          else:
               values = np.abs(values)
               values = values ** float(self.time)
          diffusion_map = psi * values

          return diffusion_map

     def fit_transform(self, X):
          self.embeddings, self.eigenvalues, self.eigenvectors = self._spectral_embedding(X)

          return self.embeddings