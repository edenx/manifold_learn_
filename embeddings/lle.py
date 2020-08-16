import warnings
from nldr import NLDR
from scipy import sparse
from sparse import eye, csr_matrix
from scipy.linalg import solve
from .utils import eigen_decomposition, null_space

class LLE(NLDR):
     def __init__(
          self
          , reg=1e-3
          , adjacency_dic # {'obj': adjacency_obj, 'algorithm': 'auto'}
          , affinity_dic # {'obj': affinity_obj, 'algorithm': 'auto'}
          , n_components=2
          , eigen_solver='amg'
          , solver_kwds=None
          , seed=100)

          super().__init__(
               adjacency_dic 
               , affinity_dic 
               , n_components=2
               , eigen_solver='amg'
               , solver_kwds=None
               , seed=100
               )
          self.reg = reg


     # based on megaman
     def barycenter_graph(self, X):
     """
     Computes the barycenter weighted graph for points in X

     Parameters
     ----------
     distance_matrix: sparse Ndarray, (N_obs, N_obs) pairwise distance matrix.
     X : Ndarray (N_obs, N_dim) observed data matrix.
     reg : float, optional
          Amount of regularization when solving the least-squares
          problem. Only relevant if mode='barycenter'. If None, use the
          default.

     Returns
     -------
     W : sparse matrix in CSR format, shape = [n_samples, n_samples]
          W[i, j] is assigned the weight of edge that connects i to j.
     """
     N = X.shape[0]
     distance_matrix = self.get_Adjacency_matrix(X)

     rows, cols = distance_matrix.nonzero()
     W = sparse.lil_matrix((N, N)) # best for W[i, nbrs_i] = w/np.sum(w)

     for i in range(N):
          nbrs_i = cols[rows == i]
          n_neighbors_i = len(nbrs_i)
          v = np.ones(n_neighbors_i, dtype=X.dtype)
          C = X[nbrs_i] - X[i]
          G = np.dot(C, C.T)
          trace = np.trace(G)
          if trace > 0:
               R = self.reg * trace
          else:
               R = self.reg
          G.flat[::n_neighbors_i + 1] += R
          w = solve(G, v, sym_pos=True)
          W[i, nbrs_i] = w / np.sum(w)

     return W


     def _lle(self, X):

          W = self.barycenter_graph(X)
          # find the null space of (I-W)
          M = eye(*W.shape, format=W.format) - W
          # symmetrize
          M = (M.T * M).tocsr()  

     # why skip 1 ????
     return null_space(M, self.n_components, k_skip=1, eigen_solver=self.eigen_solver,
                         seed=self.seed)


     def fit_transform(self, X):
          self.embeddings, self.error = self._lle(X)
          return self.embeddings


