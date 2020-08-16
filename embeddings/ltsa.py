import warnings
from nldr import NLDR
from scipy import sparse
from sparse import eye, csr_matrix
from sparse.linalg import svds, eigs
from scipy.linalg import solve
from .utils import eigen_decomposition, null_space

class LTSA(NLDR):


     def _ltsa(self, X):
          N, d_in = X.shape

          if n_components > d_in:
               raise ValueError("output dimension must be less than or equal "
                              "to input dimension")

          rows, cols = self.get_Adjacency_matrix(X).nonzero()
          M = sparse.dok_matrix((N, N))

          for i in range(N):
               neighbors_i = cols[rows == i]
               n_neighbors_i = len(neighbors_i)
               use_svd = n_neighbors_i > d_in # when n>p, full rank
               Xi = X[neighbors_i]
               Xi -= Xi.mean(0)

               # compute n_components largest eigenvalues of Xi * Xi^T
               # changed from scipy.linalg to scipy.sparse.linalg
               # not sure how it performs
               if use_svd:
                    # unordered 
                    v, s, _ = svds(Xi.astype('float'), k=self.n_components)
               else:
                    Ci = Xi * Xi.T
                    # unordered 
                    s, v = eigs(Ci.astype('float'), k=self.n_components)

               index = np.argsort(-s)
               v = v[:, index]

               # megaman
               # with warnings.catch_warnings():
               #     # sparse will complain this is better with lil_matrix but it doesn't work
               #     warnings.simplefilter("ignore")
               Gi = np.zeros((n_neighbors_i, n_components + 1))
               Gi[:, 1:] = v
               Gi[:, 0] = 1. / np.sqrt(n_neighbors_i)
               GiGiT = Gi @ Gi.T
               nbrs_x, nbrs_y = np.meshgrid(neighbors_i, neighbors_i)

               M[nbrs_x, nbrs_y] -= GiGiT
               M[neighbors_i, neighbors_i] += 1

          M = sparse.csr_matrix(M)
          return null_space(M, self.n_components, k_skip=1, eigen_solver=self.eigen_solver,
                         seed=self.seed,solver_kwds=self.solver_kwds)

     def fit_transform(self, X):
          self.embeddings, self.error = self._ltsa(X)
          
          return self.embeddings