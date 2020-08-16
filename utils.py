# drop eigenvectors of zero evalues
import warnings
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import lobpcg, eigs, eigsh
from sklearn.utils.validation import check_random_state

try:
    from pyamg import smoothed_aggregation_solver
    PYAMG_LOADED = True
    EIGEN_SOLVERS.append('amg')
AMG_KWDS = ['strength', 'aggregate', 'smooth', 'max_levels', 'max_coarse']

def _is_symmetric(G, tol = 1e-8):
    if sparse.isspmatrix(G):
        conditions = np.abs((G - G.T).data) < tol
    else:
        conditions = np.abs((G - G.T)) < tol
    return(np.all(conditions))

def eigen_decomposition(
     G
     , n_components=8
     , eigen_solver='amg'
     , seed=None
     , drop_first=True
     , largest=True
     , solver_kwds=None
     ):
     """
     Function to compute the eigendecomposition of a square matrix.

     Parameters
     ----------
     G : array_like or sparse matrix
          The square matrix for which to compute the eigen-decomposition.
     n_components : integer, optional
          The number of eigenvectors to return
     eigen_solver : {'auto', 'dense', 'arpack', 'lobpcg', or 'amg'}
          'auto' :
               attempt to choose the best method for input data (default)
          'dense' :
               use standard dense matrix operations for the eigenvalue decomposition.
               For this method, M must be an array or matrix type.
               This method should be avoided for large problems.
          'arpack' :
               use arnoldi iteration in shift-invert mode. For this method,
               M may be a dense matrix, sparse matrix, or general linear operator.
               Warning: ARPACK can be unstable for some problems.  It is best to
               try several random seeds in order to check results.
          'lobpcg' :
               Locally Optimal Block Preconditioned Conjugate Gradient Method.
               A preconditioned eigensolver for large symmetric positive definite
               (SPD) generalized eigenproblems.
          'amg' :
               Algebraic Multigrid solver (requires ``pyamg`` to be installed)
               It can be faster on very large, sparse problems, but may also lead
               to instabilities.
     random_state : int seed, RandomState instance, or None (default)
          A pseudo random number generator used for the initialization of the
          lobpcg eigen vectors decomposition when eigen_solver == 'amg'.
          By default, arpack is used.
     solver_kwds : any additional keyword arguments to pass to the selected eigen_solver

     Returns
     -------
     lambdas, diffusion_map : eigenvalues, eigenvectors
     """
     n_nodes = G.shape[0]
     if drop_first:
          n_components = n_components + 1

     # TODO: discard all eigenvaectors that has 0 evals?
     # n_components = n_components + 1

     rng = check_random_state(seed)

     # Convert G to best type for eigendecomposition
     if sparse.issparse(G):
          if G.getformat() is not 'csr':
               G.tocsr()
     G = G.astype(np.float)

     # Check for symmetry
     is_symmetric = _is_symmetric(G)

     # Try Eigen Methods:
     # if eigen_solver == 'arpack':
     #      # This matches the internal initial state used by ARPACK
     #      v0 = random_state.uniform(-1, 1, G.shape[0])
     #      if is_symmetric:
     #           if largest:
     #                which = 'LM'
     #           else:
     #                which = 'SM'
     #           lambdas, diffusion_map = eigsh(G, k=n_components, which=which,
     #                                         v0=v0,**(solver_kwds or {}))
     #      else:
     #           if largest:
     #                which = 'LR'
     #           else:
     #                which = 'SR'
     #           lambdas, diffusion_map = eigs(G, k=n_components, which=which,
     #                                         **(solver_kwds or {}))
     #      lambdas = np.real(lambdas)
     #      diffusion_map = np.real(diffusion_map)
     if not is_symmetric:
               raise ValueError("lobpcg requires symmetric matrices.")
     n_find = min(n_nodes, 5 + 2 * n_components)
     X = rng.rand(n_nodes, n_find)

     if PYAMG_LOADED:
          # separate amg & lobpcg keywords:
          if solver_kwds is not None:
               amg_kwds = {}
               lobpcg_kwds = solver_kwds.copy()
               for kwd in AMG_KWDS:
                    if kwd in solver_kwds.keys():
                    amg_kwds[kwd] = solver_kwds[kwd]
                    del lobpcg_kwds[kwd]
          else:
               amg_kwds = None
               lobpcg_kwds = None
          if not sparse.issparse(G):
               warnings.warn("AMG works better for sparse matrices")
          # Use AMG to get a preconditioner and speed up the eigenvalue problem.
          ml = smoothed_aggregation_solver(check_array(G, accept_sparse = ['csr']),**(amg_kwds or {}))
          M = ml.aspreconditioner()

          X[:, 0] = (G.diagonal()).ravel()
          eigenvalues, eigenvectros = lobpcg(G, X, M=M, largest=largest,**(lobpcg_kwds or {}))
          
     else:
          eigenvalues, eigenvectros = lobpcg(G, X, largest=largest,**(solver_kwds or {}))

     sort_order = np.argsort(eigenvalues)

     if largest:
          eigenvalues = eigenvalues[sort_order[::-1]]
          eigenvectros = eigenvectros[:, sort_order[::-1]]
     else:
          eigenvalues = eigenvalues[sort_order]
          eigenvectros = eigenvectros[:, sort_order]

     eigenvalues = eigenvalues[:n_components]
     eigenvectros = eigenvectros[:, :n_components]

     return eigenvalues, eigenvectros
     # elif eigen_solver == 'dense':
     #      if sparse.isspmatrix(G):
     #           G = G.todense()
     #      if is_symmetric:
     #           lambdas, diffusion_map = eigh(G,**(solver_kwds or {}))
     #      else:
     #           lambdas, diffusion_map = eig(G,**(solver_kwds or {}))
     #           sort_index = np.argsort(lambdas)
     #           lambdas = lambdas[sort_index]
     #           diffusion_map[:,sort_index]
     #      if largest:# eigh always returns eigenvalues in ascending order
     #           lambdas = lambdas[::-1] # reverse order the e-values
     #           diffusion_map = diffusion_map[:, ::-1] # reverse order the vectors
     #      lambdas = lambdas[:n_components]
     #      diffusion_map = diffusion_map[:, :n_components]
     

     def null_space(M, k, k_skip=1, eigen_solver='amg',
               seed=None, solver_kwds=None):
          try:
               M = 2.0 * sparse.identity(M.shape[0]) + M
               n_components = min(k + k_skip + 10, M.shape[0])
               eigen_values, eigen_vectors = eigen_decomposition(
                    M, 
                    n_components,
                    eigen_solver=eigen_solver,
                    drop_first=False,
                    largest=False,
                    seed=seed,
                    solver_kwds=solver_kwds)

               eigen_values = eigen_values - 2
               index = np.argsort(np.abs(eigen_values))
               eigen_values = eigen_values[index]
               eigen_vectors = eigen_vectors[:, index]

               return eigen_vectors[:, k_skip:k+1], np.sum(eigen_values[k_skip:k+1])

          except np.linalg.LinAlgError: # try again with bigger increase
               warnings.warn("LOBPCG failed the first time. Increasing Pos Def adjustment.")
               raise NotImplementedError