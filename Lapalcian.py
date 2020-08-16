"""
Compute Laplacian matrix for spectral embedding methods
Currently only methods to compute symmetric-normalised Laplacian 
and renomalised Lapalcian
Where does the 'method' attribute in megaman comes from???
"""
import numpy as np
from scipy.sparse import isspmatrix
from abc import ABCMeta, abstractmethod


class Laplacian(object):
     """Base class for computing laplacian matrices

     Notes
     -----
     The methods here all return the negative of the standard
     Laplacian definition.
     """
     __metaclass__ = ABCMeta
     symmetric = False

     def __init__(
          self
          , symmetrize_input=True
          # , scaling_epps=None
          , full_output=False
          ):
          self.symmetrize_input = symmetrize_input
          # self.scaling_epps = scaling_epps
          self.full_output = full_output

     @staticmethod
     def _symmetrize(A):
          B = 0.5 * (A + A.T)
          # for sparse, manual symmetrization 
          if isspmatrix(B):
               B.setdiag(1)
          return B

     def compute_laplacian(self, affinity_matrix):
          affinity_matrix = check_array(affinity_matrix, copy=False, dtype=float,
                                        accept_sparse=['csr', 'csc', 'coo'])
          if self.symmetrize_input:
               affinity_matrix = self._symmetrize(affinity_matrix)

          if isspmatrix(affinity_matrix):
               affinity_matrix = affinity_matrix.tocoo()
          else:
               affinity_matrix = affinity_matrix.copy()

          lap, lapsym, w = self._compute_laplacian(affinity_matrix)

          # if self.scaling_epps is not None and self.scaling_epps > 0.:
          #      if isspmatrix(lap):
          #           lap.data *= 4 / (self.scaling_epps ** 2)
          #      else:
          #           lap *= 4 / (self.scaling_epps ** 2)

          if self.full_output:
               return lap, lapsym, w
          else:
               return lap

     @abstractmethod
     def _compute_laplacian(self, aff_mat):
          raise NotImplementedError()

class UnNormLap(Laplacian):
    name = 'unnormalized'
    symmetric = True

    def _compute_laplacian(self, aff_mat):
        w = _degree(aff_mat)
        _subtract_from_diagonal(aff_mat, w)
        return aff_mat, aff_mat, w

class SymNormLap(Laplacian):
     name = 'symmetricnormalized'
     symmetric = True

     def _compute_laplacian(self, aff_mat):
          w, nonzero = _normalize_laplacian(aff_mat, symmetric=True, degree_exp=0.5)
          _subtract_from_diagonal(aff_mat, nonzero)
          return aff_mat, aff_mat, w


class RenormLap(Laplacian):
     name = 'renormalized'
     symmetric = False

     def __init__(
          self
          , symmetrize_input=True
          # , scaling_epps=None
          , full_output=False
          , renormalization_exponent=1
          ):
          self.symmetrize_input = symmetrize_input
          # self.scaling_epps = scaling_epps
          self.full_output = full_output
          self.renormalization_exponent = renormalization_exponent

     def _compute_laplacian(self, aff_mat):
          _normalize_laplacian(aff_mat, symmetric=True,
                              degree_exp=self.renormalization_exponent)
          # returns affinity matrix directly, unnecessary?
          # TODO: rewrite the symmetrisation of Laplacian?
          lapsym = aff_mat.copy() 
          _, _ = _normalize_laplacian(lapsym, symmetric=True, degree_exp=0.5)

          w, nonzero = _normalize_laplacian(aff_mat, symmetric=False)
          _subtract_from_diagonal(aff_mat, nonzero)

          return aff_mat, lapsym, w




"""
Utility functions, assuming either dense or coo matrix
from megaman/geometry/laplacian.py
"""


def _degree(aff_mat):
    return np.asarray(aff_mat.sum(1)).squeeze()


def _divide_along_rows(aff_mat, vals):
    if isspmatrix(aff_mat):
        aff_mat.data /= vals[aff_mat.row]
    else:
        aff_mat /= vals[:, np.newaxis]


def _divide_along_cols(aff_mat, vals):
    if isspmatrix(aff_mat):
        aff_mat.data /= vals[aff_mat.col]
    else:
        aff_mat /= vals


def _normalize_laplacian(aff_mat, symmetric=False, degree_exp=None):
    w = _degree(aff_mat)
    w_nonzero = (w != 0)
    w[~w_nonzero] = 1

    if degree_exp is not None:
        w **= degree_exp

    if symmetric:
        _divide_along_rows(aff_mat, w)
        _divide_along_cols(aff_mat, w)
    else:
        _divide_along_rows(aff_mat, w)

    return w, w_nonzero


def _subtract_from_diagonal(aff_mat, vals):
    if isspmatrix(aff_mat):
        aff_mat.data[aff_mat.row == aff_mat.col] -= vals
    else:
        aff_mat.flat[::aff_mat.shape[0] + 1] -= vals
