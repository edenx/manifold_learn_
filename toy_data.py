from sklearn.utils import check_random_state
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections

# from matplotlib.ticker import NullFormatter

class toy():
     """ base class of simulated simple manifold
     """
     __metaclass__ = ABCMeta

     def __init__(self, N, seed=100):
          self.N = N
          self.rng = check_random_state(seed)
          self.X = None
          self.col_vec = None

     @abstractmethod
     def gen(self):
          """
          returns data and color array
          """
          raise NotImplementedError
     
     @staticmethod
     def plot(X, col):
          fig = plt.figure(figsize=(4,5))
          ax = fig.add_subplot(111, projection='3d')
          ax.scatter(
               X[:, 0], X[:, 1], X[:, 2], 
               c=col, cmap=plt.cm.Spectral)
          ax.view_init(4, -72)
          plt.show()

     def _genereate_noises(self, sigmas, size, dims):
          rng = self.rng
          if isinstance(sigmas, (collections.Sequence, np.ndarray)):
               assert len(sigmas) == dims, \
                    'The size of sigmas should be the same as noises dimensions'
               return rng.multivariate_normal(np.zeros(dims),
                                                       np.diag(sigmas), size)
          else:
               return rng.normal(0, sigmas, [size, dims])

     def _add_noises(self, X, sigmas=0.1):
          size, dims = X.shape
          noises = self._genereate_noises(sigmas, size, dims)

          return X + noises

     def _add_dims(self, X, dims, sigmas=0.1):
          if dims == 0:
               return X
          else:
               noises = self._genereate_noises(sigmas, X.shape[0], dims)

               return np.hstack((X, noises))

     def add_noises(
          self,
          X,
          sigma_primary=0.1, 
          additional_dims=0, 
          sigma_additional=0.1
          ):
          noisy_X = X.copy()
          noisy_X = self._add_noises(noisy_X, sigma_primary)
          noisy_X = self._add_dims(noisy_X, additional_dims, sigma_additional)
          
          return noisy_X

class Plane2D(toy):
     
     def gen(
          self, 
          copy=True,
          sigma_primary=0.1, 
          additional_dims=0, 
          sigma_additional=0.1
          ):
          
          xMax = int(np.floor(np.sqrt(self.N)))
          yMax = int(np.ceil(self.N/xMax))

          X = np.array([(x,y,0) for x in range(xMax) for y in range(yMax)])
          col_vec = np.array([y for x in range(xMax) for y in range(yMax)])

          self.X, self.col_vec = X, col_vec
          self.noisy_X = self.add_noises(
               self.X, 
               sigma_primary, 
               additional_dims,
               sigma_additional
               )

          if copy:
               return self.noisy_X, self.X, self.col_vec

class Swiss_roll(toy):

     def __init__(self, N, seed=100, with_hole=False):
          super().__init__(N, seed)
          self.with_hole = with_hole

     @property
     def row_dim(self):
          return self.N

     @row_dim.setter
     def row_dim(self, M):
          self.N = M

     def _gen(self):

          t = (3 * np.pi/2) * (1 + 2 * self.rng.rand(self.N, 1))
          y = 21 * self.rng.rand(self.N)[:, np.newaxis]
          X = np.hstack((t*np.cos(t), y, t*np.sin(t)))

          return X, t[:, 0]

     def gen(
          self, 
          copy=True,
          sigma_primary=0.1, 
          additional_dims=0, 
          sigma_additional=0.1
          ):
          if self.with_hole:
               # double the row dimension for cropping
               N = self.N
               self.row_dim = 2 * N
               X_, t_ = self._gen()
               I = (t_ > 9) & (t_ < 12) & (X_[:, 1] > 9) & (X_[:, 1] < 14)

               I2 = np.arange(2 * N) 
               I2 = I2[~I]
               I2 = I2[:N]

               X = X_[I2]
               col_vec = t_[I2]
          
          else:
               X, col_vec = self._gen()

          self.X, self.col_vec = X, col_vec
          self.noisy_X = self.add_noises(
               self.X, 
               sigma_primary, 
               additional_dims,
               sigma_additional
               )

          if copy:
               return self.noisy_X, self.X, self.col_vec


class Corner_planes(toy):

     def gen(
          self, 
          copy=True,
          sigma_primary=0.1, 
          additional_dims=0, 
          sigma_additional=0.1
          ):
          xMax = int(np.floor(np.sqrt(self.N)))
          yMax = int(np.ceil(self.N/xMax))
          cornerPoint = int(np.floor(yMax/2))

          X = np.array([
               (x,y,0) if y <= cornerPoint 
                         else (
                              x,
                              cornerPoint+(y-cornerPoint)*np.cos(np.pi/180),
                              (y-cornerPoint)*np.sin(np.pi/180)
                              )
                         for x in range(xMax) 
                         for y in range(yMax)
                         ])
          col_vec = np.array([y for x in range(xMax) for y in range(yMax)])

          self.X, self.col_vec = X, col_vec
          self.noisy_X = self.add_noises(
               self.X, 
               sigma_primary, 
               additional_dims,
               sigma_additional
               )

          if copy:
               return self.noisy_X, self.X, self.col_vec

class Punctured_sphere(toy):
     """ by Saul & Roweis"""
     def gen(
          self, 
          copy=True,
          sigma_primary=0.1, 
          additional_dims=0, 
          sigma_additional=0.1
          ):
          inc = 9/np.sqrt(self.N)
          xx, yy = np.meshgrid(np.arange(-5, 5, inc), np.arange(-5, 5, inc))
          rr2 = xx.flatten()**2 + yy.flatten()**2
          ii = np.argsort(rr2)

          Y = np.vstack(
               (
                    xx.flatten()[ii[:self.N]], 
                    yy.flatten()[ii[:self.N]]
                    ))
          a = 4/(4 + (Y**2).sum(0))
          X = np.vstack([a*Y[0], a*Y[1], 2*(1-a)]).T

          self.X, self.col_vec = X, X[:, 2]
          self.noisy_X = self.add_noises(
               self.X, 
               sigma_primary, 
               additional_dims,
               sigma_additional
               )

          if copy:
               return self.noisy_X, self.X, self.col_vec

# class Twin_peaks(toy):
#      def gen(self, copy=True):

class Full_sphere(toy):
     def gen(
          self, 
          copy=True,
          sigma_primary=0.1, 
          additional_dims=0, 
          sigma_additional=0.1
          ):
          rng = self.rng
          theta = np.pi * rng.rand(self.N)
          phi = 2 * np.pi * rng.rand(self.N)
          X = np.vstack(
               (
               np.sin(theta)*np.cos(phi), 
               np.sin(theta)*np.sin(phi), 
               np.cos(theta)
               )).T

          self.X, self.col_vec = X, np.cos(theta).flatten()
          self.noisy_X = self.add_noises(
               self.X, 
               sigma_primary, 
               additional_dims,
               sigma_additional
               )

          if copy:
               return self.noisy_X, self.X, self.col_vec


class Half_sphere(toy):
     def gen(
          self, 
          copy=True,
          sigma_primary=0.1, 
          additional_dims=0, 
          sigma_additional=0.1
          ):
          rng = self.rng
          theta = np.pi * rng.rand(self.N)
          phi = 2 * np.pi * rng.rand(self.N)

          X = np.vstack(
               (
                    np.cos(theta)*np.cos(phi), 
                    np.cos(theta)*np.sin(phi), 
                    np.sin(theta)
                    )).T

          self.X, self.col_vec = X, np.sin(theta).flatten()
          self.noisy_X = self.add_noises(
               self.X, 
               sigma_primary, 
               additional_dims,
               sigma_additional
               )

          if copy:
               return self.noisy_X, self.X, self.col_vec

class Torus(toy):
     def gen(
          self, 
          copy=True,
          sigma_primary=0.1, 
          additional_dims=0, 
          sigma_additional=0.1
          ):
          rng = self.rng
          t1 = 2 * np.pi * rng.rand(self.N)
          t2 = 2 * np.pi * rng.rand(self.N)
          X = np.vstack(
               (
                    (2+np.cos(t1))*np.cos(t2),
                    (2+np.cos(t1))*np.sin(t2), 
                    np.sin(t1)
                    )).T

          self.X, self.col_vec = X, t2.flatten()
          self.noisy_X = self.add_noises(
               self.X, 
               sigma_primary, 
               additional_dims,
               sigma_additional
               )

          if copy:
               return self.noisy_X, self.X, self.col_vec

if __name__ == '__main__':
     # test 
     N = 2000

     # plane = Plane2D(N)
     # noisy_X, X, col = plane.gen()
     # plane.plot(noisy_X, col)

     # swiss_roll = Swiss_roll(N, with_hole=True)
     # noisy_X, X, col = swiss_roll.gen()
     # swiss_roll.plot(noisy_X, col)


     # corner_planes = Corner_planes(N)
     # noisy_X, X, col = corner_planes.gen()
     # corner_planes.plot(noisy_X, col)

     # punctured_sphere = Punctured_sphere(N)
     # noisy_X, X, col = punctured_sphere.gen()
     # punctured_sphere.plot(noisy_X, col)

     # full_sphere = Full_sphere(N)
     # noisy_X, X, col = full_sphere.gen()
     # full_sphere.plot(noisy_X, col)

     # half_sphere = Half_sphere(N)
     # noisy_X, X, col = half_sphere.gen()
     # half_sphere.plot(noisy_X, col)

     torus = Torus(N)
     noisy_X, X, col = torus.gen()
     torus.plot(noisy_X, col)




