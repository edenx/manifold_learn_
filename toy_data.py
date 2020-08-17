from sklearn.utils import check_random_state
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import NullFormatter

class toy():
     """ base class of simulated simple manifold
     """
     __metaclass__ = ABCMeta

     def __init__(self, N, seed=100):
          self.N = N
          self.rng = check_random_state(seed)

     @ abstractmethod
     def gen(self):
          """
          returns data and color array
          """
          raise NotImplementedError

     def plot(self):
          fig = plt.figure(figsize=(4,5))
          ax = fig.add_subplot(111, projection='3d')
          ax.scatter(
               self.X[:, 0], self.X[:, 1], self.X[:, 2], 
               c=self.col_vec, cmap=plt.cm.Spectral)
          ax.view_init(4, -72)
          plt.show()
     

class Plane2D(toy):
     
     def gen(self, copy=True):
          
          xMax = int(np.floor(np.sqrt(self.N)))
          yMax = int(np.ceil(self.N/xMax))

          X = np.array([(x,y,0) for x in range(xMax) for y in range(yMax)])
          col_vec = np.array([y for x in range(xMax) for y in range(yMax)])

          self.X, self.col_vec = X, col_vec

          if copy:
               return self.X, self.col_vec

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

     def gen(self, copy=True):
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

          if copy:
               return self.X, self.col_vec


class Corner_planes(toy):

     def gen(self, copy=True):
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

          if copy:
               return self.X, self.col_vec

# TODO: debug
class Punctured_sphere(toy):
     """ by Saul & Roweis"""
     def gen(self, copy=True):
          inc = 9/np.sqrt(self.N)
          xx, yy = np.meshgrid(np.arange(-5, 5, inc), np.arange(-5, 5, inc))
          rr2 = xx.flatten()**2 + yy.flatten()**2
          ii = np.argsort(rr2)

          Y = np.vstack([xx.flatten()[ii[:self.N]], yy.flatten()[ii[:self.N]]])
          a = 4/(4 + (Y**2).sum(0))
          X = np.vstack([a*Y[0], a*Y[1], 2*(1-a)]).T

          self.X, self.col_vec = X, X[:, 2]

          if copy:
               return self.X, self.col_vec


if __name__ == '__main__':
     # test 
     N = 2000

     # plane = Plane2D(N)
     # plane.gen(copy=False)
     # plane.plot()

     # swiss_roll = Swiss_roll(N, with_hole=True)
     # swiss_roll.gen(copy=False)
     # swiss_roll.plot()


     # corner_planes = Corner_planes(N)
     # corner_planes.gen(copy=False)
     # corner_planes.plot()

     # bug with the implementation
     punctured_sphere = Punctured_sphere(N)
     punctured_sphere.gen(copy=False)
     punctured_sphere.plot()



