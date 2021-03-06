"""Methods and classes for storing and manipulating the global
geometry of the physical problem.
"""
import numpy as np
from pydft.base import testmode
cell = None
"""Cell: default geometry to use globally throughout the code when no other
geometry is explicitly specified.
"""
def get_cell(cell_=None):
    """Returns the cell to use for calculations.
    """
    if cell_ is not None:
        return cell_
    else:
        return cell

def set_cell(cell_):
    """Sets the global cell to an already initialized instance.

    Args:
        cell_ (Cell): new global cell.
    """
    from pydft.bases.fourier import reset_cache
    reset_cache()    
    global cell
    cell = cell_

def set_geometry(R, S, X=None, Z=1, grid="MP", f=2):
    """Sets the global geometry that is used by default in all calculations.

    Args:
        R (numpy.ndarray): column lattice vectors of the unit cell for the
          problem.
        S (numpy.ndarray): of `int`; defines how many times to divide
          each of the lattice vectors when defining the descritizing
          grid.
        X (numpy.ndarray): of shape (N, 3), where `N` is the number of
          nucleii in the unit cell.
        Z (numpy.ndarray or int): specifying the size of charge on
          each nucleus in `X`.
        grid (str): one of ['MP', 'BCC']; defines the type of grid to use
            for sampling *real* space unit cell.
        f (int): number of electrons per orbital.
    """
    from pydft.bases.fourier import reset_cache
    reset_cache()
    global cell
    cell = Cell(R, S, X, Z, grid, f=f)
    return cell

class Cell(object):
    """Represents the unit cell in real space *and* the corresponding
    cell in reciprocal space.

    Args:
        R (numpy.ndarray): column lattice vectors of the unit cell for the
          problem.
        S (numpy.ndarray): of `int`; defines how many times to divide
          each of the lattice vectors when defining the descritizing
          grid.
        X (numpy.ndarray): of shape (N, 3), where `N` is the number of
          nucleii in the unit cell.
        Z (numpy.ndarray or int): specifying the size of charge on
          each nucleus in `X`.
        grid (str): one of ['MP', 'BCC']; defines the type of grid to use
            for sampling *real* space unit cell.
        f (int): number of electrons per orbital.

    Attributes:
        R (numpy.ndarray): column lattice vectors of the unit cell for the
          problem.
        S (numpy.ndarray): of `int`; defines how many times to divide
          each of the lattice vectors when defining the descritizing
          grid.
        X (numpy.ndarray): of shape (N, 3), where `N` is the number of
          nucleii in the unit cell.
        Z (numpy.ndarray or int): specifying the size of charge on
          each nucleus in `X`.
        vol (float): volume of the cell in real space.
        f (int): number of electrons per orbital.
    """
    def __init__(self, R, S, X=None, Z=1, grid="MP", f=2):
        self.R = np.array(R)
        self.S = np.array(S)
        self.vol = np.linalg.det(self.R)
        if X is None:
            self.X = np.array([[0,0,0]])
        else:
            self.X = np.array(X)
        self.Z = np.array([Z for i in range(len(self.X))])
        self.f = f
            
        self._M = None
        """numpy.ndarray: matrix of fractions used to define the points on which
        the functions are sampled in the unit cell.
        """
        self._N = None
        """numpy.ndarray: matrix of integers used in computing the Fourier transform of
        the unit cell sample points.
        """
        self._r = None
        """numpy.ndarray: points to sample the functions at in the unit cell.
        """
        self._G = None
        """numpy.ndarray: sample points in reciprocal space.
        """
        self._G2 = None
        """numpy.ndarray: magnitudes of the sample point vectors in reciprocal
        space.
        """
        self._K = None
        """numpy.ndarray: with shape (3, 3); holds the reciprocal lattice
        vectors for the problem.
        """
        self._Sf = None
        """numpy.ndarray: with length `self.X.shape[0]`; structure factors for
        the nucleii in the cell.
        """
        self._dr = None
        """numpy.ndarray: distance from the center of the cell to each
        of the sample points.
        """
        
        if grid != "MP":
            raise NotImplementedError("Haven't got BCC sampling in place yet.")

    @property
    def dr(self):
        """Returns a matrix of the distance from the center of the
        cell to each of the sample points.
        """
        if self._dr is None:
            center = np.sum(self.R, axis=1)/2.
            self._dr = self.r - center
        return self._dr
        
    @property
    def K(self):
        """Reciprocal lattice vectors for the problem. Has shape (3, 3).
        """
        if self._K is None:
            b1 = 2*np.pi*np.cross(self.R[:,1], self.R[:,2])/self.vol
            b2 = 2*np.pi*np.cross(self.R[:,2], self.R[:,0])/self.vol
            b3 = 2*np.pi*np.cross(self.R[:,0], self.R[:,1])/self.vol
            self._K = np.vstack((b1, b2, b3)).T
        return self._K

    @property
    def Sf(self):
        """Structure factor for the nuclei in the cell.
        """
        if self._Sf is None:
            self._Sf = np.sum(np.exp(-1j*np.dot(self.G, self.X.T)), axis=1)
        return self._Sf
    
    @property
    def r(self):

        """Points to sample the functions at in the unit cell.
        """
        if self._r is None:
            Sinv = np.diag(1./self.S)
            self._r = np.dot(self.M, np.dot(Sinv, self.R.T))
        return self._r

    @property
    def G(self):
        """Sample points in reciprocal space.
        """
        if self._G is None:
            self._G = 2*np.pi*np.dot(self.N, np.linalg.inv(self.R))
        return self._G

    @property
    def G2(self):
        """Magnitudes of the sample point vectors in reciprocal
        space.

        Returns:
            numpy.ndarray: of length `np.prod(S)` with magnitude of each `G`
              vector.
        """
        if self._G2 is None:
            self._G2 = np.linalg.norm(self.G, axis=1)**2
        return self._G2
    
    @property
    def M(self):
        """Returns the :math:`M` matrix of integers that determine points at which the
        functions are sampled in the unit cell.

        Examples:
            For `S = [2, 2, 1]`, the returned matrix is:

        .. code-block:: python

            np.ndarray([[0,0,0],
                        [1,0,0],
                        [0,1,0],
                        [1,1,0]], dtype=int)
        """
        if self._M is None:
            ms = np.arange(np.prod(self.S, dtype=int))
            m1 = np.fmod(ms, self.S[0])
            m2 = np.fmod(np.floor(ms/self.S[0]), self.S[1])
            m3 = np.fmod(np.floor(ms/(self.S[0]*self.S[1])), self.S[2])
            #Make sure we explicitly use an integer array; it's faster.
            self._M = np.asarray(np.vstack((m1, m2, m3)).T, dtype=int)
        return self._M

    @property
    def N(self):
        """"Returns the :math:`N` matrix of integers used in computing the
        Fourier transform of the unit cell sample points.
        """
        if self._N is None:
            result = []
            for i in range(3):
                odd = 1 if i % 2 == 1 else 0
                m = np.ma.array(self.M[:,i], mask=(self.M[:,i] <= self.S[i]/2))
                result.append(m-self.S[i])
            self._N = np.array(result).T
                
        return self._N

    def _latvec_plot(self, R=True, withpts=False, legend=False):
        """Plots the lattice vectors (for real or reciprocal space).
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        vecs = self.R if R else self.K
        for i in range(3):
            steps = np.linspace(0, 1, np.floor(10*np.linalg.norm(vecs[:,i])))
            Ri = vecs[:,i]
            Ri.shape = (1, 3)
            steps.shape = (len(steps), 1)
            line = np.dot(steps, Ri)
            ax.plot(line[:,0], line[:,1], line[:,2], label="R{0:d}".format(i+1))

        if withpts:
            pts = self.r if R else self.G
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='k')

        if legend:
            ax.legend()
            
        return (fig, ax)
            
    def plot(self, withpts=False):
        """Plots the unit cell.

        Args:
            withpts (bool): when True, the sampling points :attr:`r` are also
              plotted.
        """
        import matplotlib.pyplot as plt
        fig, ax = self._latvec_plot(withpts=withpts)
        plt.title("Real Lattice with Sampling Points")
        if not testmode:
            plt.show()

    def gplot(self, withpts=False):
        """Plots the reciprocal lattice vectors.

        Args:
            withpts (bool): when True, the sampling points in reciprocal space will
              also be plotted.
        """
        import matplotlib.pyplot as plt
        fig, ax = self._latvec_plot(R=False, withpts=withpts)
        plt.title("Reciprocal Lattice with Sampling Points")
        if not testmode:
            plt.show()
