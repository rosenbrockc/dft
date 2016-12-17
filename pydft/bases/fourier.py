"""Plane wave basis for the DFT solvers. Exposes the generic linear
algebra functions required by the other solvers and methods.
"""
import numpy as np
from pydft.geometry import get_cell
from pydft.solvers.la import matprod

cache = {}
"""dict: keys are one of the values in :data:`ops` representing the
specific operator; values are the :class:`numpy.ndarray` cached matrix
representations.
"""
ops = ["L", "Linv", "I", "J"]
"""list: of `str` names of the operators available in the
:data:`cache`.
"""

def reset_cache():
    """Resets the matrix caches for the global cell variable.
    """
    global cache
    cache = {k: None for k in ops}

def set_cache(op, cell, M):
    """Caches the operator matrix for the specified cell if it is the global
    cell.

    Args:
        op (str): name of the operator being cached.
        cell (pydft.geometry.Cell): describing the unit cell and sampling points.
        M (numpy.ndarray): matrix to cache for the operator.
    """
    from pydft.geometry import cell as gcell
    global cache
    if cell is gcell:
        cache[op] = M

def get_cache(op, cell):
    """Caches the operator matrix for the specified cell if it is the global
    cell.

    Args:
        op (str): name of the operator being cached.
        cell (pydft.geometry.Cell): describing the unit cell and sampling points.
    """
    from pydft.geometry import cell as gcell
    global cache
    if op in cache and cache[op] is not None:
        if cell is None or cell is gcell:
            return cache[op]

def psi(V, W, cell, forceR=True):
    """Calculates the normalized wave functions using the basis coefficients.

    Args:
        V (pydft.potential.Potential): describing the potential for the
          particles.
        W (numpy.ndarray): wave function sample points.
        cell (pydft.geometry.Cell): describing the unit cell and sampling
          points.
        forceR (bool): forces the result to be real.
    """
    WN = Y(W, cell)
    mu = np.dot(WN.conjugate().T, H(V, WN, cell))
    epsilon, D = np.linalg.eig(mu)
    if forceR:
        epsilon = np.real(epsilon)
        
    return (np.dot(WN, D), epsilon)
        
def Y(W, cell=None):
    """Returns the normalized wave function `Y` (eq. 6 in Ps2).

    Args:
        W (numpy.ndarray): wave function sample points.
        cell (pydft.geometry.Cell): describing the unit cell and sampling
          points.
    """
    from scipy.linalg import sqrtm
    Usq = sqrtm(np.linalg.inv(U(W, cell)))
    return np.dot(W, Usq)
        
def gradE(V, W, cell=None):
    """Calculates the gradient of `E` with respect to the wave functions `W`.

    Args:
        V (pydft.potential.Potential): describing the potential for the
          particles.
        W (numpy.ndarray): wave function sample points.
        cell (pydft.geometry.Cell): describing the unit cell and sampling
          points.
    """
    Uinv = np.linalg.inv(U(W, cell))
    A = H(V, W, cell)
    B1 = np.dot(W, Uinv)
    B2 = np.dot(W.conjugate().T, H(V, W, cell))
    B = O(np.dot(B1, B2))

    return cell.f*np.dot(A-B, Uinv)
          
def H(V, W, cell=None):
    """Returns the Hamiltonian operator H(W) in Fourier.

    Args:
        V (pydft.potential.Potential): describing the potential for the
          particles.
        W (numpy.ndarray): wave function sample points.
        cell (pydft.geometry.Cell): describing the unit cell and sampling
          points.
    """
    A = -1./2*L(W, cell)

    #Solve poisson's equation to get the electron-electron interactions.
    from pydft.solvers.la import diagprod
    n_, WUinv = n(W, cell)
    Phi = -4*np.pi*Linv(O(J(n_, cell), cell), cell)
    Veff = V.Vdual() + Jdag(O(Phi, cell), cell)
    IW = I(W, cell)
    B = Idag(diagprod(Veff, IW), cell)
    
    return A + B
        
def n(W, cell=None):
    """Calculate the density vector.

    Args:
        W (numpy.ndarray): wave function sample points.
        cell (pydft.geometry.Cell): describing the unit cell and sampling
          points.

    Returns:
        tuple: `(n, W Uinv)` so that the `W Uinv` can be re-used by later
        calculations.
    """
    from pydft.solvers.la import diagouter
    Uinv = np.linalg.inv(U(W, cell))
    WUinv = np.dot(W, Uinv)
    A = I(WUinv, cell)
    B = I(W, cell)
    return (cell.f*diagouter(A, B), WUinv)
        
def E(V, W, cell=None, forceR=True):
    """Calculates the energy for the specified wave functions and potential.

    Args:
        V (pydft.potential.Potential): describing the potential for the
          particles.
        W (numpy.ndarray): wave function sample points.
        cell (pydft.geometry.Cell): describing the unit cell and sampling
          points.
        forceR (bool): forces the result to be real.
    """
    n_, WUinv = n(W, cell)   
    Vd = V.Vdual()
    if len(Vd.shape) == 2:
        Vd.shape = len(Vd)

    A = -cell.f/2.*np.trace(np.dot(W.conjugate().T, L(WUinv, cell)))
    B = np.dot(V.Vdual(), n_)

    #Solve poisson's equation to get the electron-electron interactions.
    Phi = -4*np.pi*Linv(O(J(n_, cell), cell), cell)
    C = 1./2*np.dot(n_.conjugate().T, Jdag(O(Phi, cell), cell))
    
    if forceR:
        return np.real(A + B + C)
    else:
        return (A + B + C)
        
def U(W, cell=None):
    """Calculates operator `U` for the given wave functions.

    Args:
        W (numpy.ndarray): wave function sample points.
        cell (pydft.geometry.Cell): describing the unit cell and sampling points.
    """
    return np.dot(W.conjugate().T, O(W, cell))
        
def L(v=None, cell=None):
    """Returns the Laplacian operator matrix for the plane wave basis.

    Args:
        v (numpy.ndarray): if None, then return the matrix :math:`\mathbb{L}`,
          else, return :math:`\mathbb{L}\cdot v`.
        cell (pydft.geometry.Cell): that describes the unit cell and
          sampling points for real and reciprocal space.
    """
    Lij = get_cache("L", cell)
    if Lij is None:
        cell = get_cell(cell)
        Lij = -cell.vol*np.diag(cell.G2)
        set_cache("L", cell, Lij)

    if v is not None:
        return matprod(Lij, v)
    else:
        return Lij

def Linv(v=None, cell=None):
    """Returns the *inverse* Laplacian operator matrix for the plane
    wave basis.

    Args:
        v (numpy.ndarray): if None, then return the matrix
          :math:`\mathbb{L}^{-1}`, else, return
          :math:`\mathbb{L}^{-1}\cdot v`.
        cell (pydft.geometry.Cell): that describes the unit cell and
          sampling points for real and reciprocal space.
    """
    iLij = get_cache("Linv", cell)

    if iLij is None:
        cell = get_cell(cell)
        iLij = -1./cell.vol*np.diag(1./cell.G2)
        #Explicitly set the infinity to zero. We compensate for this
        #elsewhere using a constant charge density throughout the
        #cell.
        iLij[0, 0] = 0.
        set_cache("Linv", cell, iLij)

    if v is not None:
        return matprod(iLij, v)
    else:
        return iLij
    
def O(v=None, cell=None):
    """Returns the basis overlap operator, optionally applied to the
    specified vector.

    Args:
        v (numpy.ndarray): if None, then return the matrix :math:`\mathbb{O}`,
          else, return :math:`\mathbb{O}\cdot v`.
        cell (pydft.geometry.Cell): that describes the unit cell and
          sampling points for real and reciprocal space.
    """
    cell = get_cell(cell)
    if v is None:
        return np.diag(np.ones(np.prod(cell.S))*cell.vol)
    else:
        return cell.vol*v

def _Iap(cell, inv=False):
    """Returns the I or J matrix for the given cell.
    """
    #We don't want integer division; cast the S matrix as a float.
    Sf = np.asarray(cell.S, float)
    if inv:
        imul = -1.j
    else:
        imul = 1.j
    Iap = np.exp(2*np.pi*imul*np.dot(cell.N/Sf, cell.M.T))
    return Iap

def Idag(v=None, cell=None):
    """Computes the complex conjugate of the `I` operator for Fourier basis.

    Args:
        v (numpy.ndarray): if None, then return the matrix :math:`\mathbb{I^\dag}`,
          else, return :math:`\mathbb{I^\dag}\cdot v`.
        cell (pydft.geometry.Cell): that describes the unit cell and
          sampling points for real and reciprocal space.
    """
    #It turns out that for Fourier, the complex conjugate only differs by a -1
    #on the i (symmetric in R and G), so that we can return J instead.
    cell = get_cell(cell)
    def ifft(X):
        FB = np.fft.ifftn(np.reshape(X, cell.S, order='F'))
        return np.reshape(FB, X.shape, order='F')
    return matprod(ifft, v)*np.prod(cell.S)

def I(v=None, cell=None, fast=True):
    """Returns the forward transform matrix :math:`\mathbb{I}`,
    optionally applied to a vector.

    Args:
        v (numpy.ndarray): if None, then return the matrix :math:`\mathbb{I}`,
          else, return :math:`\mathbb{I}\cdot v`.
        cell (pydft.geometry.Cell): that describes the unit cell and
          sampling points for real and reciprocal space.
    """
    cell = get_cell(cell)
    if fast:
        def fft(X):
            FF = np.fft.fftn(np.reshape(X, cell.S, order='F'))
            return np.reshape(FF, X.shape, order='F')
        return matprod(fft, v)
    else:
        Iap = get_cache("I", cell)
        if Iap is None:
            Iap = _Iap(cell)
            set_cache("I", cell, Iap)

        if v is not None:
            return matprod(Iap, v)
        else:
            return Iap

def Jdag(v=None, cell=None):
    """Computes the complex conjugate of the `J` operator for Fourier basis.

    Args:
        v (numpy.ndarray): if None, then return the matrix :math:`\mathbb{J^\dag}`,
          else, return :math:`\mathbb{J^\dag}\cdot v`.
        cell (pydft.geometry.Cell): that describes the unit cell and
          sampling points for real and reciprocal space.
    """
    #It turns out that for Fourier, the complex conjugate only differs by a -1
    #on the i (symmetric in R and G), so that we can return I instead.
    cell = get_cell(cell)
    def fft(X):
        FF = np.fft.fftn(np.reshape(X, cell.S, order='F'))
        return np.reshape(FF, X.shape, order='F')
    return matprod(fft, v)/np.prod(cell.S)
        
def J(v=None, cell=None, fast=True):
    """Returns the forward transform matrix :math:`\mathbb{J}`,
    optionally applied to a vector.

    Args:
        v (numpy.ndarray): if None, then return the matrix :math:`\mathbb{J}`,
          else, return :math:`\mathbb{J}\cdot v`.
        cell (pydft.geometry.Cell): that describes the unit cell and
          sampling points for real and reciprocal space.
    """
    cell = get_cell(cell)
    if fast:
        def ifft(X):
            FB = np.fft.ifftn(np.reshape(X, cell.S, order='F'))
            return np.reshape(FB, X.shape, order='F')
        return matprod(ifft, v)
    else:
        Jap = get_cache("J", cell)
        if Jap is None:
            Sprod = np.prod(cell.S)
            Jap = _Iap(cell, True)/Sprod
            set_cache("J", cell, Jap)

        if v is not None:
            return matprod(Jap, v)
        else:
            return Jap
