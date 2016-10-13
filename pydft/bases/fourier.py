"""Plane wave basis for the DFT solvers. Exposes the generic linear
algebra functions required by the other solvers and methods.
"""
import numpy as np

_L = None
"""numpy.ndarray: Laplacian matrix for the problem. Represents the
operator for the *global* cell only.
"""
_Linv = None
"""numpy.ndarray: inverse Laplacian matrix for the problem. Represents
the operator for the *global* cell only.
"""
_I = None
"""numpy.ndarray: forward transform matrix for the problem. Represents the
operator for the *global* cell only.
"""
_J = None
"""numpy.ndarray: reverse transform matrix for the problem. Represents the
operator for the *global* cell only. Is the inverse of :data:`_I`.
"""

def reset_cache():
    """Resets the matrix caches for the global cell variable.
    """
    global _L, _Linv, _I, _J
    _L = None
    _Linv = None
    _I = None
    _J = None

def L(v=None, cell=None):
    """Returns the Laplacian operator matrix for the plane wave basis.

    Args:
        v (numpy.ndarray): if None, then return the matrix :math:`\mathbb{L}`,
          else, return :math:`\mathbb{L}\cdot v`.
        cell (pydft.geometry.Cell): that describes the unit cell and
          sampling points for real and reciprocal space.
    """
    from pydft.geometry import cell as gcell
    global _L
    Lij = None
    if _L is not None:
        if cell is None or cell is gcell:
            Lij = _L

    if Lij is None:
        if cell is None:
            cell = gcell
        Lij = -cell.vol*np.diag(cell.G2)
        if cell is gcell:
            _L = Lij

    if v is not None:
        return np.dot(Lij, v)
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
    from pydft.geometry import cell as gcell
    global _Linv
    iLij = None
    if _Linv is not None:
        if cell is None or cell is gcell:
            iLij = _Linv

    if iLij is None:
        if cell is None:
            cell = gcell
        iLij = -1./cell.vol*np.diag(1./cell.G2)
        #Explicitly set the infinity to zero. We compensate for this
        #elsewhere using a constant charge density throughout the
        #cell.
        iLij[0,0] = 0.
        if cell is gcell:
            _Linv = iLij

    if v is not None:
        return np.dot(iLij, v)
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
    from pydft.geometry import cell as gcell
    if cell is None:
        cell = gcell
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
    
def I(v=None, cell=None, fast=True):
    """Returns the forward transform matrix :math:`\mathbb{I}`,
    optionally applied to a vector.

    Args:
        v (numpy.ndarray): if None, then return the matrix :math:`\mathbb{I}`,
          else, return :math:`\mathbb{I}\cdot v`.
        cell (pydft.geometry.Cell): that describes the unit cell and
          sampling points for real and reciprocal space.
    """
    from pydft.geometry import cell as gcell
    if fast:
        if cell is None:
            cell = gcell                

        fft = np.fft.fftn(np.reshape(v, cell.S, order='F'))
        return np.reshape(fft, v.shape, order='F')
    else:
        global _I
        Iap = None
        if _I is not None:
            if cell is None or cell is gcell:
                Iap = _I

        if Iap is None:
            if cell is None:
                cell = gcell

            Iap = _Iap(cell)
            if cell is gcell:
                _I = Iap

        if v is not None:
            return np.dot(Iap, v)
        else:
            return Iap

def J(v=None, cell=None, fast=True):
    """Returns the forward transform matrix :math:`\mathbb{J}`,
    optionally applied to a vector.

    Args:
        v (numpy.ndarray): if None, then return the matrix :math:`\mathbb{J}`,
          else, return :math:`\mathbb{J}\cdot v`.
        cell (pydft.geometry.Cell): that describes the unit cell and
          sampling points for real and reciprocal space.
    """
    from pydft.geometry import cell as gcell
    if fast:
        if cell is None:
            cell = gcell
        fft = np.fft.ifftn(np.reshape(v, cell.S, order='F'))
        return np.reshape(fft, v.shape, order='F')
    else:
        global _J
        Jap = None
        if _J is not None:
            if cell is None or cell is gcell:
                Jap = _J

        if Jap is None:
            if cell is None:
                cell = gcell

            Sprod = np.prod(cell.S)
            Jap = _Iap(cell, True)/Sprod
            if cell is gcell:
                _J = Jap
        if v is not None:
            return np.dot(Jap, v)
        else:
            return Jap
