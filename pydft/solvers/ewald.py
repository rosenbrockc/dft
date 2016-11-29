"""Ewald summation solver for 3D periodic cells.
"""
import numpy as np
from pydft.geometry import get_cell
def Eq(basis, n, cell=None, sigma=None):
    """Calculates the approximate total energy due to the nuclei using
    the Arias quick-and-dirty method.

    Args:
        basis (str): one of the *modules* in `pydft.bases` that
          implements the necessary operators.
        n (numpy.ndarray): density sampled at each of the points in
          real-space.
    """
    from importlib import import_module
    B = import_module("pydft.bases.{}".format(basis))
    O = B.O
    J = B.J
    o = get_cell(cell)

    from pydft.solvers import poisson
    phip = poisson.phi(basis, n, o)
    Unum = 0.5*np.real(np.dot(J(phip), O(J(n))))
    
    if sigma is not None:
        if not isinstance(sigma, list):
            sigma =  [sigma]*len(o.Z)
        Uself = np.sum(o.Z**2/(2*np.sqrt(np.pi))*(1./np.array(sigma)))
    else:
        Uself = np.sum(o.Z**2/(2*np.sqrt(np.pi)))

    return Unum-Uself
    
def E(cell=None, alpha=None, R=None, accuracy=1e-2):
    """Returns the total energy due to the nucleii
    electrostatic potential.

    Args:
    alpha (float): width parameter for the `erf` windowing
          function.
        R (float): maximum extent in real-space to consider for the short-ranged
          sum; defaults to one lattice parameter.
        accuracy (float): desired accuracy for the sum.
    """
    o = get_cell(cell)

    from itertools import product
    #First, construct a matrix of all the points likely to be within
    #the error function window.    
    #Exclude the zero point in the list, since it is just the regular
    #point (no lattice vector summation).
    if alpha is None:
        if R is None:
            R = 0.65*o.vol**(1./3)
        p = np.abs(np.log(accuracy))
        K = 2*p/R
        alpha = K/np.sqrt(p)/2

    nmax=int(np.ceil(np.abs(R/np.linalg.norm(np.dot(o.R, [1,1,1])))))+1
    ni = list(range(nmax)) + list(range(-nmax+1, 0))
    npts = np.array(list(product(ni, repeat=3)))[1:]
    n = np.dot(o.R, npts.T).T

    kmax = int(np.ceil(np.abs(K/np.linalg.norm(np.dot(o.K, [1,1,1])))*2*np.pi))+1
    ki = list(range(kmax)) + list(range(-kmax+1, 0))
    kpts  = np.array(list(product(ki, repeat=3)))[1:]
    k = np.dot(o.K, kpts.T).T

    #First, we calculate the short-ranged contributions for the sum
    #that converges quickly in real space.
    Fs = 0.
    
    from scipy.special import erfc            
    for i in range(o.X.shape[0]):
        for j in range(o.X.shape[0]):
            rij = o.X[i,:] - o.X[j,:]
            if i != j:
                #Handle the atom in the central cell explicitly if it is
                #not on the point we are actually looking.
                absr = np.linalg.norm(rij)
                Fs += o.Z[i]*o.Z[j]*erfc(alpha*absr)/absr
            nr = np.linalg.norm(n + rij, axis=1)
            Fs += o.Z[i]*o.Z[j]*np.sum(erfc(alpha*nr)/nr)
    
    #Next, compute the long range sum. The Fourier transform using the
    #Gaussian charge trick (erf window) has already been calculated,
    #so we can just use it directly.
    Fl = 0.
    k2 = np.linalg.norm(k, axis=1)
    absk = np.exp(-(np.pi*k2/alpha)**2)

    for i in range(o.X.shape[0]):
        for j in range(o.X.shape[0]):
            rij = o.X[i,:] - o.X[j,:]
            ekr = np.exp(2*np.pi*1j*np.dot(k, rij))
            Fl += o.Z[i]*o.Z[j]*np.sum(absk*ekr/k2**2)

    #Round off any random complex pieces that showed up.
    coeff = 1./(2*np.pi*o.vol)
    return Fs/2. + np.real(Fl)*coeff - alpha/(2*np.sqrt(np.pi))*np.dot(o.Z, o.Z)
