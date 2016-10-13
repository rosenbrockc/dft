"""Tests the poisson solver on a simple problem with an analytic
solution.
"""
import pytest
import numpy as np
def test_gaussians():
    """Tests the double gaussian density.
    """
    from pydft.solvers.poisson import phi
    from pydft.geometry import set_geometry
    cube = set_geometry(np.diag([6., 6., 6.]), np.array([20,15,15]))
    center = np.sum(cube.R, axis=1)/2.
    dr = cube.r - center
    rpts = np.linalg.norm(dr, axis=1)
    fast = True
    
    sigma1=0.75
    sigma2=0.50
    pi = np.pi
    g2 = np.exp(-rpts**2/(2*sigma2**2))/np.sqrt(2*pi*sigma2**2)**3
    g1 = np.exp(-rpts**2/(2*sigma1**2))/np.sqrt(2*pi*sigma1**2)**3
    npts = g2-g1

    phip = phi("fourier", npts, fast=fast)
    #Make sure that the box is charge neutral and that the Gaussians are
    #normalized.
    Npts = np.prod(cube.S)
    assert abs(np.sum(g1)*cube.vol/Npts - 1.) < 1e-3
    assert abs(np.sum(g2)*cube.vol/Npts - 1.) < 1e-3
    assert abs(np.sum(npts)*cube.vol/Npts) < 1e-3

    #Calculate the total energies of this system.
    from pydft.bases.fourier import J, O
    Unum = 0.5*np.real(np.dot(J(phip, fast=fast), O(J(npts, fast=fast))))
    Uanal = (((1/sigma1 + 1/sigma2)/2.-np.sqrt(2) /
              np.sqrt(sigma1**2 + sigma2**2))/np.sqrt(np.pi))
    assert abs(Unum - Uanal) < 1e-4
