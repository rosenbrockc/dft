"""Tests the full and quick-and-dirty ewald summation methods.
"""
import pytest
import numpy as np
def test_quick():
    """Tests the Arias method for Ewald summation.
    """
    from pydft.geometry import set_geometry
    cube = set_geometry(np.diag([6., 6., 6.]), np.array([20, 15, 15]),
                        [[0., 0., 0.], [1.75, 0., 0.]])
    center = np.sum(cube.R, axis=1)/2.
    dr = cube.r - center
    rpts = np.linalg.norm(dr, axis=1)
    
    sigma = 0.25
    pi = np.pi
    g = np.exp(-rpts**2/(2*sigma**2))/np.sqrt(2*pi*sigma**2)**3

    from pydft.bases.fourier import I, J
    n = I(J(g)*cube.Sf)
    npts = np.real(n)

    Npts = np.prod(cube.S)
    assert abs(np.sum(g)*cube.vol/Npts - 1.) < 2e-3
    #We have a total of two nucleii in the cell, each with a charge of
    #1, so we expect total charge to be 2. I had to play with the
    #cutoff because they weren't exactly below 1e-3.
    assert abs(2.-np.sum(npts)*cube.vol/Npts) < 4e-3

    from pydft.solvers.ewald import Eq
    Et = Eq("fourier", npts, sigma=sigma)
    #print("Quick and Dirty: {0:.5e}".format(Et))

def test_full():
    from pydft.geometry import Cell
    cube = Cell(np.diag([6., 6., 6.]), [20,15,15], [[0,0,0],[1.75,0,0]])
    cube2 = Cell(np.diag([10., 10., 10.]), [20,15,15], [[0,0,0],[4.00,0,0]])

    from pydft.solvers import ewald
    E1 = ewald.E(cube, R=3.85, accuracy=1e-3)
    assert abs(E1+0.333) < 1e-3
    assert abs(ewald.E(cube2, R=12.4, accuracy=1e-4)-0.9<1e-2)
