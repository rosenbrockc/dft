"""Test steepest descent algorithms for minimizing the energy.
"""
import pytest
import numpy as np
from pydft.potential import QHO

@pytest.fixture(scope="module", autouse=True)
def adjcube():
    from pydft.geometry import set_geometry
    c = set_geometry(np.diag([20., 25., 30.]), np.array([6, 6, 6]),
                     X=[[0,0,0],[1.75,0,0]])
    return c

def W4(cell):
    """Returns a fixed random-seed initial guess wave function.
    """
    from numpy.matlib import randn
    Ns=4
    np.random.seed(2004)
    return np.array(randn(np.prod(cell.S), Ns) + 1j*randn(np.prod(cell.S), Ns))

def test_complete(adjcube):
    """Tests the complete solution of the code.
    """
    from pydft.solvers.sd import simple
    cell = adjcube
    W = W4(cell)

    # Converge using steepest descents
    V = QHO(cell)
    W, E = simple(V, W, cell, 400)

    # Extract and display final results
    from pydft.bases.fourier import psi, I
    Psi, epsilon = psi(V, W, cell)

    for st in range(W.shape[1]):
        print("=== State # {0:d}, Energy = {1:.4f} ===".format(st+1, epsilon[st]))
        dat = np.abs(I(Psi[:,st]))**2
    
def test_simple(adjcube):
    """Test the simple steepest descent algorithm.
    """
    from pydft.solvers.sd import simple
    cell = adjcube
    V = QHO(cell)
    W = W4(cell)
    
    WN, E, Elist = simple(V, W, cell, 20, Elist=True)
    assert np.allclose(Elist, sorted(Elist, reverse=True))

    #Now do the full descent to see if we get close to the correct
    #answer.
    WN, E = simple(V, W, cell, 250)
    assert abs(E-273.) < 1
