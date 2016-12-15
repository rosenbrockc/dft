"""Test steepest descent algorithms for minimizing the energy.
"""
import pytest
import numpy as np
from pydft.potential import QHO

@pytest.fixture(scope="module", autouse=True)
def adjcube():
    from pydft.geometry import set_geometry
    c = set_geometry(np.diag([20., 25., 30.]), np.array([6, 6, 6]))
    return c

def test_simple(adjcube):
    """Test the simple steepest descent algorithm.
    """
    from pydft.solvers.sd import simple
    from numpy.matlib import randn
    cell = adjcube
    V = QHO(cell)

    Ns=4
    np.random.seed(2004)
    W = np.array(randn(np.prod(cell.S), Ns) + 1j*randn(np.prod(cell.S), Ns))

    WN, E, Elist = simple(V, W, cell, 20, Elist=True)
    assert np.allclose(Elist, sorted(Elist, reverse=True))

    #Now do the full descent to see if we get close to the correct
    #answer.
    WN, E = simple(V, W, cell, 250)
    assert abs(E-203.) < 1
