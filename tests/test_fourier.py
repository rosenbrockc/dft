"""Tests the plane wave basis operators.
"""
import pytest
import numpy as np
@pytest.fixture(scope="module", autouse=True)
def adjcube():
    from pydft.geometry import set_geometry
    c = set_geometry(np.diag([6., 6., 6.]), np.array([6, 6, 4]))
    return c

def test_E_real(adjcube):
    """Tests that the result of the calculation is real.
    """
    from pydft.bases.fourier import E
    from numpy.matlib import randn
    cell = adjcube
    
    #Single columns of random complex data
    W = np.array(randn(np.prod(cell.S), 4) + 1j*randn(np.prod(cell.S), 4))
    #Setup a harmonic oscillator potential
    from pydft.potential import Potential
    V = Potential(cell, lambda dr: 2*np.linalg.norm(dr, axis=1)**2)
    En = E(V, W, cell, forceR=False)
    
    assert np.imag(En) < 1e-14

def test_IJ(adjcube):
    """Tests the I and J operators."""
    from pydft.bases.fourier import I, J
    #This also tests accessing the geometry via the global variable.
    Sprod = np.prod(adjcube.S)
    for i in range(10):
        v = np.random.random(size=Sprod)
        #Our v is real; but due to round-off problems, there will be
        #tiny imaginary values. Chop them off.
        it = J(I(v))
        if abs(np.max(np.imag(it))) < 1e-14:
            it = np.real(it)
        assert np.allclose(it, v)

def test_LLinv(adjcube):
    """Tests L and its inverse.
    """
    from pydft.bases.fourier import L, Linv
    Sprod = np.prod(adjcube.S)
    for i in range(10):
        v = np.random.random(size=Sprod)
        #Our v is real; but due to round-off problems, there will be
        #tiny imaginary values. Chop them off. We only keep the last
        #N-1 components because the 0 component is NaN.
        it = Linv(L(v))[1:]
        if abs(np.max(np.imag(it))) < 1e-14:
            it = np.real(it)
        assert np.allclose(it, v[1:])

def test_O(adjcube):
    """Tests the overlap matrix definition.
    """
    from pydft.bases.fourier import O
    #Create a random (normally distributed) 10 vector.
    for i in range(10):
        v = np.random.random(size=10)
        out = O(v)
        assert np.allclose(out/np.linalg.det(adjcube.R), v)

def test_IJdag(adjcube):
    """Tests the operator definitions for :math:`I^\dag` and
    :math:`J^\dag`.
    """
    #We are interested in testing the identities for Idag and Jdag
    from pydft.bases.fourier import Idag, Jdag, I, J
    from numpy.matlib import randn
    cell = adjcube
    
    # Single columns of random complex data
    a = np.array(randn(np.prod(cell.S), 1) + 1j*randn(np.prod(cell.S), 1))
    b = np.array(randn(np.prod(cell.S), 1) + 1j*randn(np.prod(cell.S), 1))

    LHS = np.dot(a.T.conjugate(), I(b)).conjugate()
    RHS = np.dot(b.T.conjugate(), Idag(a))
    assert np.allclose(LHS, RHS)

    LHS = np.dot(a.T.conjugate(), J(b)).conjugate()
    RHS = np.dot(b.T.conjugate(), Jdag(a))
    assert np.allclose(LHS, RHS)

def test_IJdag_M(adjcube):
    """Tests operator definitions on matrix-valued inputs.
    """
    #We are interested in testing the identities for Idag and Jdag
    from pydft.bases.fourier import I
    from numpy.matlib import randn
    cell = adjcube
    
    # Single columns of random complex data
    a = np.array(randn(np.prod(cell.S), 3) + 1j*randn(np.prod(cell.S), 3))

    out1 = I(a)
    out2 = np.array([I(a[:,0]), I(a[:,1]), I(a[:,2])]).T
    assert np.allclose(out1, out2)
