"""Tests the plane wave basis operators.
"""
import pytest
import numpy as np
@pytest.fixture(scope="module", autouse=True)
def adjcube():
    from pydft.geometry import set_geometry
    c = set_geometry(np.diag([6., 6., 6.]), np.array([6, 6, 4]))
    return c

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
