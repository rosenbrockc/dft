"""Tests the plane wave basis operators.
"""
import pytest
import numpy as np
from pydft.potential import QHO
@pytest.fixture(scope="module", autouse=True)
def adjcube():
    from pydft.geometry import set_geometry
    c = set_geometry(np.diag([6., 6., 6.]), np.array([6, 6, 4]))
    return c

def W4(cell):
    from numpy.matlib import randn
    np.random.seed(2004)
    Ns=4
    return np.array(randn(np.prod(cell.S), Ns) + 1j*randn(np.prod(cell.S), Ns))    

def test_psi(adjcube):
    """Tests retrieval of the wave functions and eigenvalues.
    """
    from pydft.bases.fourier import psi, O, H
    cell = adjcube
    V = QHO(cell)
    W = W4(cell)
    Ns = W.shape[1]
    Psi, epsilon = psi(V, W, cell, forceR=False)

    #Make sure that the eigenvalues are real.
    assert np.sum(np.imag(epsilon)) < 1e-13
    
    checkI = np.dot(Psi.conjugate().T, O(Psi, cell))
    assert abs(np.sum(np.diag(checkI))-Ns) < 1e-13 # Should be the identity
    assert np.abs(np.sum(checkI)-Ns) < 1e-13
    
    checkD = np.dot(Psi.conjugate().T, H(V, Psi, cell))
    diagsum = np.sum(np.diag(checkD))
    assert np.abs(np.sum(checkD)-diagsum) < 1e-12 # Should be diagonal

    # Should match the diagonal elements of previous matrix
    assert np.allclose(np.diag(checkD), epsilon)

def test_Y(adjcube):
    """Tests the wave function normalization routine.
    """
    from pydft.bases.fourier import O, Y
    cell = adjcube
    W = W4(cell)
    Ns = W.shape[1]
    WN = Y(W, cell)

    #Chop off the machine precision issues.
    check = np.dot(WN.conjugate().T, O(WN, cell))
    assert abs(np.sum(np.diag(check))-Ns) < 1e-13
    assert np.abs(np.sum(check)-Ns) < 1e-13

def test_gradE(adjcube):
    """Tests the gradient of `E` using finite difference methods.
    """
    from pydft.bases.fourier import gradE, E
    from numpy.matlib import randn
    cell = adjcube
    V = QHO(cell)

    Ns=4
    #He set the random seed; we could do the same, but the
    #implementation is probably different between numpy and matlab:
    #randn('seed', 0.2004)
    W = np.array(randn(np.prod(cell.S), Ns) + 1j*randn(np.prod(cell.S), Ns))

    # Compute intial energy and gradient
    E0 = E(V, W, cell)
    g0 = gradE(V, W, cell)

    # Choose a random direction to explore
    dW = np.array(randn(W.shape) + 1j*randn(W.shape))

    # Explore a range of step sizes decreasing by powers of ten
    steps = np.logspace(np.log10(1e-3), np.log10(1e-7), 8)
    for delta in steps:
        # Directional derivative formula
        dE = 2*np.real(np.trace(np.dot(g0.conjugate().T, delta*dW)))

        # Print ratio of actual change to expected change, along with estimate
        # of the error in this quantity due to rounding
        ratio = abs(1.-(E(V, W+delta*dW, cell)-E0)/dE)
        print(int(np.log10(ratio)), int(np.log10(delta)), ratio)
        assert abs(int(np.log10(ratio)) - int(np.log10(delta))) <= 2

def test_H_herm(adjcube):
    """Tests that `H` is a Hermitian operator.
    """
    from pydft.bases.fourier import H
    from numpy.matlib import randn
    cell = adjcube

    a = np.array(randn(np.prod(cell.S), 1) + 1j*randn(np.prod(cell.S), 1))
    b = np.array(randn(np.prod(cell.S), 1) + 1j*randn(np.prod(cell.S), 1))

    V = QHO(cell)
    LHS = np.dot(a.T.conjugate(), H(V, b, cell)).conjugate()
    RHS = np.dot(b.T.conjugate(), H(V, a, cell))
    assert np.allclose(LHS, RHS)

def test_E_real(adjcube):
    """Tests that the result of the calculation is real.
    """
    from pydft.bases.fourier import E
    from numpy.matlib import randn
    cell = adjcube
    
    #Single columns of random complex data
    W = np.array(randn(np.prod(cell.S), 4) + 1j*randn(np.prod(cell.S), 4))
    #Setup a harmonic oscillator potential
    V = QHO(cell)
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
