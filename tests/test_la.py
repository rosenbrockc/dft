"""Tests the linear algebra routines.
"""
import pytest
import numpy as np

def test_diagprod():
    """Tests diagonal product.
    """
    from pydft.solvers.la import diagprod
    a = np.random.random(10)
    B = np.random.random((10,3))

    model = np.dot(np.diag(a), B)    
    ours = diagprod(a, B)
    
    assert np.allclose(model, ours)

def test_diagouter():
    """Tests diagonal of outer product.
    """
    A = np.random.random((3,4)) + 1j*np.random.random((3,4))
    B = np.random.random((3,4)) + 1j*np.random.random((3,4))

    model = np.diag(np.dot(A, B.T.conjugate()))
    from pydft.solvers.la import diagouter
    ours = diagouter(A, B)
    
    assert np.allclose(model, ours)
