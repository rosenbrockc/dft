"""Tests the geometry initialization routines that construct sampling
matrices, etc.
"""
import pytest
import numpy as np
def test_M(cube):
    """Tests construction of the M matrix.
    """
    result = np.array([[0,0,0],
                       [1,0,0],
                       [2,0,0],
                       [0,1,0],
                       [1,1,0],
                       [2,1,0],
                       [0,2,0],
                       [1,2,0],
                       [2,2,0],
                       [0,0,1],
                       [1,0,1],
                       [2,0,1],
                       [0,1,1],
                       [1,1,1],
                       [2,1,1],
                       [0,2,1],
                       [1,2,1],
                       [2,2,1]], dtype=int)
    assert np.allclose(result, cube.M)

def test_N(cube):
    """Tests construction of the N matrix using the components of the
    M matrix.
    """
    result = np.array([[ 0,  0,  0],
                       [ 1,  0,  0],
                       [-1,  0,  0],
                       [ 0,  1,  0],
                       [ 1,  1,  0],
                       [-1,  1,  0],
                       [ 0, -1,  0],
                       [ 1, -1,  0],
                       [-1, -1,  0],
                       [ 0,  0,  1],
                       [ 1,  0,  1],
                       [-1,  0,  1],
                       [ 0,  1,  1],
                       [ 1,  1,  1],
                       [-1,  1,  1],
                       [ 0, -1,  1],
                       [ 1, -1,  1],
                       [-1, -1,  1]])
    assert np.allclose(result, cube.N)

def test_rs(cube):
    """Tests the generation of the sample points in the unit cell.
    """
    model = np.array([[ 0.,  0.,  0.],
                      [ 2.,  0.,  0.],
                      [ 4.,  0.,  0.],
                      [ 0.,  2.,  0.],
                      [ 2.,  2.,  0.],
                      [ 4.,  2.,  0.],
                      [ 0.,  4.,  0.],
                      [ 2.,  4.,  0.],
                      [ 4.,  4.,  0.],
                      [ 0.,  0.,  3.],
                      [ 2.,  0.,  3.],
                      [ 4.,  0.,  3.],
                      [ 0.,  2.,  3.],
                      [ 2.,  2.,  3.],
                      [ 4.,  2.,  3.],
                      [ 0.,  4.,  3.],
                      [ 2.,  4.,  3.],
                      [ 4.,  4.,  3.]])
    assert np.allclose(cube.r, model)

def test_Gs(cube):
    """Tests the construction of the reciprocal lattice vectors and
    sample points.
    """
    model = np.load("tests/model/G.666.npy")
    assert np.allclose(model, cube.G)
    modelK = np.array([[ 1.04719755,  0.        ,  0.        ],
                       [ 0.        ,  1.04719755,  0.        ],
                       [ 0.        ,  0.        ,  1.04719755]])
    assert np.allclose(modelK, cube.K)
