"""Extra linear algebra methods.
"""
import numpy as np

def matprod(M, v):
    """Produces the product of the matrix `M` with `v`. If `v` is also
    a matrix, then the matrix is multiplied by *each column* in `v` to
    produce a new matrix of the same shape.

    Args:
    M (numpy.ndarray): matrix to multiple from the left.
    v (numpy.ndarray): can be a vector or matrix; if a matrix, then
      `M` is multiplied by each column separately to produce the
      result.
    """
    if hasattr(M, "__call__"):
        #This is actually a function that is being called on the
        #vector, not a matrix for standard multiplication.
        if len(v.shape) == 2 and v.shape[1] != 1:
            result = np.zeros(v.shape, dtype=v.dtype)
            for i in range(v.shape[1]):
                result[:,i] = M(v[:,i])
        else:
            result = M(v)      
    else:
        if len(v.shape) == 2 and v.shape[1] != 1:
            result = np.zeros(v.shape, dtype=v.dtype)
            for i in range(v.shape[1]):
                result[:,i] = np.dot(M, v[:,i])
        else:
            result = np.dot(M, v)    

    return result

def diagouter(A, B):
    """Takes the diagonal values of the outer product matrix between
    `A` and `B`.

    Args:
        A (numpy.ndarray): with shape `(n, m)`.
        B (numpy.ndarray): with shape `(n, m)`.
    """
    #We are after c_i = \sum_n A_{i,n}B^*_{i,n}.
    return np.sum(A*B.conjugate(), axis=1)
