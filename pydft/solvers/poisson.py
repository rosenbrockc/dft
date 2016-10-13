"""Poisson solver to calculate the electron density self-interaction
contribution to the potential in the DFT formalism.
"""
import numpy as np
def phi(basis, n, cell=None, fast=True):
    """Solves the Poisson equation using the given basis and density
    vector.

    Args:
        basis (str): one of the *modules* in `pydft.bases` that
          implements the necessary operators.
        n (numpy.ndarray): density sampled at each of the points in
          real-space.
        cell (pydft.geometry.Cell): custom cell instance to use for the
          calculation.

    Returns:
        tuple: (phi, npts) where phi is the potential evaluated at each of
          the points in the space and npts is the value of the density
          evaluated at each of the points.
    """
    from importlib import import_module
    B = import_module("pydft.bases.{}".format(basis))
    Linv = B.Linv
    O = B.O
    J = B.J
    I = B.I

    from pydft.geometry import cell as gcell
    ocell = None
    if cell is not None:
        from pydft.geometry import set_cell
        ocell = gcell
        set_cell(cell)
    else:
        cell = gcell

    result = I(Linv(-4*np.pi*O(J(n, fast=fast))), fast=fast)
    if abs(np.max(np.imag(result))) < 1e-14:
        result = np.real(result)

    if ocell is not None:
        #Set the cell back to what it was before.
        set_cell(ocell)

    return result
