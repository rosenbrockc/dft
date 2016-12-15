"""Steepest descent solvers for minimizing the energy.
"""
import numpy as np
def simple(V, W, cell=None, Nit=10, alpha=3e-5, basis="fourier", Elist=False):
    """Simple steepest descent minimization.

    Args:
        V (pydft.potential.Potential): describing the potential for the
          particles.
        W (numpy.ndarray): wave function sample points.
        cell (pydft.geometry.Cell): describing the unit cell and sampling
          points.
        Nit (int): maximum number of iterations to perform.
        alpha (float): step size for the descent.
        basis (str): one of the *modules* in `pydft.bases` that
          implements the necessary operators.
        Elist (bool): when True, return a list of the energy at each step
          of the algorithm (used for unit testing).

    Returns:
        tuple: of `(W, E)` where `W` is a :class:`numpy.ndarray` of the
        new wave function column vectors and `E` is the energy of the
        system. 
    """
    from importlib import import_module
    from pydft.geometry import get_cell
    from pydft import msg
    
    B = import_module("pydft.bases.{}".format(basis))
    Y = B.Y
    gradE = B.gradE
    E = B.E
    
    #Get the normalized wave function.
    cell = get_cell(cell)
    WN = Y(W, cell)
    if Elist:
        Es = []
        
    for i in range(Nit):
        WN = WN - alpha*gradE(V, WN, cell)
        En = float(E(V, WN, cell))
        msg.info("E = {0:.10e}".format(En), 0)
        if Elist:
            Es.append(En)

    if Elist:
        return (WN, En, np.array(Es))
    else:
        return (WN, En)
