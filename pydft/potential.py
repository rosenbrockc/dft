"""Methods and objects defining the potential for a given cell.
"""
class Potential(object):
    """Represents a potential that may be discretely sampled using
    grids defined on a :class:`pydft.geometry.Cell` object.

    Args:
        cell (pydft.geometry.cell): cell that has information about the
          grid in real space that the potential will be sampled on.
        pot (callable): function defined in terms of `dr` distances of
          mesh points.

    .. todo:: generalize the `pot` callable to have more parameters
      possible.
    """
    def __init__(self, cell, pot):
        self.cell = cell
        self.pot = pot

        self._V = None
        """numpy.ndarray: potential on each of the points defined
        in the cell.
        """
        self._Vdual = None
        """numpy.ndarray: dual of the potential on each of the points
        defined in the cell.
        """
        
    @property
    def V(self):
        """Evaluation of the potential on each of the points defined
        in the cell.
        """
        if self._V is None:
            self._V = self.pot(self.cell.dr)

    def Vdual(self, basis="fourier"):
        """Returns the dual of the potential points evaluated on
        discrete points.

        Args:
            basis (str): one of the *modules* in `pydft.bases` that
              implements the necessary operators.
        """
        if self._Vdual is None:
            from importlib import import_module
            B = import_module("pydft.bases.{}".format(basis))
            O = B.O
            Jdag = B.Jdag
            J = B.J

            self._Vdual = Jdag(O(J(self.V)))       
