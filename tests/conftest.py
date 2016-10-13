import pytest
import numpy as np
@pytest.fixture(scope="session", autouse=True)
def cube(request):
    """Initializes a cube with some default parameters for the
    sampling.

    Returns:
        pydft.geometry.Cell: initialized cell with sides of 6. Bohr.
    """
    from pydft.geometry import set_geometry
    c = set_geometry(np.diag([6., 6., 6.]), np.array([3, 3, 2]))
    return c
