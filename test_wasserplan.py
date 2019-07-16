import pytest
import wasserplan
from gerrychain.grid import Grid


@pytest.fixture
def horizontal_4x4():
    """Creates a 4x4 grid with 4 districts in horizontal stripes.

    1111
    2222
    3333
    4444
    """


@pytest.fixture
def vertical_4x4():
    """Creates a 4x4 grid with 4 districts in vertical stripes.

    1234
    1234
    1234
    1234
    """



    pass


def test_node_embedding(horizontal_4x4, vertical_4x4):
    pass


def test_population_embedding(horizontal_4x4, vertical_4x4):
    pass
