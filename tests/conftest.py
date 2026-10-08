import numpy as np
import pytest
from obb_synthetic import make_landmarks, make_wing_image


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def landmarks(rng):
    return make_landmarks(rng, angle_deg=7.0)


@pytest.fixture
def wing_image():
    return make_wing_image()
