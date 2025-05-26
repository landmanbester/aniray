"""
Pytest configuration and shared fixtures for aniray tests.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import ray


@pytest.fixture(scope="session")
def ray_context():
    """Initialize Ray for distributed tests."""
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def sample_data():
    """Generate sample numerical data for testing."""
    np.random.seed(42)
    return {
        "small_array": np.random.randn(10, 10),
        "large_array": np.random.randn(1000, 100),
        "vector": np.random.randn(100),
    }


@pytest.fixture
def config():
    """Default configuration for tests."""
    from aniray.core.config import Config
    return Config(
        use_gpu=False,
        n_workers=2,
        precision="float64"
    )