"""Unit tests for training reproducibility utilities."""

import random

import numpy as np
import torch

from biometric.training.reproducibility import seed_everything


class TestSeedEverything:
    """Tests for seed_everything."""

    def test_same_seed_produces_same_random_values(self) -> None:
        """Same seed yields identical random values across Python, NumPy, and PyTorch."""
        seed_everything(42)
        py1 = random.random()
        np1 = np.random.rand()
        torch1 = torch.rand(3).tolist()

        seed_everything(42)
        py2 = random.random()
        np2 = np.random.rand()
        torch2 = torch.rand(3).tolist()

        assert py1 == py2
        assert np1 == np2
        assert torch1 == torch2

    def test_different_seeds_produce_different_values(self) -> None:
        """Different seeds yield different random values."""
        seed_everything(42)
        py_42 = random.random()
        np_42 = np.random.rand()
        torch_42 = torch.rand(3).tolist()

        seed_everything(43)
        py_43 = random.random()
        np_43 = np.random.rand()
        torch_43 = torch.rand(3).tolist()

        assert py_42 != py_43
        assert np_42 != np_43
        assert torch_42 != torch_43

    def test_cudnn_flags_set(self) -> None:
        """CuDNN deterministic and benchmark flags are set correctly."""
        seed_everything(0)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
