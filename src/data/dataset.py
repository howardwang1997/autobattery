import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Union

from .loader import SyntheticDataLoader, ExperimentalDataLoader
from .preprocessor import Preprocessor


class SyntheticDataset(Dataset):
    """
    PyTorch Dataset for synthetic simulation data.

    Each sample is a tuple of:
    - (t, params) as input
    - (voltage) as target
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        n_points: int = 200,
        normalize_params: bool = True,
    ):
        self.loader = SyntheticDataLoader(data_path)
        self.n_points = n_points
        self.preprocessor = Preprocessor()
        self.normalize_params = normalize_params

        self._param_names = self.loader.param_names
        self._param_values = self.loader.get_all_params()

        if normalize_params:
            self.param_mean = self._param_values.mean(axis=0)
            self.param_std = self._param_values.std(axis=0) + 1e-12
        else:
            self.param_mean = np.zeros(len(self._param_names))
            self.param_std = np.ones(len(self._param_names))

    def __len__(self) -> int:
        return self.loader.num_simulations

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sim = self.loader.get_simulation(idx)

        t, v, i = self.preprocessor.resample(
            sim["time"], sim["voltage"], sim["current"], self.n_points
        )
        t_norm = self.preprocessor.normalize_time(t)

        params = np.array([sim["params"][n] for n in self._param_names])
        params_norm = (params - self.param_mean) / self.param_std

        params_expanded = np.tile(params_norm, (self.n_points, 1))

        x = np.concatenate(
            [t_norm[:, None], params_expanded], axis=1
        ).astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(v.astype(np.float32)),
            torch.from_numpy(i.astype(np.float32)),
        )

    @property
    def num_params(self) -> int:
        return len(self._param_names)

    @property
    def param_names(self) -> list[str]:
        return self._param_names


class ExperimentalDataset(Dataset):
    """
    PyTorch Dataset for experimental cycling data.

    Used for inverse problem training (parameter identification).
    Each sample provides observation points (t, V_obs) for a single experiment.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        n_points: int = 200,
        file_format: str = "csv",
    ):
        self.data_dir = Path(data_dir)
        self.n_points = n_points
        self.preprocessor = Preprocessor()
        self.file_format = file_format

        self._files = sorted(
            list(self.data_dir.glob(f"*.{file_format}"))
            + list(self.data_dir.glob("*.npy"))
        )
        if not self._files:
            raise FileNotFoundError(
                f"No data files found in {self.data_dir}"
            )

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict:
        path = self._files[idx]
        loader = ExperimentalDataLoader()

        if path.suffix == ".npy":
            data = loader.load_neware(path)
        else:
            data = loader.load_csv(path)

        prepped = self.preprocessor.prepare_inverse_training(
            data["time"], data["voltage"], data["current"], self.n_points
        )

        return {
            "t_colloc": torch.from_numpy(prepped["t_colloc"]),
            "v_obs": torch.from_numpy(prepped["v_obs"]),
            "i_input": torch.from_numpy(prepped["i_input"]),
            "file_name": path.name,
        }


class CollocationDataset(Dataset):
    """
    Generate random collocation points for PDE residual evaluation.

    Points are sampled uniformly in the spatiotemporal domain:
    - t in [0, 1] (normalized time)
    - x in [0, 1] (normalized through-cell position)
    - r in [0, 1] (normalized radial position in particle)
    """

    def __init__(
        self,
        n_points: int = 10000,
        n_spatial_dims: int = 3,  # t, x, r
        seed: Optional[int] = None,
    ):
        self.n_points = n_points
        self.n_dims = n_spatial_dims
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_points

    def __getitem__(self, idx: int) -> torch.Tensor:
        points = self.rng.uniform(0, 1, size=(1, self.n_dims)).astype(np.float32)
        return torch.from_numpy(points[0])

    def sample_batch(self, batch_size: int) -> torch.Tensor:
        """Sample a batch of collocation points."""
        points = self.rng.uniform(0, 1, size=(batch_size, self.n_dims)).astype(np.float32)
        return torch.from_numpy(points)
