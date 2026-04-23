import torch
import numpy as np
from pathlib import Path
from typing import Optional


def load_mlff_calculator(
    model_type: str = "mace",
    model_size: str = "medium",
    device: str = "cuda",
):
    """
    Load a pretrained ML force field as an ASE calculator.

    Supported models:
    - mace: MACE-MP-0 (MIT license, good for materials)
    - uma: FAIR Chemistry UMA (MIT license, universal)

    Returns an ASE calculator object.
    """
    if model_type == "mace":
        from mace.calculators import mace_mp
        calc = mace_mp(
            model=model_size,
            dispersion=False,
            default_dtype="float32",
            device=device,
        )
        return calc
    elif model_type == "uma":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        predictor = pretrained_mlip.get_predict_unit(
            f"uma-s-1p2", device=device
        )
        calc = FAIRChemCalculator(predictor, task_name="omat")
        return calc
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_diffusion_coefficient(
    trajectory_file: str,
    species_idx: list[int],
    dt: float,
    temperature: float = 300.0,
    dimensions: int = 3,
):
    """
    Compute diffusion coefficient from MD trajectory via MSD analysis.

    D = MSD / (2 * d * t)

    Args:
        trajectory_file: path to ASE trajectory (.traj) or XYZ file
        species_idx: atom indices for the diffusing species
        dt: time step in seconds
        temperature: temperature in K
        dimensions: spatial dimensions (3 for bulk)

    Returns:
        dict with D, MSD, and time arrays
    """
    from ase.io import read

    traj = read(trajectory_file, index=":")

    positions = np.array([atoms.get_positions()[species_idx] for atoms in traj])
    n_frames, n_atoms, _ = positions.shape

    positions_unwrapped = np.cumsum(
        np.diff(positions, axis=0), axis=0
    )
    positions_unwrapped = np.concatenate(
        [positions[0:1], positions[0] + positions_unwrapped], axis=0
    )

    msd = np.zeros(n_frames)
    for lag in range(1, n_frames):
        displacements = positions_unwrapped[lag:] - positions_unwrapped[:-lag]
        sq_disp = np.sum(displacements ** 2, axis=-1)
        msd[lag] = np.mean(sq_disp)

    time = np.arange(n_frames) * dt

    linear_start = n_frames // 4
    linear_end = n_frames * 3 // 4
    slope, _ = np.polyfit(
        time[linear_start:linear_end],
        msd[linear_start:linear_end],
        1,
    )
    D = slope / (2 * dimensions)

    return {
        "D": D,
        "D_units": "m^2/s",
        "msd": msd,
        "time": time,
        "temperature": temperature,
    }


def compute_ocv_from_energies(
    energies: dict[float, float],
    metal_energy: float,
    n_electrons: int = 1,
):
    """
    Compute OCV curve from DFT/MLFF energies at different Na/Li compositions.

    OCV(x) = -[E(Na_{x+dx}MO2) - E(Na_xMO2) - dx * E(Na)] / (dx * n * F)

    Args:
        energies: dict mapping composition x to total energy (eV/formula unit)
        metal_energy: energy per atom of the alkali metal (eV)
        n_electrons: electrons transferred per formula unit

    Returns:
        dict mapping composition x to voltage (V)
    """
    F = 96485.3329 / 1.602e-19  # convert to eV/V

    compositions = sorted(energies.keys())
    ocv = {}

    for i in range(len(compositions) - 1):
        x1, x2 = compositions[i], compositions[i + 1]
        dx = x2 - x1
        voltage = -(energies[x2] - energies[x1] - dx * metal_energy) / (dx * n_electrons)
        ocv[(x1 + x2) / 2] = voltage

    return ocv
