#!/usr/bin/env python3
"""
Load and visualize Neware experimental data.

Usage:
    python scripts/load_experimental.py
"""

import numpy as np
import matplotlib.pyplot as plt
from src.data.loader import ExperimentalDataLoader


def extract_cycle(data: dict, cycle_num: int, step_type: str = "discharge") -> dict:
    """
    Extract V(t) curve for a specific cycle and step type.
    
    Args:
        data: loaded data dict from ExperimentalDataLoader
        cycle_num: cycle number to extract
        step_type: "charge" or "discharge"
    
    Returns:
        dict with normalized time and voltage arrays
    """
    cycles = data["cycle"]
    current = data["current"]
    
    if step_type == "discharge":
        mask = (cycles == cycle_num) & (current < -0.01)
    else:
        mask = (cycles == cycle_num) & (current > 0.01)
    
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    
    t = data["time"][indices]
    v = data["voltage"][indices]
    
    # Normalize time to [0, 1]
    t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-6)
    
    return {"time": t, "time_norm": t_norm, "voltage": v, "current": current[indices]}


def plot_cycles(data: dict, num_cycles: int = 5, step_type: str = "discharge"):
    """Plot several discharge curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    max_cycle = int(data["cycle"].max())
    cycle_interval = max(max_cycle // num_cycles, 1)
    
    cycle_list = list(range(1, max_cycle + 1, cycle_interval))[:num_cycles]
    colors = plt.cm.viridis(np.linspace(0, 1, len(cycle_list)))
    
    for i, cycle in enumerate(cycle_list):
        cycle_data = extract_cycle(data, cycle, step_type)
        if cycle_data is None:
            continue
        
        t = cycle_data["time_norm"]
        v = cycle_data["voltage"]
        
        ax.plot(t * 100, v, color=colors[i], label=f"Cycle {cycle}", linewidth=1.5)
    
    ax.set_xlabel("Time (%)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"Experimental {step_type.capitalize()} Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/figures/experimental_cycles.png", dpi=150)
    print(f"Saved to outputs/figures/experimental_cycles.png")
    plt.close()


def main():
    # Load data
    loader = ExperimentalDataLoader()
    data_path = "/root/data/raw/exapmles/NEWAREA_1205XXL01006.xlsx"
    
    print(f"Loading {data_path}...")
    data = loader.load_neware_xlsx(data_path)
    
    print(f"Loaded {len(data['time'])} data points")
    print(f"  Time range: {data['time'][0]:.0f} - {data['time'][-1]:.0f} s ({data['time'][-1]/3600:.1f} hours)")
    print(f"  Voltage: {data['voltage'].min():.3f} - {data['voltage'].max():.3f} V")
    print(f"  Current: {data['current'].min():.3f} - {data['current'].max():.3f} A")
    print(f"  Cycles: {int(data['cycle'].max())}")
    
    # Extract first discharge curve
    cycle_1 = extract_cycle(data, 1, "discharge")
    if cycle_1 is not None:
        print(f"\nFirst discharge curve (Cycle 1):")
        print(f"  Points: {len(cycle_1['voltage'])}")
        print(f"  Voltage range: {cycle_1['voltage'].min():.3f} - {cycle_1['voltage'].max():.3f} V")
    
    # Plot
    import os
    os.makedirs("outputs/figures", exist_ok=True)
    plot_cycles(data, num_cycles=5, step_type="discharge")
    plot_cycles(data, num_cycles=5, step_type="charge")


if __name__ == "__main__":
    main()