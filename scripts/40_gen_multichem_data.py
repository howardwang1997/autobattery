"""Generate multi-chemistry training data for the Battery Foundation Model.

Chemistries:
  0: Chen2020 (NMC811/graphite)
  1: Marquis2019 (NCA/graphite)
  2: Prada2013 (LFP/graphite)
  3: Ai2020 (LFP/graphite v2)
  4: OKane2022 (NMC/graphite, degradation physics)
  5: Ramadass2004 (LCO/graphite)

Total: ~13K simulations × ~3.5 C-rates ≈ ~45K sims
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import sys
sys.path.insert(0, ".")

from src.foundation.data.multichem_generator import generate_multichem_data

if __name__ == "__main__":
    output_path = "data/foundation/multichem_train.h5"
    generate_multichem_data(
        output_path=output_path,
        chemistry_ids=[0, 1, 2, 3, 4, 5],
        processes=30,
    )
