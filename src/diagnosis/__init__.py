"""Mechanistic-degradation diagnosis tooling.

This package modularises the differential-voltage signature method that
``scripts/28_degradation_modes.py`` originally implemented as a one-off
prototype, and adds the supporting plumbing (uncertainty quantification,
landmark/morphology constraints, cross-validation utilities) needed for
the publication roadmap (``docs/plan_publication_roadmap.md``).
"""

from .dv_signature import (
    SignatureLibrary,
    DegradationDiagnosis,
    build_signature_library,
    fit_cycle_modes,
    DEFAULT_SIGNATURE_NAMES_LMB,
)

__all__ = [
    "SignatureLibrary",
    "DegradationDiagnosis",
    "build_signature_library",
    "fit_cycle_modes",
    "DEFAULT_SIGNATURE_NAMES_LMB",
]
