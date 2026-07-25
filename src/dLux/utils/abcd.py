"""Pure ABCD matrix constructors and diagnostics."""

from abcdLux.abcd import (
    abcd_angular_magnification,
    abcd_back_focal_length,
    abcd_distance,
    abcd_effective_focal_length,
    abcd_fraunhofer,
    abcd_free_space,
    abcd_front_focal_length,
    abcd_geometric_magnification,
    abcd_lens,
    abcd_mirror,
    abcd_paraxial_power,
    abcd_surface_power,
    abcd_unimodularity,
    compose_abcd,
    is_free_space,
    is_surface,
)

__all__ = [
    "abcd_surface_power",
    "abcd_lens",
    "abcd_mirror",
    "abcd_free_space",
    "abcd_fraunhofer",
    "compose_abcd",
    "is_surface",
    "is_free_space",
    "abcd_effective_focal_length",
    "abcd_paraxial_power",
    "abcd_geometric_magnification",
    "abcd_angular_magnification",
    "abcd_distance",
    "abcd_front_focal_length",
    "abcd_back_focal_length",
    "abcd_unimodularity",
]
