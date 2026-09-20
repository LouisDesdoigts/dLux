"""Optical unit helpers backed by Zodiax's unit catalogue.

Grid and propagation calculations use explicit SI factors, independently of the
unit settings used by Zodiax expressions.
"""

import equinox as eqx
import zodiax as zdx

__all__ = [
    "Unitful",
    "canonical_unit",
    "unit_dimension",
    "unit_factor",
    "unit_factor_to_rad",
    "convert",
]


# Zodiax supplies the spelling, prefixes and conversion factor. dLux only narrows
# its dimensions to those used by optical coordinates and photon counts.
_DIMENSIONS = {
    ("angular", 1): "angle",
    ("cartesian", 1): "length",
    ("cartesian", -1): "spatial_frequency",
    ("photon", 1): "photon",
}


def _unit_info(unit, dimension=None, name="unit"):
    """Resolve a catalogue entry and check its optical dimension."""
    if isinstance(unit, zdx.Unit):
        unit = unit.unit
    info = zdx.unit_info(unit)
    physical_dimension = _DIMENSIONS.get((info.category, info.power))
    if physical_dimension is None:
        raise ValueError(f"Unsupported {name} {unit!r} for optical calculations.")
    if dimension is not None and physical_dimension != dimension:
        raise ValueError(
            f"{name} must have dimension {dimension!r}; received {unit!r} "
            f"({physical_dimension})."
        )
    return info


def unit_dimension(unit: str | zdx.Unit) -> str:
    """Return angle, length, spatial_frequency, or photon for a supported unit.

    Takes a unit name or Zodiax Unit; unsupported dimensions raise ValueError.
    """
    info = _unit_info(unit)
    return _DIMENSIONS[info.category, info.power]


def canonical_unit(unit: str, *, dimension=None, name="unit") -> str:
    """Return the canonical spelling of an optical unit.

    ``dimension`` optionally requires angle, length, spatial_frequency, or photon.
    ``name`` labels the input in errors for unknown or incompatible units.
    """
    return _unit_info(unit, dimension, name).unit


def unit_factor_to_rad(unit: str):
    """Return the factor satisfying ``value_in_unit * factor = value_in_radians``."""
    return unit_factor(unit, dimension="angle")


def unit_factor(unit: str | None, *, dimension=None, name="unit"):
    """Return the multiplier from ``unit`` to metres, radians, or photons.

    Reciprocal lengths convert to inverse metres. Optional ``dimension`` and
    ``name`` follow ``canonical_unit``; global Zodiax settings have no effect.
    None leaves values unchanged and returns one.
    """
    if unit is None:
        return 1.0
    return _unit_info(unit, dimension, name).factor


def convert(
    value,
    unit_in: str,
    unit_out: str,
    scale: float = 1.0,
    *,
    dimension=None,
    name="unit",
):
    """Convert a scalar or array from ``unit_in`` to ``scale * unit_out``.

    Zodiax performs the conversion and rejects incompatible units. Optional
    ``dimension`` and ``name`` follow ``canonical_unit``. Array shape is preserved
    except for ordinary broadcasting with ``scale``.
    """
    source = canonical_unit(unit_in, dimension=dimension, name=name)
    target = canonical_unit(unit_out, dimension=dimension, name=name)
    return zdx.convert(value, source, target) / scale


class Unitful(zdx.Module):
    """A model with a declared unit and shared constructor validation.

    Concrete classes assign ``self.unit = self.validate_unit(unit)`` and state
    which inputs or outputs use that unit. Numerical calculations apply any
    required conversion explicitly; this base never rescales model leaves.
    """

    unit: str | None = eqx.field(kw_only=True, default=None)

    @staticmethod
    def validate_unit(unit, *, dimension=None, name="unit") -> str | None:
        """Return a canonical name, or None when no unit is declared."""
        if unit is None:
            return None
        if not isinstance(unit, str):
            raise TypeError(f"{name} must be a string or None.")
        return canonical_unit(unit, dimension=dimension, name=name)
