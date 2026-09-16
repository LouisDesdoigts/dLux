"""Named sequential composition of compatible callable modules."""

from __future__ import annotations

from collections import OrderedDict
import zodiax as zdx

from .helpers import list2dictionary, insert_layer, remove_layer

__all__ = ["Sequential"]


def _steps(steps, step_type=None):
    """Name components and validate their callable contract recursively."""
    entries = list(steps.items()) if isinstance(steps, dict) else steps
    named = list2dictionary(entries, True)
    for step in named.values():
        if isinstance(step, Sequential):
            _steps(step.steps, step_type)
        elif not isinstance(step, zdx.Module) or not callable(step):
            raise TypeError("Sequential steps must be callable Zodiax Modules.")
        elif step_type is not None and not isinstance(step, step_type):
            raise TypeError(f"Expected {step_type}, got {type(step)}.")
    return named


class Sequential(zdx.Module):
    """Apply named compatible components in insertion order.

    Each component receives the preceding result and explicit caller context.
    Arrays, wavefronts and intensities use the same composition machinery; callers
    select components with compatible input/output contracts. Nested sequences keep
    their names and structure. Shared links are populated at the common owner scope
    before application, while each component owns its local expression resolution.
    Components are stored in ``layers`` so existing structural parameter paths remain
    valid; ``steps`` provides a generic read-only view. An empty sequence is the
    identity.
    """

    layers: OrderedDict

    def __init__(self, layers=(), layer_type=None, *, alias=None):
        """Construct a named sequence.

        Parameters
        ----------
        layers : mapping or sequence
            Callable modules, optionally named with ``(name, module)`` pairs.
        layer_type : type or None
            Optional construction-time restriction on every nested component.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        self.layers = _steps(layers, layer_type)
        self.alias = alias

    @property
    def steps(self):
        """Named components; stored as layers to preserve structural parameter paths."""
        return self.layers

    def apply(self, target, **context):
        """Pass the target through every component, returning the final result."""
        for step in self.populate().steps.values():
            target = step(target, **context)
        return target

    def __call__(self, target, **context):
        """Apply the sequence to an array or compatible field object."""
        return self.apply(target, **context)

    def debug(self, target, **context):
        """Return the final result and named intermediate values, including input."""
        outputs = {"input": target}
        for name, step in self.populate().steps.items():
            target = step(target, **context)
            outputs[name] = target
        return target, outputs

    def insert(self, step, index, step_type=None):
        """Return a copy with a compatible component inserted at ``index``."""
        if step_type is None:
            return self.insert_layer(step, index)
        return self.insert_layer(step, index, step_type)

    def remove(self, key):
        """Return a copy without the named component."""
        return self.remove_layer(key)

    def insert_layer(self, layer, index, layer_type=None):
        """Insert a named or unnamed component, optionally checking its type."""
        layers = insert_layer(self.layers, layer, index, ())
        return self.set(layers=_steps(layers, layer_type))

    def remove_layer(self, key):
        """Return a copy with the named component removed."""
        return self.set(layers=remove_layer(self.layers, key))
