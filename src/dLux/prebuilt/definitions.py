"""Constructor definitions for sampled models and configured systems."""

from abc import abstractmethod
from typing import Any

import zodiax as zdx

__all__ = ["Def"]


class Def(zdx.Module):
    """A factory consumed when constructing a model.

    Subclasses define their construction inputs and return an ordinary model,
    expression or sampled array from ``__call__``. For example, a basis definition
    may return sampled modes on a supplied grid or a dynamic expansion that uses
    runtime coordinates. The returned model owns numerical evaluation; definitions
    are never executed as optical layers or resolved into arrays during modelling.
    """

    @abstractmethod
    def __call__(self, **context) -> Any:
        """Construct the model described by this definition."""
