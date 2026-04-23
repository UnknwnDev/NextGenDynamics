"""Lightweight debug-plot registry.

Any code path (reward, observation, command, …) can call
``env.debug_plot.image()`` or ``env.debug_plot.scatter()`` to register
data for visualisation.  play.py reads the registry and renders
everything with matplotlib when ``--debug_plot`` is passed.

Only store **single-env** tensors (first env) to keep things lean.
"""

from __future__ import annotations

import torch


class DebugPlotRegistry:

    def __init__(self) -> None:
        self._plots: dict[str, tuple] = {}
        self.enabled = False

    # -- registration API --------------------------------------------------

    def image(self, name: str, data: torch.Tensor) -> None:
        """Register a 2-D image (H, W) for ``imshow``."""
        if not self.enabled:
            return
        self._plots[name] = ("image", data)

    def scatter(self, name: str, xy: torch.Tensor, c: torch.Tensor) -> None:
        """Register a scatter plot.  *xy*: (P, 2), *c*: (P,)."""
        if not self.enabled:
            return
        self._plots[name] = ("scatter", xy, c)

    # -- read API ----------------------------------------------------------

    @property
    def plots(self) -> dict[str, tuple]:
        return self._plots
