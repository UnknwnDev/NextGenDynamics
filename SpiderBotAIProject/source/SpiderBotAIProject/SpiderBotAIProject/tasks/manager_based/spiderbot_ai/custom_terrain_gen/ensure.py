# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

from ..paths import CUSTOM_TERRAIN_USD_PATH
from .custom_terrain_config import CustomTerrainCfg
from .custom_terrain_generator import CustomTerrainGenerator


def ensure_custom_terrain_usd(
    *,
    size_x: float,
    size_y: float,
    meter_per_grid: float,
    seed: int,
    force: bool = False,
) -> Path:
    """Ensure the terrain USD exists at the canonical path."""
    usd_path = Path(CUSTOM_TERRAIN_USD_PATH)
    if usd_path.exists() and not force:
        return usd_path

    cfg = CustomTerrainCfg(
        size=(float(size_x), float(size_y)),
        meter_per_grid=float(meter_per_grid),
        seed=int(seed),
        usd_path=CUSTOM_TERRAIN_USD_PATH,
    )
    generator = CustomTerrainGenerator(cfg)
    return generator.initialize(export_usd=True, force_export=force)
