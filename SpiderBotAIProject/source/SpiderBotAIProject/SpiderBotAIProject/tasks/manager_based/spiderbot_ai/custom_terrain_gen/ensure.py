# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

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
    """Ensure the terrain USD exists at the canonical path.

    Regenerates when the config hash changes, even if the USD file already exists.
    """
    usd_path = Path(CUSTOM_TERRAIN_USD_PATH)
    hash_path = usd_path.with_suffix(".hash")

    cfg = CustomTerrainCfg(
        size=(float(size_x), float(size_y)),
        meter_per_grid=float(meter_per_grid),
        seed=int(seed),
        usd_path=CUSTOM_TERRAIN_USD_PATH,
    )
    current_hash = cfg.config_hash()

    if not force and usd_path.exists() and hash_path.exists():
        if hash_path.read_text().strip() == current_hash:
            return usd_path

    log.info("Generating custom terrain USD ...")
    generator = CustomTerrainGenerator(cfg)
    result = generator.initialize(export_usd=True, force_export=True)
    hash_path.write_text(current_hash)
    log.info("Custom terrain USD saved to: %s", result)
    return result
