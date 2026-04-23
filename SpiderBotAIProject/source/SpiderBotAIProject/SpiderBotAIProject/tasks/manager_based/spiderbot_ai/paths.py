# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    """Return the SpiderBotAIProject python package root directory."""
    # This file lives at:
    # SpiderBotAIProject/tasks/manager_based/spiderbot_ai/paths.py
    return Path(__file__).resolve().parents[3]


ASSETS_DIR = package_root() / "assets"
ROBOTS_DIR = ASSETS_DIR / "spider"
TERRAINS_DIR = ASSETS_DIR / "terrains"


SPIDER_USD_PATH = ROBOTS_DIR / "spider.usd"
CUSTOM_TERRAIN_USD_PATH = TERRAINS_DIR / "custom_terrain.usd"
