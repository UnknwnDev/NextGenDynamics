"""Standalone script to (re)generate the custom terrain USD."""

from __future__ import annotations

from pathlib import Path

from .custom_terrain_config import CustomTerrainCfg, TerracedZone
from .custom_terrain_generator import CustomTerrainGenerator


def main() -> None:
    cfg = CustomTerrainCfg()

    generator = CustomTerrainGenerator(cfg)
    path = generator.initialize(export_usd=True, force_export=True)

    hash_path = Path(path).with_suffix(".hash")
    hash_path.write_text(cfg.config_hash())

    print(f"Terrain USD saved to: {path}")


if __name__ == "__main__":
    main()
