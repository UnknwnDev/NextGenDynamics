# SpiderBotAIProject (Isaac Lab Extension)

## Overview

This repository contains the `SpiderBotAIProject` Isaac Lab extension and task implementations.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/SpiderBotAIProject
    ```

- Train (skrlcustom-only):

    ```bash
    python scripts/skrlcustom/train.py --task=SpiderBotAIProject-v0
    ```

- Note: On first run, the script generates `custom_terrain.usd` under `source/SpiderBotAIProject/SpiderBotAIProject/assets/terrains/`.
  This requires `opensimplex` in your Isaac Sim python environment.

- Play (skrlcustom-only):

    ```bash
    python scripts/skrlcustom/play.py --task=SpiderBotAIProject-v0 --checkpoint <path-to-checkpoint>
    ```

- Debug visualization (Command markers):

    ```bash
    python scripts/skrlcustom/play.py --task=SpiderBotAIProject-v0 --debug_vis
    ```

## Docker (Headless Training)

This project can be trained headlessly using the official Isaac Lab prebuilt container image.

```bash
docker compose -f docker/compose.yaml up --build
```

Notes:
- The compose file runs `scripts/skrlcustom/train.py` with `--headless`.
- Training artifacts are saved to `./logs` and `./outputs` (mounted as volumes).
- To override the env count, set `NUM_ENVS` (e.g. `NUM_ENVS=64 docker compose -f docker/compose.yaml up --build`).

### Docker (2GPU DDP)

If you want to run in 2 GPU env, run this:

```bash
docker compose -f docker/compose.yaml -f docker/compose.ddp.yaml up --build
```

Notes:
- `--num_envs` is interpreted as per gpu (total envs = `--num_envs * WORLD_SIZE`).

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/SpiderBotAIProject/SpiderBotAIProject/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/SpiderBotAIProject"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
