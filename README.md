# wheel_legged_robot_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)

## Overview

**wheel_legged_robot_lab** is a specialized RL extension library for wheel-legged robots, based on the excellent [robot_lab](https://github.com/fan-ziqi/robot_lab) framework by Fan Ziqi. This repository focuses specifically on developing reinforcement learning algorithms for wheel-legged robots that combine the efficiency of wheeled locomotion with the versatility of legged locomotion.

The primary robot in this repository is the universal wheel-legged robot designed for versatile locomotion tasks.

> [!NOTE]
> This repository is a specialized fork of [robot_lab](https://github.com/fan-ziqi/robot_lab), focusing specifically on wheel-legged robot implementations. Special thanks to Fan Ziqi for creating the comprehensive robot_lab framework that serves as the foundation for this work.
>
> Discuss in [Github Discussion](https://github.com/fan-ziqi/robot_lab/discussions) or [Discord](http://www.robotsfan.com/dc_robot_lab) for general framework questions.

## Version Dependency

| wheel_legged_robot_lab Version | Isaac Lab Version             | Isaac Sim Version         |
|------------------------------- | ----------------------------- | ------------------------- |
| `main` branch                  | `main` branch                 | Isaac Sim 4.5 / 5.0 / 5.1 |

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this specialized wheel-legged robot repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

  ```bash
  git clone https://github.com/EmberLuo/wheel_legged_robot_lab.git
  cd wheel_legged_robot_lab
  ```

- Using a python interpreter that has Isaac Lab installed, install the library

  ```bash
  python -m pip install -e source/robot_lab
  ```

- Verify that the extension is correctly installed by running the following command to print all the available environments in the extension:

  ```bash
  python scripts/tools/list_envs.py
  ```

<details>

<summary>Set up IDE (Optional, click to expand)</summary>

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.

</details>

<details>

<summary>Setup as Omniverse Extension (Optional, click to expand)</summary>

We provide an example UI extension that will load upon enabling your extension defined in `source/robot_lab/robot_lab/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `wheel_legged_robot_lab/source`
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

</details>

## Docker setup

<details>

<summary>Click to expand</summary>

### Building Isaac Lab Base Image

Currently, we don't have the Docker for Isaac Lab publicly available. Hence, you'd need to build the docker image
for Isaac Lab locally by following the steps [here](https://isaac-sim.github.io/IsaacLab/main/source/deployment/index.html).

Once you have built the base Isaac Lab image, you can check it exists by doing:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED          SIZE
# isaac-lab-base                   latest    28be62af627e   32 minutes ago   18.9GB
```

### Building robot_lab Image

Following above, you can build the docker container for this project. It is called `robot-lab`. However,
you can modify this name inside the [`docker/docker-compose.yaml`](docker/docker-compose.yaml).

```bash
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build robot-lab
```

You can verify the image is built successfully using the same command as earlier:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED             SIZE
# robot-lab                        latest    00b00b647e1b   2 minutes ago       18.9GB
# isaac-lab-base                   latest    892938acb55c   About an hour ago   18.9GB
```

### Running the container

After building, the usual next step is to start the containers associated with your services. You can do this with:

```bash
docker compose --env-file .env.base --file docker-compose.yaml up
```

This will start the services defined in your `docker-compose.yaml` file, including robot-lab.

If you want to run it in detached mode (in the background), use:

```bash
docker compose --env-file .env.base --file docker-compose.yaml up -d
```

### Interacting with a running container

If you want to run commands inside the running container, you can use the `exec` command:

```bash
docker exec --interactive --tty -e DISPLAY=${DISPLAY} wheel-legged-robot-lab /bin/bash
```

### Shutting down the container

When you are done or want to stop the running containers, you can bring down the services:

```bash
docker compose --env-file .env.base --file docker-compose.yaml down
```

This stops and removes the containers, but keeps the images.

</details>

## Try examples

You can use the following commands to run the wheel-legged robot environments:

RSL-RL:

```bash
# Train on flat terrain
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Wheel-Legged-Universal-v0 --headless

# Train on rough terrain
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Rough-Wheel-Legged-Universal-v0 --headless

# Play a trained model on flat terrain
python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Wheel-Legged-Universal-v0

# Play a trained model on rough terrain
python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Rough-Wheel-Legged-Universal-v0
```

> [!NOTE]
> If you want to control a **SINGLE ROBOT** with the keyboard during playback, add `--keyboard` at the end of the play script.
>
> ```
> Key bindings:
> ====================== ========================= ========================
> Command                Key (+ve axis)            Key (-ve axis)
> ====================== ========================= ========================
> Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
> Move along y-axis      Numpad 4 / Arrow Right    Numpad 6 / Arrow Left
> Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
> ====================== ========================= ========================
> ```

* Record video of a trained agent (requires installing `ffmpeg`), add `--video --video_length 200`
* Play/Train with custom number of environments, add `--num_envs <number>`
* Play on specific folder or checkpoint, add `--load_run run_folder_name --checkpoint /PATH/TO/model.pt`
* Resume training from folder or checkpoint, add `--resume --load_run run_folder_name --checkpoint /PATH/TO/model.pt`
* To train with multiple GPUs, use the following command, where --nproc_per_node represents the number of available GPUs:
    ```bash
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Wheel-Legged-Universal-v0 --headless --distributed
    ```

## Special Features for Wheel-Legged Robots

This repository includes specialized control methods for wheel-legged robots, particularly:

- **VMC (Virtual Model Control)**: Implementation of virtual model control for wheel-legged robots that combines wheeled and legged locomotion
- **Adaptive Locomotion**: Algorithms that switch between wheeled and legged modes based on terrain conditions
- **Hybrid Control**: Controllers that leverage both wheel rotation and leg movements for enhanced mobility

The wheel-legged robot implementation uses a unique control architecture where the VMC controller manages leg dynamics while wheel motors provide propulsion.

## Tensorboard

To view tensorboard for your training results, run:

```bash
tensorboard --logdir=logs/rsl_rl/RobotLab-Isaac-Velocity-Flat-Wheel-Legged-Universal-v0
```
or for rough terrain:
```bash
tensorboard --logdir=logs/rsl_rl/RobotLab-Isaac-Velocity-Rough-Wheel-Legged-Universal-v0
```

## Code formatting

A pre-commit template is given to automatically format the code.

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

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

**Note: Replace `<path-to-isaac-lab>` with your own IsaacLab path.**

```json
{
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/source/robot_lab",
        "/<path-to-isaac-lab>/source/isaaclab",
        "/<path-to-isaac-lab>/source/isaaclab_assets",
        "/<path-to-isaac-lab>/source/isaaclab_mimic",
        "/<path-to-isaac-lab>/source/isaaclab_rl",
        "/<path-to-isaac-lab>/source/isaaclab_tasks",
    ]
}
```

### Clean USD Caches

Temporary USD files are generated in `/tmp/IsaacLab/usd_{date}_{time}_{random}` during simulation runs. These files can consume significant disk space and can be cleaned by:

```bash
rm -rf /tmp/IsaacLab/usd_*
```

## Acknowledgements

This project is built upon the excellent [robot_lab](https://github.com/fan-ziqi/robot_lab) framework by Fan Ziqi. Special thanks to the original authors for creating such a comprehensive and extensible framework for robotic reinforcement learning. The VMC control implementation and other specialized features for wheel-legged robots were developed specifically for this repository.

The project also uses some code from the following open-source code repositories:

- [linden713/humanoid_amp](https://github.com/linden713/humanoid_amp)
- [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
