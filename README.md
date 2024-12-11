## Installation

1. **Preparing conda env**

   Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
   ```bash
   # We require python>=3.9 and cmake>=3.14
   conda create -n habitat python=3.9 cmake=3.14.0
   conda activate habitat
   ```

1. **conda install habitat-sim**
   - To install habitat-sim with bullet physics
      ```
      conda install habitat-sim withbullet -c conda-forge -c aihabitat
      ```
      Note, for newer features added after the most recent release, you may need to install `aihabitat-nightly`. See Habitat-Sim's [installation instructions](https://github.com/facebookresearch/habitat-sim#installation) for more details.

1. **pip install habitat-lab stable version**.

      ```bash
      git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
      cd habitat-lab
      pip install -e habitat-lab  # install habitat_lab
      ```
      Please notice that download the "habitat-lab" project under the path "$VLN_ROOT/dependencies/".
1. **Install habitat-baselines**.

    The command above will install only core of Habitat-Lab. To include habitat_baselines along with all additional requirements, use the command below after installing habitat-lab:

      ```bash
      pip install -e habitat-baselines  # install habitat_baselines
      ```

1. **Install pytorch (assuming cuda 11.3)**:
    ```bash
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

1. **Install detectron2**:
    Install the detectron2 based on:
    ```
    https://github.com/facebookresearch/detectron2?tab=readme-ov-file
    ```

1. **Add repository to python path**:
    ```
    export PYTHONPATH=$PYTHONPATH:$VLN_ROOT
    ```

## Datasets
Please use the datasets of HM3D-v0.2-val_split. You should download both Scenes datasets and the task datasets.
[Common task and episode datasets used with Habitat-Lab](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).



## Perception Model

Download the image segmentation model [[URL](https://utexas.box.com/s/sf4prmup4fsiu6taljnt5ht8unev5ikq)] to `$VLN_ROOT/dependencies/mask_rcnn/`.'

## Quick start and quick evaluation
```
python task_evaluation.py
```

For dataset switching(HM3D V1/V2), it can be achieved by setting `habitat_config.habitat.dataset.data_path` in the `$VLN_ROOT/env_tools/data_utils.py`



## Train the RL Policy
```
python train.py
```

For the train scenes setting, add the following code in the `$VLN_ROOT/train.py`(It should be noted that some episodes may have ambiguity, so the interference of ambiguous episodes can be eliminated by specifying categories as follows.):
```
id_dict = {
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00669-DNWbUAJYsPy/DNWbUAJYsPy.basis.glb":["tv_monitor", "bed", "sofa", "chair", "toilet"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00166-RaYrxWt5pR1/RaYrxWt5pR1.basis.glb":["tv_monitor", "toilet", "chair", "plant", "sofa"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00404-QN2dRqwd84J/QN2dRqwd84J.basis.glb":["sofa", "bed", "plant", "tv_monitor", "toilet"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00706-YHmAkqgwe2p/YHmAkqgwe2p.basis.glb":["bed", "toilet", "chair", "sofa"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00324-DoSbsoo4EAg/DoSbsoo4EAg.basis.glb":["bed", "tv_monitor"],

    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00017-oEPjPNSPmzL/oEPjPNSPmzL.basis.glb":["bed", "tv_monitor", "toilet", "sofa", "plant"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00031-Wo6kuutE9i7/Wo6kuutE9i7.basis.glb":["bed", "tv_monitor", "toilet"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00099-226REUyJh2K/226REUyJh2K.basis.glb":["bed", "tv_monitor"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00105-xWvSkKiWQpC/xWvSkKiWQpC.basis.glb":["tv_monitor", "toilet", "sofa"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00250-U3oQjwTuMX8/U3oQjwTuMX8.basis.glb":["bed", "toilet", "sofa", "plant"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00251-wsAYBFtQaL7/wsAYBFtQaL7.basis.glb":["bed", "toilet", "sofa"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00254-YMNvYDhK8mB/YMNvYDhK8mB.basis.glb":["chair", "plant"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00255-NGyoyh91xXJ/NGyoyh91xXJ.basis.glb":["bed", "tv_monitor", "toilet"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00323-yHLr6bvWsVm/yHLr6bvWsVm.basis.glb":["bed", "tv_monitor", "toilet"],
    "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00327-xgLmjqzoAzF/xgLmjqzoAzF.basis.glb":["bed", "toilet", "chair"],

}
selected_episodes = []
for index, temp_episode in enumerate(habitat_env.episodes):
    if(temp_episode.scene_id in id_dict.keys()):
        if(temp_episode.object_category in id_dict[temp_episode.scene_id]):
            selected_episodes.append(temp_episode)
random.shuffle(selected_episodes)
```

## Visualization of Training Process
When `train.by` is executed, the trend of SR and SPL changes in the last 20 episodes in the training set can be directly obtained.

You can execute the following instructions on the terminal to visualize the training curve:
```
 tensorboard --host 127.0.0.1 --logdir log_files_train/your_train_log_file --samples_per_plugin scalars=999999999
```

When `task_evaluation_train_val.py` is executed, you can obtain the performance change curve of the policy on the validation set during the training process.
```
 tensorboard --host 127.0.0.1 --logdir log_files/your_train_val_log_file --samples_per_plugin scalars=999999999
```

When `task_evaluation_train_val_train_data.py` is executed, you can obtain the performance change curve of the policy on the train set during the training process.
```
 tensorboard --host 127.0.0.1 --logdir log_files/your_train_val_train_data_log_file --samples_per_plugin scalars=999999999
```

## Visualization of Episode
The related tools for episode visualization are located in `$VLN_ROOT/vis_tools/`, you can save the execution status of the episode in `*.mp4` format locally, as follows:

0. Set the path to save the video in the `--pre_path` parameter of `$VLN_ROOT/vis_tools/arguments.py`

1. Set the `args.is_vis = False`

2. Add the following code at the beginning of each episode:
```
if(args.is_vis==True):
    video_writer, map_writer = init_mp4(pre_model=args.graph_pre_model,    episode_index=index_in_episodes+1)
    get_top_down_map(habitat_env)
```
It is used to initialize the objects for recording videos and initialize the top-down map.

3. Add the following code after selecting the sub-goal:
```
if(args.is_vis==True):
    achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal, graph_train=False, rl_graph=rl_graph, video_writer=video_writer, map_writer=map_writer)
else:
    achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal)
```

4. Add the following code after executing the work steps in the simulator in `$VLN_ROOT/navigation/sub_goal_reach.py`:
```
if(env_args.is_vis==True):
    save_mp4(video_writer, map_writer, habitat_env, topo_graph, rl_graph, action_node, object_goal)
```

5. After completing the above steps, you will obtain the RGB video and topology change video for each episode in the designated folder for saving videos, which will facilitate debugging of the later code.

## Manual debugging
Just set the `args.is_auto=False`, then you can manually debug the navigation process of the robots in each episode.



## Habitat Tips
0. **Please git clone the habitat-lab commit of "c9e5366".**

1. **env.py(`$VLN_ROOT/dependencies/habitat-lab/habitat-lab/habitat/core/env.py`)**
- Origin
```
with read_write(self._config):
    self._config.simulator.scene_dataset = (
        self.current_episode.scene_dataset_config
    )
```

- Now
```
with read_write(self._config):
    self._config.simulator.scene_dataset = (
        "$VLN_ROOT/dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
    )
```

2. **rgbd_agent.yaml(`$VLN_ROOT/dependencies/habitat-lab/habitat-lab/habitat/config/habitat/simulator/sensor_setups/rgbd_agent.yaml`)**

- Origin
```
- /habitat/simulator/sim_sensors@sim_sensors.depth_sensor: depth_sensor
```
- Now
```
- /habitat/simulator/sim_sensors@sim_sensors.equirect_depth_sensor: equirect_depth_sensor
```

3. **objectnav_hm3d.yaml(`$VLN_ROOT/dependencies/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml`)**
- Origin
```
rgb_sensor:
    width: 640
    height: 480
    hfov: 79
    position: [0, 0.88, 0]
depth_sensor:
    width: 640
    height: 480
    hfov: 79
    min_depth: 0.5
    max_depth: 5.0
    position: [0, 0.88, 0]
```
- Now
```
rgb_sensor:
    width: 512
    height: 512
    hfov: 90
    position: [0, 0.88, 0]
equirect_depth_sensor:
    width: 481
    height: 241
    min_depth: 0.0
    max_depth: 10.0
    position: [0, 0.88, 0]
```

4. **data(`$VLN_ROOT/dependencies/habitat-lab/data/`)**
- Use the data that I provide, including `$VLN_ROOT/dependencies/habitat-lab/data/datasets/` and `$VLN_ROOT/dependencies/habitat-lab/data/scene_datasets/`, and put it under the dictionary of `$VLN_ROOT/dependencies/habitat-lab/data/`.
