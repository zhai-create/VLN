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
