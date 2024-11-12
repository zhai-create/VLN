from dependencies import *
import habitat
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)


HM3D_CONFIG_PATH = "./dependencies/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"

# AgentPositionSensorConfig revelent

from dataclasses import dataclass
from habitat.config.default_structured_configs import LabSensorConfig
from omegaconf import MISSING
from register_new_sensors_and_measures import AgentPositionSensor


@dataclass
class AgentPositionSensorConfig(LabSensorConfig):
    type: str = "my_supercool_sensor"
    answer_to_life: int = MISSING

def hm3d_config(path:str=HM3D_CONFIG_PATH,stage:str='val',episodes=200, max_steps=500):
    habitat_config = habitat.get_config(path)

    # print("\n\n\n\n\n")
    # print(habitat_config.habitat.simulator.agents.main_agent.sim_sensors)
    # print("\n\n\n\n\n")

    with read_write(habitat_config):
        habitat_config.habitat.task.lab_sensors[
            "agent_position_sensor"
        ] = AgentPositionSensorConfig(answer_to_life=5)

        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "./dependencies/habitat-lab/data/scene_datasets"
        habitat_config.habitat.dataset.data_path = "./dependencies/habitat-lab/data/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    # visibility_dist=5.0,
                    visibility_dist=10.0,
                    fov=90,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        # habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        # habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.equirect_depth_sensor.max_depth=10.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.equirect_depth_sensor.normalize_depth=True
        habitat_config.habitat.task.measurements.success.success_distance = 1.0
        habitat_config.habitat.environment.max_episode_steps = max_steps
    return habitat_config
    

