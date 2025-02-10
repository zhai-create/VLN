from dependencies import *
import habitat
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
import habitat_sim
import quaternion

HM3D_CONFIG_PATH = "./dependencies/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
# AgentPositionSensorConfig revelent
from dataclasses import dataclass
from habitat.config.default_structured_configs import LabSensorConfig
from omegaconf import MISSING
from register_new_sensors_and_measures import AgentPositionSensor
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from env_tools.arguments import args



@dataclass
class AgentPositionSensorConfig(LabSensorConfig):
    type: str = "my_supercool_sensor"
    answer_to_life: int = MISSING

def hm3d_config(path:str=HM3D_CONFIG_PATH,stage:str='val',episodes=200, max_steps=500):
    path = HM3D_CONFIG_PATH
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
        # habitat_config.habitat.dataset.data_path = "./dependencies/habitat-lab/data/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz"
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
        
        if(args.is_one_rgb==True):
            habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 640
            habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 480
            habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = 79
    return habitat_config


def make_cfg(settings, habitat_env):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    # sim_cfg.scene_dataset_config_file = "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"

    # Note: all sensors must have the same resolution
    sensor_specs = []

    gt_spec = habitat_sim.CameraSensorSpec()
    gt_spec.uuid = "gt"
    gt_spec.sensor_type = habitat_sim.SensorType.COLOR
    gt_spec.resolution = [settings["height"], settings["width"]]
    gt_spec.position = [0, settings["camera_height"], 0]
    gt_spec.hfov = settings["hfov"]
    
    # euler = quaternion.as_euler_angles(habitat_env.sim.agents[0].get_state().sensor_states["rgb"].rotation)
    # print("euler:", euler)
    
    # if euler[0]!=0:
    #     euler[1] = 2*euler[0]-euler[1]
    #     euler[0] = 0
    #     euler[2] = 0

    # gt_spec.orientation = euler
    gt_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(gt_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        HabitatSimActions.move_forward: habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        HabitatSimActions.turn_left: habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        HabitatSimActions.turn_right: habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }
    agent_cfg.radius = settings["radius"]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])



def init_gt_sensor(habitat_env):
    test_scene = habitat_env.current_episode.scene_id.replace(".basis.", ".semantic.")
    # test_scene = habitat_env.current_episode.scene_id
    sim_settings = {
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "hfov": 90,
        "camera_height": 0.88, 
        "radius": 0.18,
        "scene": test_scene,  # Scene path
        "default_agent": 0,
        "color_sensor": True,  # RGB sensor
        "semantic_sensor": False,  # Semantic sensor
        "depth_sensor": False  # Depth sensor
    }
    cfg = make_cfg(sim_settings, habitat_env)
    vln_sim = habitat_sim.Simulator(cfg)
    # Set agent state
    vln_agent = vln_sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = habitat_env._sim.get_agent_state(0).position
    agent_state.rotation = habitat_env._sim.get_agent_state(0).rotation
    vln_agent.set_state(agent_state)
    return vln_sim


# =====> color_dict <=====
color_dict = {
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00669-DNWbUAJYsPy/DNWbUAJYsPy.basis.glb":{
        "tv_monitor": ["480336", "101838"], 
        "bed": ["024813"], 
        "sofa": ["830000"], 
        "chair": ["6A4006", "FC0001", "010000", "31264A", "4F021A", "018100", "29170A", "9B5A52", "18122C"], 
        "toilet": ["2B0D00", "004070"]
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00166-RaYrxWt5pR1/RaYrxWt5pR1.basis.glb":{
        "tv_monitor": ["675BD4"], 
        "toilet": ["8E1C13", "C9F88E"], 
        "chair": ["0E6018", "01300E", "005583", "077B08", "7F7708", "A9071C", "15A003", "000072", "210A17", "D72B08", "D1A03D"], 
        "plant": ["D41F0F", "3B0030", "14062C", "0B8343", "1005E3", "05EF33", "CE8903", "A20315", "0D0001", "034C0A", "709200"], 
        "sofa": ["03067F"]
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00404-QN2dRqwd84J/QN2dRqwd84J.basis.glb":{
        "sofa": ["15000D", "7900F8", "000646"], 
        "bed": ["004700", "B10025"], 
        "plant": ["113F5A", "0E1E0D", "00E337"], 
        "tv_monitor": ["16F5DD", "332BE0", "004000"], 
        "toilet": ["020800", "0000F5", "14000E"]
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00706-YHmAkqgwe2p/YHmAkqgwe2p.basis.glb":{
        "bed": ["70904F"], 
        "toilet": ["09074F"], 
        "chair": ["750614", "030021", "024E62", "4858AE", "2E4A73", "220E89", "8CA700", "000107", "92097F", "830001", "01364E", "00030C", "080233", "2D1C01", "031A00"], 
        "sofa": ["404A00", "193C1A", "380106"]
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00324-DoSbsoo4EAg/DoSbsoo4EAg.basis.glb":{
        "bed": ["080E12", "520F55"], 
        "tv_monitor": ["A50D67", "CB1400", "0A0606"]
        },

        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00017-oEPjPNSPmzL/oEPjPNSPmzL.basis.glb":{
        "bed": ["3D732D", "0EE00E"], 
        "tv_monitor": ["976712"], 
        "toilet": ["010936"], 
        "sofa": ["5D2C08"], 
        "plant": ["1C049D",  "002601", "3703CB", "0A00EF", "8C1C0A"]
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00031-Wo6kuutE9i7/Wo6kuutE9i7.basis.glb":{
        "bed": ["191D02", "BE1211"], 
        "tv_monitor": ["4C265A"], 
        "toilet": ["0004C0"]
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00099-226REUyJh2K/226REUyJh2K.basis.glb":{
        "bed": ["1EC600", "700609", "092101"], 
        "tv_monitor": ["397D01", "062503", "0472BE"]},
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00105-xWvSkKiWQpC/xWvSkKiWQpC.basis.glb":{
        "tv_monitor": ["123BA0", "02EF18", "030100", "A90007"], 
        "toilet": ["06F840"], 
        "sofa": ["5FDD72"]
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00250-U3oQjwTuMX8/U3oQjwTuMX8.basis.glb":{
        "bed": ["312E00", "000100", "0FCE77", "000007", "7D1F00", "290124"], 
        "toilet": ["513405", "46002B", "1E0200", "2303BE"],
        "sofa": ["AE0701", "04F82B", "00F2D7", "4B0536", "013EF5", "850005", "A52958"], 
        "plant": ["010003", "65000B", "440204", "8C04EF", "349701", "074A00", "377D12", "15B11B", "F53100", "584A3E", "5F0200"],
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00251-wsAYBFtQaL7/wsAYBFtQaL7.basis.glb":{
        "bed": ["4B0201", "3363AE", "420001", "080BEC"], 
        "toilet": ["010A0A", "440009"],
        "sofa": ["905F00", "587715", "0800C3"]
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00254-YMNvYDhK8mB/YMNvYDhK8mB.basis.glb":{
        "chair": ["010047", "200058", "00B31C", "997BBE", "014603", "920201", "CE0DA2", "0A2BAE", "421A06", "0B6300", "2C0100", "0015AC", "000021", "15038C", "900039", "00E9EC", "07D44A", "011700", "FC023C", "00921D", "022640", "05006E", "060658", "3B270A", "5A9D58", "012192", "3F0804", "2C2F2A", "DA0047", "369007"], 
        "plant": ["35015A", "243915", "F26A58", "A065A0"]
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00255-NGyoyh91xXJ/NGyoyh91xXJ.basis.glb":{
        "bed": ["040101", "0DC001", "674AA7", "7D0872"], 
        "tv_monitor": ["D40A58", "03000C", "001916"], 
        "toilet": ["0009E6", "1AAC38", "010000", "003775", "4008BB"]
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00323-yHLr6bvWsVm/yHLr6bvWsVm.basis.glb":{
        "bed": ["8C0781", "03D175", "E901EF", "0B02A9"], 
        "tv_monitor": ["7D9B02"], 
        "toilet": ["CE29E9", "010060"]
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00327-xgLmjqzoAzF/xgLmjqzoAzF.basis.glb":{
        "bed": ["003467", "131E06", "00001C", "73B108"], 
        "toilet": ["1E050C", "600663", "520209"], 
        "chair":["1C0401", "011C0D", "CE0006", "CE0308", "0E0F03", "1C0213", "8EEC39", "5D0305", "30DDBB", "E36A07", "29CE99", "AE8721", "9D194B", "0D777F", "D76E16", "E3A52C", "260037", "3177EC", "03EF5F"]
        },


        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/val/00831-yr17PDCnDDW/yr17PDCnDDW.basis.glb":{
        "tv_monitor": ["4A7202", "69010D", "628C09", "07625D", "010079", "E907B3", "CB01B6" "00DA08", "131D6E", "005D51", "01001C", "2F2F60"],
        "plant": ["811B04", "010408", "380000", "07AE16", "7D0C04", "4B3100", "465A01", "4B2A02", "44C000", "9D0221", "19000A", "8C0051", "073B67", "0617C3", "4B0011", "006309", "000852"],
        "sofa": ["040405"],
        "chair": ["1E1A30", "074005", "0E5424", "00EF06", "290747", "B34F00", "040000", "5B02C3", "0E2BC6", "3DFFC3", "05ECC9", "002600", "2110FC", "52E305", "121E46", "093E00", "306226", "4E1B4F", "6C0000", "0062C0", "030000", "907303", "A7E30A", "EF6C3E", "FF5404", "ACA504", "F22103", "235B00", "464013"]
        },
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/val/00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.glb":{
        "sofa": ["4000CE"],
        "tv_monitor": ["773C01", "090236"]
        }
    }
# =====> color_dict <=====