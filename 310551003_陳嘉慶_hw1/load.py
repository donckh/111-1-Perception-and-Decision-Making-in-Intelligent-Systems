import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os

# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "apartment_0/apartment_0/habitat/mesh_semantic.ply"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": -np.pi/2,  # sensor pitch (x rotation in rads)
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img


def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img


def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        0.0,
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        0.0,
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        0.0,
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # RGB sensor BEV view
    rgb_bev_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_bev_sensor_spec.uuid = "color_bev_sensor"
    rgb_bev_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_bev_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_bev_sensor_spec.position = [0.0, 2.0, 0.0]  #under 2.6
    rgb_bev_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_bev_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor BEV view
    depth_bev_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_bev_sensor_spec.uuid = "depth_bev_sensor"
    depth_bev_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_bev_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_bev_sensor_spec.position = [0.0, 2.0, 0.0]
    depth_bev_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_bev_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor BEV view
    semantic_bev_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_bev_sensor_spec.uuid = "semantic_bev_sensor"
    semantic_bev_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_bev_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_bev_sensor_spec.position = [0.0, 2.0, 0.0]
    semantic_bev_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_bev_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec, rgb_bev_sensor_spec, depth_bev_sensor_spec, semantic_bev_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([5.0, 0.0, 8.0])  # agent in world space, level  [3.68, 0.0, 3.37], [0.5, 1.0, 9.5]
# agent_state.orientation =[0.34, 0, 0]
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print("#############################")


def navigateAndSee(action="", num=0, name=""):
    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)
        img_rgb = transform_rgb_bgr(observations["color_sensor"])
        img_depth = transform_depth(observations["depth_sensor"])
        img_sem = transform_semantic(observations["semantic_sensor"])
        img_rgb_bev = transform_rgb_bgr(observations["color_bev_sensor"])
        img_depth_bev = transform_depth(observations["depth_bev_sensor"])
        img_sem_bev = transform_semantic(observations["semantic_bev_sensor"])
        cv2.imshow("RGB", img_rgb)
        cv2.imshow("depth", img_depth)
        cv2.imshow("semantic", img_sem)
        cv2.imwrite("./save/RGB{}/{}.png".format(name, num), img_rgb)
        cv2.imwrite("./save/depth{}/{}.png".format(name, num), img_depth)
        # cv2.imwrite("./save/semantic/{}.png".format(num), img_sem)
        # cv2.imwrite("./save/RGB_bev/{}.png".format(num), img_rgb_bev)
        # cv2.imwrite("./save/depth_bev/{}.png".format(num), img_depth_bev)
        # cv2.imwrite("./save/semantic_bev/{}.png".format(num), img_sem_bev)
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        fp = open('./save/record%s.txt' % name, 'a')
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z, file=fp)
        fp.close()
        num+=1
        return num


action = "move_forward"
navigateAndSee(action)
count = 0
name = "_"
fp = open('./save/record%s.txt' % name, 'w')
fp.close()
os.makedirs('./save/RGB%s/' % name, exist_ok=True)
os.makedirs('./save/depth%s/' % name, exist_ok=True)
os.makedirs('./save/semantic/', exist_ok=True)
os.makedirs('./save/RGB_bev/', exist_ok=True)
os.makedirs('./save/depth_bev/', exist_ok=True)
os.makedirs('./save/semantic_bev/', exist_ok=True)

while True:
    keystroke = cv2.waitKey(0)
    if keystroke == ord(FORWARD_KEY):
        action = "move_forward"
        count = navigateAndSee(action, count, name)
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        count = navigateAndSee(action, count, name)
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        count = navigateAndSee(action, count, name)
        print("action: RIGHT")
    elif keystroke == ord(FINISH):
        print("action: FINISH")
        break
    else:
        print("INVALID KEY")
        continue
