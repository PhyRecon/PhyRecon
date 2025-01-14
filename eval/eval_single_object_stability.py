import os
import time
import numpy as np
import trimesh
import imageio
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch


def quaternion_to_euler(x, y, z, w):
    """
    Converts a quaternion into Euler angles (Z-Y-X order)
    """
    # Calculate yaw (psi), z-axis rotation
    psi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    # Calculate pitch (theta), y-axis rotation
    theta = np.arcsin(2 * (w * x - y * z))
    
    # Calculate roll (phi), x-axis rotation
    phi = np.arctan2(2 * (w * y + z * x), 1 - 2 * (x**2 + y**2))
    
    return np.degrees(psi), np.degrees(theta), np.degrees(phi)  # Convert radians to degrees


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
custom_parameters = [
    {"name": "--asset_root", "type": str, "required": True, "help": "path to assets"},
    {"name": "--obj_urdf_name", "type": str, "required": True, "help": "name to object urdf file"},
    {"name": "--bg_urdf_name", "type": str, "required": True, "help": "name to background urdf file"},
    {"name": "--obj_mesh_path", "type": str, "required": True, "help": "path to object stl file, used for camera look at"},
    {"name": "--mode", "type": str, "default": 'eval'},
]
args = gymutil.parse_arguments(description="Stability Evaluation", custom_parameters=custom_parameters)

ASSET_ROOT = args.asset_root
OBJ_ASSET_FILE = os.path.join('urdf', args.obj_urdf_name)
BG_ASSET_FILE = os.path.join('urdf', args.bg_urdf_name)
obj_mesh_path = args.obj_mesh_path
mode = args.mode

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim_params.up_axis = gymapi.UP_AXIS_Z  ## set z up axis in sim
sim_params.gravity.x = 0
sim_params.gravity.y = 0
sim_params.gravity.z = -9.81
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# set up the env grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, 1.0)

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.vhacd_enabled = True
asset_options.vhacd_params = gymapi.VhacdParams()
asset_options.vhacd_params.resolution = 1000000
asset_options.linear_damping = 10
asset_options.angular_damping = 10
asset_options.max_angular_velocity = 0.05
asset_options.max_linear_velocity = 0.05

print("Loading asset '%s' from '%s'" % (OBJ_ASSET_FILE, ASSET_ROOT))
object_asset = gym.load_asset(sim, ASSET_ROOT, OBJ_ASSET_FILE, asset_options)

asset_options.fix_base_link = True          # fix the base link of the background
asset_options.vhacd_params.resolution = 5000000
print("Loading asset '%s' from '%s'" % (BG_ASSET_FILE, ASSET_ROOT))
background_asset = gym.load_asset(sim, ASSET_ROOT, BG_ASSET_FILE, asset_options)

obj_idx = args.obj_urdf_name.split('_')[-1].split('.')[0]
cam_root_path = os.path.join(ASSET_ROOT, 'sim-render', f'obj_{obj_idx}')
os.makedirs(cam_root_path, exist_ok=True)
results_root_path = os.path.join(ASSET_ROOT, 'results')
os.makedirs(results_root_path, exist_ok=True)

# Create environment 0
env = gym.create_env(sim, env_lower, env_upper, 2)
# initial the base pose of the object
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
initial_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
object_actor = gym.create_actor(env, object_asset, initial_pose, 'object', 0)
initial_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
actor_bg = gym.create_actor(env, background_asset, initial_pose, 'background', 0)

# Viewer look at the first env
obj_mesh = trimesh.load_mesh(obj_mesh_path)
obj_center = obj_mesh.centroid
print(f'Object center: {obj_center}')

cam_pos = gymapi.Vec3(0.1, 0, 0)
cam_target = gymapi.Vec3(obj_center[0], obj_center[1], obj_center[2])
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# add camera
cam_props = gymapi.CameraProperties()
cam_props.width = 512
cam_props.height = 512
cam_props.enable_tensors = True
cam_handle = gym.create_camera_sensor(env, cam_props)
gym.set_camera_location(cam_handle, env, cam_pos, cam_target)
# obtain camera tensor
cam_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)
# wrap camera tensor in a pytorch tensor
torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)

# set object contact_offset
object_rigid_props = gym.get_actor_rigid_shape_properties(env, object_actor)
for prop in object_rigid_props:
    prop.contact_offset = 0.003                     # default is 0.01999
gym.set_actor_rigid_shape_properties(env, object_actor, object_rigid_props)

# set bg contact_offset
bg_rigid_props = gym.get_actor_rigid_shape_properties(env, actor_bg)
for prop in bg_rigid_props:
    prop.contact_offset = 0.003                     # default is 0.01999
gym.set_actor_rigid_shape_properties(env, actor_bg, bg_rigid_props)

# save the initial state
initial_state = gym.get_actor_rigid_body_states(env, object_actor, gymapi.STATE_ALL)[0]
initial_pos = np.array(initial_state["pose"]["p"].tolist())
initial_rot = np.array(initial_state["pose"]["r"].tolist())
initial_lin_vel = np.array(initial_state["vel"]["linear"].tolist())
initial_ang_vel = np.array(initial_state["vel"]["angular"].tolist())
print('*' * 50)
print(f'Initial OBJECT STATES')
print(f'position: {initial_state["pose"]["p"]}')
print(f'rotation: {initial_state["pose"]["r"]}')
print(f'linear velocity: {initial_state["vel"]["linear"]}')
print(f'angular velocity: {initial_state["vel"]["angular"]}')

# Simulate
sim_step = 0
if mode == 'eval':
    while sim_step < 200:
        print(f'Step: {sim_step}')
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Get the object state for evaluation
        object_state = gym.get_actor_rigid_body_states(env, object_actor, gymapi.STATE_ALL)[0]
        # print('-' * 50)
        # print(f'STEP: {sim_step} | OBJECT STATES')
        # print(f'position: {object_state["pose"]["p"]}')
        # print(f'rotation: {object_state["pose"]["r"]}')
        # print(f'linear velocity: {object_state["vel"]["linear"]}')
        # print(f'angular velocity: {object_state["vel"]["angular"]}')

        # count steps
        sim_step += 1

        # render sensors and refresh camera tensors
        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)

        # render the final state
        fname = f'{sim_step}_step.png'
        fpath = os.path.join(cam_root_path, fname)
        cam_img = torch_cam_tensor.cpu().numpy()
        imageio.imwrite(fpath, cam_img)

        gym.end_access_image_tensors(sim)


    # calculate the change between the initial and final position
    final_state = gym.get_actor_rigid_body_states(env, object_actor, gymapi.STATE_ALL)[0]
    pos_change = np.array(final_state["pose"]["p"].tolist()) - initial_pos
    rot_change = np.array(final_state["pose"]["r"].tolist())                                      # quaternion PhysX is (x, y, z, w)
    z_angle, y_angle, x_angle = quaternion_to_euler(*rot_change)
    lin_vel_change = np.array(final_state["vel"]["linear"].tolist()) - initial_lin_vel
    ang_vel_change = np.array(final_state["vel"]["angular"].tolist()) - initial_ang_vel
    print('*' * 50)
    print(f'Change for {sim_step} steps')
    print(f'position: {pos_change}')
    print(f'rotation: {rot_change}')
    print(f'rotation (XYZ euler): {x_angle, y_angle, z_angle}')
    print(f'linear velocity: {lin_vel_change}')
    print(f'angular velocity: {ang_vel_change}')
    print('*' * 50)

    max_pos_change = np.max(np.abs(pos_change))
    z_pos_drop = -1.0 * pos_change[2]                           # drop direction is negative z
    max_rot_change = max(np.abs(y_angle), np.abs(x_angle))
    stable_flag = 0
    if max_rot_change < 5 and z_pos_drop < 0.05:
        stable_flag = 1
        print('The object is stable')

    # save the change to txt file
    change_txt = os.path.join(results_root_path, f'obj_{obj_idx}_results_{stable_flag}.txt')
    with open(change_txt, 'w') as f:
        f.write(f'Change for {sim_step} steps\n')
        f.write(f'position: {pos_change}\n')
        f.write(f'rotation: {rot_change}\n')
        f.write(f'rotation (XYZ euler): {x_angle, y_angle, z_angle}\n')
        f.write(f'linear velocity: {lin_vel_change}\n')
        f.write(f'angular velocity: {ang_vel_change}\n')
        f.write(f'Stable flag: {stable_flag}\n')

elif mode == 'vis':
    while not gym.query_viewer_has_closed(viewer):
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)
        time.sleep(0.2)

        # Get the object state for evaluation
        object_state = gym.get_actor_rigid_body_states(env, object_actor, gymapi.STATE_ALL)[0]
        print('-' * 50)
        print(f'STEP: {sim_step} | OBJECT STATES')
        print(f'position: {object_state["pose"]["p"]}')
        print(f'rotation: {object_state["pose"]["r"]}')
        print(f'linear velocity: {object_state["vel"]["linear"]}')
        print(f'angular velocity: {object_state["vel"]["angular"]}')

        # count steps
        sim_step += 1

        # render sensors and refresh camera tensors
        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)

        # render the final state
        fname = f'{sim_step}_step.png'
        fpath = os.path.join(cam_root_path, fname)
        cam_img = torch_cam_tensor.cpu().numpy()
        imageio.imwrite(fpath, cam_img)

        gym.end_access_image_tensors(sim)

else:
    raise ValueError(f'Invalid mode: {mode}')

print('Evaluation Done')
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
