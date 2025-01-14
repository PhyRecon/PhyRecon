import os
import trimesh
import shutil
import argparse
import numpy as np
import time


def resolve_collision(obj_mesh, background_mesh, z_floor, num_samples=1000, split_num=1000):

    # fix z collision
    z_floor = z_floor + 0.001       # add a small value to avoid z collision
    obj_z_min = obj_mesh.vertices[:, 2].min()
    delta_z = max(0, z_floor - obj_z_min)
    obj_mesh.vertices[:, 2] += delta_z

    sample_points, _ = trimesh.sample.sample_surface(obj_mesh, num_samples)

    ray_origins = np.zeros_like(sample_points)
    ray_origins[:, 2] = sample_points[:, 2]

    ray_directions = sample_points - ray_origins
    ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True)

    t1 = time.time()

    all_locations = []
    all_index_ray = []
    all_index_tri = []

    for i in range(0, len(ray_directions), split_num):
        batch_ray_directions = ray_directions[i:i + split_num]
        batch_ray_origins = ray_origins[i:i + split_num]

        # mesh intersection points with rays
        locations, index_ray, index_tri = background_mesh.ray.intersects_location(
            ray_origins=batch_ray_origins,
            ray_directions=batch_ray_directions
        )

        all_locations.append(locations)
        all_index_ray.append(index_ray + i)
        all_index_tri.append(index_tri)

    all_locations = np.vstack(all_locations) if all_locations else np.empty((0, 3))
    all_index_ray = np.hstack(all_index_ray) if all_index_ray else np.array([], dtype=int)
    all_index_tri = np.hstack(all_index_tri) if all_index_tri else np.array([], dtype=int)

    intersection_points = []
    ray_distances = np.linalg.norm(sample_points - ray_origins, axis=1)
    grouped_locations = {}
    for loc, ray_idx in zip(all_locations, all_index_ray):
        if ray_idx not in grouped_locations:
            grouped_locations[ray_idx] = []
        grouped_locations[ray_idx].append(loc)

    max_intersections_dist = 0
    max_intersections_dx = 0
    max_intersections_dy = 0
    for ray_idx, point_distance in enumerate(ray_distances):
        if ray_idx in grouped_locations:
            distances = np.linalg.norm(grouped_locations[ray_idx] - ray_origins[ray_idx], axis=1)
            min_indices = np.argmin(distances)
            nearest_distance = distances[min_indices]
            if nearest_distance < point_distance:

                if point_distance - nearest_distance > max_intersections_dist:
                    max_intersections_dist = point_distance - nearest_distance
                    max_intersections_dx = sample_points[ray_idx][0] - grouped_locations[ray_idx][min_indices][0]
                    max_intersections_dy = sample_points[ray_idx][1] - grouped_locations[ray_idx][min_indices][1]

                intersection_points.append(sample_points[ray_idx])

    t2 = time.time()
    print(f'Intersection detection time: {t2 - t1:.2f}s, num of intersection points: {len(intersection_points)}, max_dist: {max_intersections_dist}, max_dx: {max_intersections_dx}, max_dy: {max_intersections_dy}')

    obj_mesh.vertices[:, 0] -= max_intersections_dx
    obj_mesh.vertices[:, 1] -= max_intersections_dy

    error_flag = False
    max_change = max(max_intersections_dx, max_intersections_dy)
    if max_change > 0.1:
        error_flag = True
        print(f'Warning: max change is {max_change}, maybe error occurs.')

    return obj_mesh, error_flag, max_intersections_dx, max_intersections_dy

def ply2stl(ply_path, stl_path, bg_mesh_path, z_floor, solve_collision=False):
    
    mesh = trimesh.load(ply_path)

    if solve_collision:
        bg_mesh = trimesh.load(bg_mesh_path)
        mesh, error_flag, max_intersections_dx, max_intersections_dy = resolve_collision(mesh, bg_mesh, z_floor)

    mesh.export(stl_path)

    if solve_collision:
        if error_flag:
            return False, max_intersections_dx, max_intersections_dy
        
    return True, 0, 0

def replace_string_in_urdf(file_path, save_path, old_str, new_str):
    """
    Reads a URDF file, replaces occurrences of old_str with new_str, and writes the changes back to the file.

    Args:
    file_path (str): Path to the URDF file.
    save_path (str): Path to save the new URDF file.
    old_str (str): String to be replaced.
    new_str (str): String to replace with.
    """
    # Read the content of the file
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Replace the specified string
    updated_content = file_content.replace(old_str, new_str)

    # Write the updated content to the save_path
    with open(save_path, 'w') as file:
        file.write(updated_content)


parser = argparse.ArgumentParser(
    description='Arguments to generate URDF.'
)

parser.add_argument('--root_path', type=str)
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--scan_id', type=int)
parser.add_argument('--test_idx', type=int)
args = parser.parse_args()

root_path = args.root_path
dataset_type = args.dataset_type
scan_id = args.scan_id
test_idx = args.test_idx

urdf_folder = os.path.join(root_path, 'urdf')
if os.path.exists(urdf_folder):
    shutil.rmtree(urdf_folder)
os.makedirs(urdf_folder, exist_ok=True)
mesh_folder = os.path.join(root_path, 'mesh')
if os.path.exists(mesh_folder):
    shutil.rmtree(mesh_folder)
os.makedirs(mesh_folder, exist_ok=True)

exp_name = f"{dataset_type}_{scan_id}"

template_color_str = '0.4 0.7 1.0 1.0'          # blue
color_str_list = [
    '1.0 1.0 0.8 1.0',          # yellow
    '0.8 1.0 0.8 1.0',          # green
    '1.0 0.8 0.8 1.0',          # red
    '0.4 0.7 1.0 1.0',          # blue
    '0.8 0.9 0.84 1.0',         # light green
    '0.9 0.8 0.9 1.0',          # light purple
    '0.87 0.72 0.53 1.0',       # light brown
]

# background
template_urdf_path = 'eval/template-urdf/bg.urdf'     # background template URDF file
old_str = 'EXAMPLE_PATH/bg.stl'
new_str = 'mesh/' + exp_name + f'_t{test_idx}_bg.stl'
save_urdf_path = os.path.join(urdf_folder, f'{exp_name}_bg.urdf')
replace_string_in_urdf(template_urdf_path, save_urdf_path, old_str, new_str)
src_bg_mesh = os.path.join('data', dataset_type, 'GTmesh', f'scan{scan_id}', 'bg.stl')
dst_bg_mesh = os.path.join(root_path, new_str)
shutil.copyfile(src_bg_mesh, dst_bg_mesh)

# z floor
z_floor_txt_path = os.path.join('data', dataset_type, f'scan{scan_id}', f'{dataset_type}_scan{scan_id}_zfloor.txt')
with open(z_floor_txt_path, 'r') as f:
    z_floor = float(f.readline().strip())

# note error txt
error_txt_path = os.path.join(root_path, 'error.txt')
if os.path.exists(error_txt_path):
    os.remove(error_txt_path)

mesh_list = os.listdir(root_path)
mesh_list = [mesh for mesh in mesh_list if mesh.endswith('.ply')]
mesh_list = [mesh for mesh in mesh_list if 'surface_' in mesh]
for mesh in mesh_list:
    obj_idx = mesh.split('_')[-1].split('.')[0]
    if obj_idx == 'whole' or obj_idx == '0':
        continue

    print(f'processing {obj_idx}...')
    ply_path = os.path.join(root_path, mesh)
    ply_name = os.path.basename(ply_path)
    stl_name = 'mesh/' + exp_name + f'_t{test_idx}_' + ply_name.replace('.ply', '.stl')     # to avoid issacgym confilct
    stl_path = os.path.join(root_path, stl_name)

    done_flag, max_intersections_dx, max_intersections_dy = ply2stl(ply_path, stl_path, src_bg_mesh, z_floor, solve_collision=True)
    if not done_flag:
        with open(error_txt_path, 'a') as f:
            f.write(f'{exp_name} obj_{obj_idx} max_intersections_dx: {max_intersections_dx}, max_intersections_dy: {max_intersections_dy}\n')

    template_urdf_path = 'eval/template-urdf/object.urdf'     # object template URDF file
    old_str = 'EXAMPLE_PATH/object.stl'
    new_str = stl_name
    save_urdf_path = os.path.join(urdf_folder, f'{exp_name}_object_{obj_idx}.urdf')
    replace_string_in_urdf(template_urdf_path, save_urdf_path, old_str, new_str)

    old_str = template_color_str
    new_str = color_str_list[int(obj_idx) % len(color_str_list)]
    replace_string_in_urdf(save_urdf_path, save_urdf_path, old_str, new_str)

    print(f'{obj_idx} done')


print(f'{exp_name} generate URDF done')
