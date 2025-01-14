import os
import argparse
from glob import glob


parser = argparse.ArgumentParser(
    description='Arguments to evaluate stable ratio.'
)

parser.add_argument('--dataset_type', type=str)
parser.add_argument('--test_idx', type=int, default=1)          # different test_idx for different test times to avoid issacgym conflict
parser.add_argument('--vis', default=False, action="store_true", help="If set, visualize the simulation")
args = parser.parse_args()

exp_root_path = 'exps'
dataset_type = args.dataset_type                # scannet, scannetpp, replica
test_idx = args.test_idx
mode = 'eval'
if args.vis:
    mode = 'vis'

sum_stable_ratio = 0                # for all scenes
total_scene = 0

save_results_txt = os.path.join(exp_root_path, f'stable_ratio_{dataset_type}.txt')
if os.path.exists(save_results_txt):
    os.system(f'rm {save_results_txt}')

exp_list = os.listdir(exp_root_path)
exp_list.sort()
for exp_name in exp_list:
    if f'_{dataset_type}_' not in exp_name:
        continue

    scan_id = exp_name.split('_')[-1]
    temp_exp_path = os.path.join(exp_root_path, exp_name)
    run_exp_list = os.listdir(temp_exp_path)
    run_exp_list.sort()
    run_exp_name = run_exp_list[-1]             # use the latest exp
    exp_path = os.path.join(temp_exp_path, run_exp_name, 'plots')

    # generate urdf files
    cmd = f'python eval/template-urdf/gen_stl_urdf.py --root_path {exp_path} --dataset_type {dataset_type} --scan_id {scan_id} --test_idx {test_idx}'
    print(cmd)
    os.system(cmd)

    urdf_path = os.path.join(exp_path, 'urdf')
    urdf_list = os.listdir(urdf_path)
    obj_urdf_list = [x for x in urdf_list if 'object' in x]
    obj_num = len(obj_urdf_list)
    print(f'{dataset_type} {scan_id} obj_num: {obj_num}')

    if os.path.exists(os.path.join(exp_path, 'results')):
        os.system(f'rm -r {os.path.join(exp_path, "results")}')
    if os.path.exists(os.path.join(exp_path, 'sim-render')):
        os.system(f'rm -r {os.path.join(exp_path, "sim-render")}')

    for obj_idx in range(1, obj_num+1):
        obj_urdf_name = f'{dataset_type}_{scan_id}_object_{obj_idx}.urdf'
        bg_urdf_name = f'{dataset_type}_{scan_id}_bg.urdf'

        mesh_root_path = os.path.join(exp_path, 'mesh')
        for mesh_name in os.listdir(mesh_root_path):
            if f'_{obj_idx}.stl' in mesh_name:
                break
        obj_mesh_path = os.path.join(mesh_root_path, mesh_name)
        cmd = f'python eval/eval_single_object_stability.py --asset_root {exp_path} --obj_urdf_name {obj_urdf_name} --bg_urdf_name {bg_urdf_name} --obj_mesh_path {obj_mesh_path} --mode {mode}'
        os.system(cmd)

    result_root_path = os.path.join(exp_path, 'results')
    stable_obj_list = glob(os.path.join(result_root_path, '*_1.txt'))       # 1 means stable, 0 means unstable
    stable_ratio = len(stable_obj_list) * 1.0 / obj_num * 1.0
    print(f'{dataset_type} {scan_id} stable_ratio: {stable_ratio}')

    with open(save_results_txt, 'a') as f:
        f.write(f'{dataset_type} {scan_id} stable_ratio: {stable_ratio}\n')

    sum_stable_ratio += stable_ratio
    total_scene += 1


average_stable_ratio = sum_stable_ratio / total_scene
print(f'{dataset_type} total_scene: {total_scene}, average stable_ratio: {average_stable_ratio}')
with open(save_results_txt, 'a') as f:
    f.write(f'{dataset_type} total_scene: {total_scene}, average stable_ratio: {average_stable_ratio}\n')

