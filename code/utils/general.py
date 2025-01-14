import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import transforms
import numpy as np
import json
import trimesh

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels, n_pixels=10000):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        if 'object_mask' in data:
            data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        if 'depth' in data:
            data['depth'] = torch.index_select(model_input['depth'], 1, indx)
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs

def get_max_component_mesh(mesh):

    connected_components = mesh.split(only_watertight=False)
    max_vertices = 0
    largest_component = None
    for component in connected_components:
        if len(component.vertices) > max_vertices:
            max_vertices = len(component.vertices)
            largest_component = component

    return largest_component

def get_obj_bbox(mesh, z_floor, delta):

    x_min, x_max = mesh.vertices[:, 0].min() - delta, mesh.vertices[:, 0].max() + delta
    y_min, y_max = mesh.vertices[:, 1].min() - delta, mesh.vertices[:, 1].max() + delta
    z_min, z_max = z_floor, mesh.vertices[:, 2].max() + delta

    obj_bbox = [[x_min, y_min, z_min], [x_max, y_max, z_max]]

    return obj_bbox

def calculate_bbox_distance(bbox_A, bbox_B):
    min_A, max_A = bbox_A
    min_B, max_B = bbox_B
    
    dist = np.zeros(3)
    for i in range(3):
        if max_A[i] < min_B[i]:
            dist[i] = min_B[i] - max_A[i]
        elif max_B[i] < min_A[i]:
            dist[i] = min_A[i] - max_B[i]
        else:
            dist[i] = 0

    return np.linalg.norm(dist)

def get_filtered_mesh(mesh, obj_bbox, bbox_dist_threshold):

    connected_components = mesh.split(only_watertight=False)
    filtered_connected_components = []

    for component in connected_components:

        x_mean, y_mean, z_mean = component.vertices.mean(axis=0)
        # mean points whether in the obj_bbox
        if obj_bbox[0][0] < x_mean < obj_bbox[1][0] and obj_bbox[0][1] < y_mean < obj_bbox[1][1] and obj_bbox[0][2] < z_mean < obj_bbox[1][2]:
            filtered_connected_components.append(component)
        else:           # if center not in obj bbox, check whether two bbox are near
            x_min, x_max = component.vertices[:, 0].min(), component.vertices[:, 0].max()
            y_min, y_max = component.vertices[:, 1].min(), component.vertices[:, 1].max()
            z_min, z_max = component.vertices[:, 2].min(), component.vertices[:, 2].max()
            obj_bbox_temp = [[x_min, y_min, z_min], [x_max, y_max, z_max]]
            # calculate two bbox distance
            bbox_dist = calculate_bbox_distance(obj_bbox, obj_bbox_temp)
            if bbox_dist < bbox_dist_threshold:
                filtered_connected_components.append(component)

    filtered_mesh = trimesh.util.concatenate(filtered_connected_components)
    return filtered_mesh

def refined_obj_bbox(plots_dir, z_floor, delta=0.02, bbox_dist_threshold=0.05):
    '''
    filter floaters and get the refined bbox
    '''

    bbox_root_path = os.path.join(plots_dir, 'bbox')

    mesh_list = os.listdir(plots_dir)
    mesh_list = [x for x in mesh_list if 'surface_' in x]
    epoch_list = [int(x.split('_')[1]) for x in mesh_list]
    epoch = max(epoch_list)
    latest_mesh_list = [x for x in mesh_list if f'surface_{epoch}_' in x]

    for mesh_name in latest_mesh_list:

        if '_0.ply' in mesh_name or '_whole.ply' in mesh_name:      # skip the whole and bg mesh
            continue

        obj_id = (mesh_name.split('.')[0]).split('_')[2]
        bbox_json_path = os.path.join(bbox_root_path, f'bbox_{obj_id}.json')
        if os.path.exists(bbox_json_path):
            os.remove(bbox_json_path)

        mesh_path = os.path.join(plots_dir, mesh_name)
        mesh = trimesh.load(mesh_path)
        max_component = get_max_component_mesh(mesh)
        obj_bbox = get_obj_bbox(max_component, z_floor, delta)

        filtered_mesh = get_filtered_mesh(mesh, obj_bbox, bbox_dist_threshold)
        refined_bbox = get_obj_bbox(filtered_mesh, z_floor, delta)
        with open(bbox_json_path, 'w') as f:
            json.dump(refined_bbox, f)
        print(f'obj {obj_id} refined bbox save to {bbox_json_path}')

        filtered_mesh_path = os.path.join(plots_dir, f'filtered_{mesh_name}')
        filtered_mesh.export(filtered_mesh_path)

def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)

def get_time():
    torch.cuda.synchronize()
    return time.time()

trans_topil = transforms.ToPILImage()


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points
