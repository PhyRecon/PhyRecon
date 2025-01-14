# adapted from https://github.com/zju3dv/manhattan_sdf
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh
import torch
import glob
import os
import pyrender
import os
from tqdm import tqdm
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# hard-coded image size
H = 680
W = 1200


def inv_fix_pose(pose):
    # replica dataset not need to fix pose, so need to inv fix pose
    # 3D Rotation about the x-axis.
    t = -np.pi
    c = np.cos(t)
    s = np.sin(t)
    R = np.array([[1, 0, 0],
                    [0, c, -s],
                    [0, s, c]])
    axis_transform = np.eye(4)
    axis_transform[:3, :3] = R
    return pose @ axis_transform

# load pose
def load_poses(path):
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = inv_fix_pose(c2w)                     # inv fix pose
        c2w = torch.from_numpy(c2w).float()
        poses.append(c2w)
    return poses

class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh, smooth = False)

    def delete(self):
        self.renderer.delete()

def refuse(mesh, poses, K):
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=3 * 0.01,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for pose in tqdm(poses):
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K
        
        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl)
        
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    return volume.extract_triangle_mesh()


parser = argparse.ArgumentParser(
    description='Arguments to cull the mesh.'
)

parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be culled')
parser.add_argument('--input_scalemat', type=str, default=None,  help='path to the scale mat')
parser.add_argument('--traj', type=str,  help='path to the trajectory')
parser.add_argument('--output_mesh', type=str,  help='path to the output mesh')
args = parser.parse_args()

# for replica dataset
fx = 600.0
fy = 600.0
fx = 600.0
cx = 599.5
cy = 339.5
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

poses = load_poses(args.traj)
n_imgs = len(poses)
mesh = trimesh.load(args.input_mesh, process=False)

# transform to original coordinate system with scale mat
if args.input_scalemat is not None:
    scalemat = np.load(args.input_scalemat)['scale_mat_0']
    mesh.vertices = mesh.vertices @ scalemat[:3, :3].T + scalemat[:3, 3] 
else:
    # print('not input scalemat')
    scalemat = np.eye(4)
    mesh.vertices = mesh.vertices @ scalemat[:3, :3].T + scalemat[:3, 3] 

mesh = refuse(mesh, poses, K)
out_mesh_path = args.output_mesh
o3d.io.write_triangle_mesh(out_mesh_path, mesh)
print(f"Saved tsdf mesh to {out_mesh_path}")
