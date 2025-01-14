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

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# hard-coded image size
H, W = 968, 1296


# load pose
def load_poses(scan_id):
    pose_path = os.path.join(f'data/scannet/scan{scan_id}', 'pose')
    poses = []
    pose_paths = sorted(glob.glob(os.path.join(pose_path, '*.txt')),
                        key=lambda x: int(os.path.basename(x)[:-4]))
    for pose_path in pose_paths[::10]:
        c2w = np.loadtxt(pose_path)
        if np.isfinite(c2w).any():
            poses.append(c2w)
    poses = np.array(poses)
    
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
        return pyrender.Mesh.from_trimesh(mesh)

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
parser.add_argument('--input_scalemat', type=str,  help='path to the scale mat')
parser.add_argument('--output_mesh', type=str,  help='path to the output mesh')
parser.add_argument('--use_gt', default=False, action="store_true", help='whether to use ground truth poses')
args = parser.parse_args()

mesh = trimesh.load(args.input_mesh, process=False)

cam_dict = np.load(args.input_scalemat)
scale_mat = cam_dict['scale_mat_0']
if not args.use_gt:
    mesh.vertices = (scale_mat[:3, :3] @ mesh.vertices.T + scale_mat[:3, 3:]).T

# load pose and intrinsic for render depth 
poses = []
poses_len = int(len(cam_dict.files) / 3)
for i in range(poses_len):
    camera_extrinsic = cam_dict['camera_extrinsic_' + str(i)]
    pose = np.linalg.inv(camera_extrinsic)
    poses.append(pose)
poses = np.array(poses)

intrinsic_path = (args.input_scalemat).replace('cameras.npz', 'camera_intrinsic.npy')
K = np.load(intrinsic_path)[:3, :3]

mesh = refuse(mesh, poses, K)

# save mesh
out_mesh_path = args.output_mesh
o3d.io.write_triangle_mesh(out_mesh_path, mesh)
print(f"Saved tsdf mesh to {out_mesh_path}")
