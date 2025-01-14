import torch
import numpy as np


def xyz_offset_mc(voxels, index_grid):
    '''
    simple implementation of marching cubes
    only calculate vertices, not faces

    voxels: [R, R, R], sdf
    index_grid: [R, R, R, 3], index grid
    NOTE: in this code, the scene is normalize in a grid_boundary, 
         so index_grid is the same as the pc coords, no need to transfer index_grid to pc coords
    '''

    def get_interpolate_index(voxels, voxels_offset, index_grid, index_grid_offset):
        '''
        voxels: [R, R, R], sdf
        voxels_offset: [R, R, R], sdf offset along x/y/z axis
        index_grid: [R, R, R, 3], index grid
        index_grid_offset: [R, R, R, 3], index grid offset along x/y/z axis

        s1 = voxels, s2 = voxels_offset, p1 = index_grid, p2 = index_grid_offset
        s1 + lamb * (s2 - s1) = 0, assume s1 < 0, s2 > 0
        lamb = s1 / (s1 - s2)
        p = p1 + lamb * (p2 - p1)
        '''
        eps = 1e-5
        lamb =  voxels / (voxels - voxels_offset + eps)
        index_grid_interpolate = index_grid.to(torch.float32) + lamb.unsqueeze(-1) * (index_grid_offset.to(torch.float32) - index_grid.to(torch.float32))

        return index_grid_interpolate
    
    # x offset sdf
    voxels_x_offset = torch.zeros_like(voxels)
    voxels_x_offset[:-1, :, :] = voxels[1:, :, :]
    voxels_x_offset[-1, :, :] = voxels[-1, :, :]        # last row is the same as the last row of voxels
    # x offset index
    index_grid_x_offset = torch.zeros_like(index_grid)
    index_grid_x_offset[:-1, :, :] = index_grid[1:, :, :]
    index_grid_x_offset[-1, :, :] = index_grid[-1, :, :]        # last row is the same as the last row of voxels
    # get x interpolate index
    index_grid_x_interpolate = get_interpolate_index(voxels, voxels_x_offset, index_grid, index_grid_x_offset)
    x_mask = (voxels * voxels_x_offset) < 0
    x_pc = index_grid_x_interpolate[x_mask]

    # y offset sdf
    voxels_y_offset = torch.zeros_like(voxels)
    voxels_y_offset[:, :-1, :] = voxels[:, 1:, :]
    voxels_y_offset[:, -1, :] = voxels[:, -1, :]        # last row is the same as the last row of voxels
    # y offset index
    index_grid_y_offset = torch.zeros_like(index_grid)
    index_grid_y_offset[:, :-1, :] = index_grid[:, 1:, :]
    index_grid_y_offset[:, -1, :] = index_grid[:, -1, :]        # last row is the same as the last row of voxels
    # get y interpolate index
    index_grid_y_interpolate = get_interpolate_index(voxels, voxels_y_offset, index_grid, index_grid_y_offset)
    y_mask = (voxels * voxels_y_offset) < 0
    y_pc = index_grid_y_interpolate[y_mask]

    # z offset sdf
    voxels_z_offset = torch.zeros_like(voxels)
    voxels_z_offset[:, :, :-1] = voxels[:, :, 1:]
    voxels_z_offset[:, :, -1] = voxels[:, :, -1]        # last row is the same as the last row of voxels
    # z offset index
    index_grid_z_offset = torch.zeros_like(index_grid)
    index_grid_z_offset[:, :, :-1] = index_grid[:, :, 1:]
    index_grid_z_offset[:, :, -1] = index_grid[:, :, -1]        # last row is the same as the last row of voxels
    # get z interpolate index
    index_grid_z_interpolate = get_interpolate_index(voxels, voxels_z_offset, index_grid, index_grid_z_offset)
    z_mask = (voxels * voxels_z_offset) < 0
    z_pc = index_grid_z_interpolate[z_mask]

    vertices = torch.cat([x_pc, y_pc, z_pc], dim=0)

    return vertices


def get_coarse_surface_points_mc(get_outputs, resolution, obj_bbox, obj_idx, adaptive_spacing=None, mc_level=0.0):
    '''
    get_outputs: use 'get_sdf_vals_and_sdfs' to get all sdf values(raw_sdf)
    mc_level: level of marching cubes
    '''

    print('********** get coarse surface points by marching cubes **********')

    grid_min = [obj_bbox[0][0], obj_bbox[0][1], obj_bbox[0][2]]
    grid_max = [obj_bbox[1][0], obj_bbox[1][1], obj_bbox[1][2]]

    if adaptive_spacing == None:
        xs = np.linspace(grid_min[0], grid_max[0], resolution)
        ys = np.linspace(grid_min[1], grid_max[1], resolution)
        zs = np.linspace(grid_min[2], grid_max[2], resolution)
        x_resolution, y_resolution, z_resolution = resolution, resolution, resolution
    else:
        x_resolution = int((grid_max[0] - grid_min[0]) / adaptive_spacing)
        y_resolution = int((grid_max[1] - grid_min[1]) / adaptive_spacing)
        z_resolution = int((grid_max[2] - grid_min[2]) / adaptive_spacing) + 32     # add 32 to make sure the z grid is enough
        x_resolution = max(min(x_resolution, 128), 96)
        y_resolution = max(min(y_resolution, 128), 96)
        z_resolution = max(min(z_resolution, 128), 96)
        xs = np.linspace(grid_min[0], grid_max[0], x_resolution)
        ys = np.linspace(grid_min[1], grid_max[1], y_resolution)
        zs = np.linspace(grid_min[2], grid_max[2], z_resolution)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

    # calculate all sdf values
    sdf_all_dic = {}
    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):

        _, sdf_all = get_outputs(pnts)          # sdf_all [N, instance_num]

        if not sdf_all_dic:
            for i in range(sdf_all.shape[-1]):
                sdf_all_dic[i] = sdf_all[:, i]
        else:
            for i in range(sdf_all.shape[-1]):
                sdf_all_dic[i] = torch.cat([sdf_all_dic[i], sdf_all[:, i]], dim=0)
    
    # get surface points by marching cubes
    points = points.reshape(x_resolution, y_resolution, z_resolution, 3)      # index_grid [M, R, R, R, 3], M is the number of objects
    sdf_obj = sdf_all_dic[obj_idx].reshape(x_resolution, y_resolution, z_resolution)
    surface_pc_mc = xyz_offset_mc(sdf_obj - mc_level, points)

    return surface_pc_mc


def get_fine_surface_points_mc(get_outputs, points, obj_idx, thresh=1e-3):
    '''
    get_outputs: use 'get_specific_outputs' to specific instance sdf and gradients
    '''

    def evaluate(pnts):

        sdf, _, gradients, _, _, indices = get_outputs(pnts)
        points_surface = pnts - sdf * gradients

        return points_surface, indices, sdf, gradients
    

    point_cloud = []
    object_indices = []
    sdf_list = []
    gradients_list = []
    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):

        points_surface, indices, sdf, gradients = evaluate(pnts)
        point_cloud.append(points_surface)
        object_indices.append(indices)
        sdf_list.append(sdf)
        gradients_list.append(gradients)

    point_cloud = torch.cat(point_cloud, dim=0)
    object_indices = torch.cat(object_indices, dim=0)
    sdf_list = torch.cat(sdf_list, dim=0)
    gradients_list = torch.cat(gradients_list, dim=0)

    # sdf offset mask
    offset_norm = torch.norm(sdf_list * gradients_list, dim=-1)       # [N]
    sdf_mask = (offset_norm < thresh).squeeze(-1)
    # index mask
    index_mask = (object_indices == obj_idx).squeeze(-1)
    mask = sdf_mask * index_mask

    # only save the points of specific object
    fine_point_cloud = point_cloud[mask]
    fine_gradients = gradients_list[mask]

    return fine_point_cloud, fine_gradients


