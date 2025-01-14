import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import sys
sys.path.append(os.getcwd())

class DenseGrid(nn.Module):
    '''
    Dense grid, each point records importance sampling factors directly.
    NOTE: grid mapping total scene.
    '''
    def __init__(self, fill_data, resolution=256, divide_factor=1.1, device='cuda'):
        super().__init__()

        self.resolution = resolution
        self.divide_factor = divide_factor  # used to ensure the object bbox is in the grid
        
        # dense grid
        self.phygrid = nn.Parameter(torch.empty(1, 1, resolution, resolution, resolution, dtype=torch.float32, device=device))
        self.reset_parameters(fill_data)

        # create gaussian kernel
        self.kernel_size = 5
        self.sigma = 0.5
        self.kernel = self.gaussian_kernel(size=self.kernel_size, sigma=self.sigma, device=device)

    def reset_parameters(self, fill_data):
        
        self.phygrid.data.fill_(fill_data)

    def gaussian_kernel(self, size=5, sigma=0.5, device='cuda'):
        """
        creates gaussian kernel with side length size and a sigma of sigma
        """
        grid_x, grid_y, grid_z = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32, device=device) for _ in range(3)], indexing='ij'
        )
        grid_x = grid_x - size // 2
        grid_y = grid_y - size // 2
        grid_z = grid_z - size // 2
        sq_distances = grid_x ** 2 + grid_y ** 2 + grid_z ** 2
        kernel = torch.exp(-sq_distances / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # add two dimensions for 'batch' and 'channel'

    def update_phygrid(self):
        # update phygrid
        # NOTE: update phygrid after optimizer.step()
        # use gaussian filter to smooth phygrid
        self.phygrid.data = F.conv3d(self.phygrid.data, self.kernel, padding=self.kernel_size//2)

    def forward(self, points):
        '''
        points: [num_points, 3]
        '''
        num_points = points.shape[0]

        # ensure the object bbox is in the grid
        points = points / self.divide_factor

        # NOTE: transpose x, z
        points = points[:, [2, 1, 0]]                       # [num_points, 3]

        points = points.reshape(1, 1, 1, num_points, 3)     # [1, 1, 1, num_points, 3]

        sampling_factor = F.grid_sample(self.phygrid, points, align_corners=True)  # [1, 1, 1, 1, num_points]
        sampling_factor = sampling_factor.reshape(num_points, 1)      # [num_points, 1]

        return sampling_factor
    
    def get_loss(self, sampling_factor):
        '''
        sampling_factor: [num_points, 1]
        '''
        loss = -1.0 * torch.sum(sampling_factor)           # increase contact points importance

        return loss
    
    def grid_parameters(self):
        print("grid parameters", len(list(self.parameters())))
        for p in self.parameters():
            print(p.shape)
        return self.parameters()

