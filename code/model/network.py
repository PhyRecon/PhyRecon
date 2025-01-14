import os
import json
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler
import matplotlib.pyplot as plt
import numpy as np
import math

from torch import vmap

from simulator.sim_collision import Simulator
from model.PhyGrid import DenseGrid
from utils.spmc import get_coarse_surface_points_mc, get_fine_surface_points_mc

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=True,
            sigmoid = 10
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        print(multires, dims)
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.d_out = d_out
        self.sigmoid = sigmoid

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # Geometry initalization for compositional scene, bg SDF sign: inside + outside -, fg SDF sign: outside + inside -
                    # The 0 index is the background SDF, the rest are the object SDFs
                    # background SDF with postive value inside and nagative value outside
                    torch.nn.init.normal_(lin.weight[:1, :], mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[:1], bias) 
                    # inner objects with SDF initial with negative value inside and positive value outside, ~0.6 radius of background
                    torch.nn.init.normal_(lin.weight[1:,:], mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[1:], -0.6*bias) 

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.pool = nn.MaxPool1d(self.d_out, return_indices=True)

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:self.d_out]
        d_output = torch.ones_like(y[:, :1], requires_grad=False, device=y.device)
        g = []
        for idx in range(y.shape[1]):
            gradients = torch.autograd.grad(
                outputs=y[:, idx:idx+1],
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            g.append(gradients)
        g = torch.cat(g)
        # add the gradient of minimum sdf
        # sdf = -self.pool(-y.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-y.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        g_min_sdf = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        g = torch.cat([g, g_min_sdf])
        return g

    def get_outputs(self, x, beta=None):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf_raw = torch.minimum(sdf_raw, sphere_sdf.expand(sdf_raw.shape))
        if beta == None:
            semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        else:
            semantic = 0.5/beta *torch.exp(-sdf_raw.abs()/beta)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1) # get the minium value of sdf
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, sdf_raw

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:self.d_out]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        # sdf = -self.pool(-sdf) # get the minium value of sdf  if bound apply in the final 
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf.expand(sdf.shape))
        # sdf = -self.pool(-sdf.unsqueeze(1)).squeeze(-1) # get the minium value of sdf if bound apply before min
        sdf, indices = self.pool(-sdf.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        return sdf
    
    def get_sdf_raw(self, x):
        return self.forward(x)[:, :self.d_out]
    
    def get_object_sdf_vals(self, x, idx):
        sdf = self.forward(x)[:, idx]
        return sdf
    
    def get_sdf_vals_and_sdfs(self, x):
        sdf = self.forward(x)[:,:self.d_out]
        sdf_raw = sdf
        # sdf = -self.pool(-sdf.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-sdf.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        return sdf, sdf_raw
    
    def get_specific_outputs(self, x, idx):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)

        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, sdf_raw[:, idx]

    def get_shift_sdf_raw(self, x):
        sdf_raw = self.forward(x)[:, :self.d_out]
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)

        # shift raw sdf
        pos_min_sdf = -sdf          # other object sdf must bigger than -sdf
        pos_min_sdf_expand = pos_min_sdf.expand_as(sdf_raw)
        shift_mask = (sdf < 0)
        shift_mask_expand = shift_mask.expand_as(sdf_raw)

        shift_sdf_raw = torch.where(shift_mask_expand, torch.max(sdf_raw, pos_min_sdf_expand), sdf_raw)
        shift_sdf_raw[torch.arange(indices.size(0)), indices.squeeze()] = sdf.squeeze()

        return shift_sdf_raw

    def get_outputs_and_indices(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]

        # if self.sigmoid_optim:
        #     sigmoid_value = torch.exp(self.sigmoid_basis)
        # else:
        sigmoid_value = self.sigmoid

        semantic = sigmoid_value * torch.sigmoid(-sigmoid_value * sdf_raw)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1) # get the minium value of sdf
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return sdf, feature_vectors, gradients, semantic, sdf_raw, indices

from hashencoder.hashgrid import HashEncoder
class ObjectImplicitNetworkGrid(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0, # radius of the sphere in geometric initialization
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            base_size = 16,
            end_size = 2048,
            logmap = 19,
            num_levels=16,
            level_dim=2,
            divide_factor = 1.5, # used to normalize the points range for multi-res grid
            use_grid_feature = True, # use hash grid embedding or not, if not, it is a pure MLP with sin/cos embedding
            sigmoid = 20
    ):
        super().__init__()
        
        self.d_out = d_out
        self.sigmoid = sigmoid
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]
        self.embed_fn = None
        self.divide_factor = divide_factor
        self.grid_feature_dim = num_levels * level_dim
        self.use_grid_feature = use_grid_feature
        dims[0] += self.grid_feature_dim
        
        print(f"[INFO]: using hash encoder with {num_levels} levels, each level with feature dim {level_dim}")
        print(f"[INFO]: resolution:{base_size} -> {end_size} with hash map size {logmap}")
        self.encoding = HashEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim, 
                    per_level_scale=2, base_resolution=base_size, 
                    log2_hashmap_size=logmap, desired_resolution=end_size)
        
        '''
        # can also use tcnn for multi-res grid as it now supports eikonal loss
        base_size = 16
        hash = True
        smoothstep = True
        self.encoding = tcnn.Encoding(3, {
                        "otype": "HashGrid" if hash else "DenseGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 19,
                        "base_resolution": base_size,
                        "per_level_scale": 1.34,
                        "interpolation": "Smoothstep" if smoothstep else "Linear"
                    })
        '''
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
        # print("network architecture")
        # print(dims)
        
        self.num_layers = len(dims)
        self.skip_in = skip_in
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # Geometry initalization for compositional scene, bg SDF sign: inside + outside -, fg SDF sign: outside + inside -
                    # The 0 index is the background SDF, the rest are the object SDFs
                    # background SDF with postive value inside and nagative value outside
                    torch.nn.init.normal_(lin.weight[:1, :], mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[:1], bias)
                    # inner objects with SDF initial with negative value inside and positive value outside, ~0.5 radius of background
                    torch.nn.init.normal_(lin.weight[1:,:], mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias[1:], -0.5*bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.cache_sdf = None

        self.pool = nn.MaxPool1d(self.d_out, return_indices=True)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.use_grid_feature:
            # normalize point range as encoding assume points are in [-1, 1]
            # assert torch.max(input / self.divide_factor)<1 and torch.min(input / self.divide_factor)>-1, 'range out of [-1, 1], max: {}, min: {}'.format(torch.max(input / self.divide_factor),  torch.min(input / self.divide_factor))
            feature = self.encoding(input / self.divide_factor)
        else:
            feature = torch.zeros_like(input[:, :1].repeat(1, self.grid_feature_dim))

        if self.embed_fn is not None:
            embed = self.embed_fn(input)
            input = torch.cat((embed, feature), dim=-1)
        else:
            input = torch.cat((input, feature), dim=-1)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:self.d_out]
        d_output = torch.ones_like(y[:, :1], requires_grad=False, device=y.device)
        f = lambda v: torch.autograd.grad(outputs=y,
                    inputs=x,
                    grad_outputs=v.repeat(y.shape[0], 1),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
        
        N = torch.eye(y.shape[1], requires_grad=False).to(y.device)
        
        # start_time = time.time()
        if self.use_grid_feature: # using hashing grid feature, cannot support vmap now
            g = torch.cat([torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=idx.repeat(y.shape[0], 1),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0] for idx in N.unbind()])
        # torch.cuda.synchronize()
        # print("time for computing gradient by for loop: ", time.time() - start_time, "s")
                
        # using vmap for batched gradient computation, if not using grid feature (pure MLP)
        else:
            g = vmap(f, in_dims=1)(N).reshape(-1, 3)
        
        # add the gradient of scene sdf
        # sdf = -self.pool(-y.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-y.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        g_min_sdf = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        g = torch.cat([g, g_min_sdf])
        return g

    def get_outputs(self, x, beta=None):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        # if self.sdf_bounding_sphere > 0.0:
        #     sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
        #     sdf_raw = torch.minimum(sdf_raw, sphere_sdf.expand(sdf_raw.shape))

        if beta == None:
            semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        else:
            # change semantic to the gradianct of density
            semantic = 1/beta * (0.5 + 0.5 * sdf_raw.sign() * torch.expm1(-sdf_raw.abs() / beta))
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1) # get the minium value of sdf
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        feature_vectors = output[:, self.d_out:]

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, sdf_raw

    def get_sdf_vals(self, x):
        sdf_raw = self.forward(x)[:,:self.d_out]
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        return sdf

    def get_sdf_raw(self, x):
        return self.forward(x)[:, :self.d_out]
    
    def get_object_sdf_vals(self, x, idx):
        sdf = self.forward(x)[:, idx]
        return sdf
    
    def get_sdf_vals_and_sdfs(self, x):
        sdf = self.forward(x)[:,:self.d_out]
        sdf_raw = sdf
        # sdf = -self.pool(-sdf.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-sdf.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        return sdf, sdf_raw

    def get_specific_outputs(self, x, idx):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]
        semantic = self.sigmoid * torch.sigmoid(-self.sigmoid * sdf_raw)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)

        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients, semantic, sdf_raw[:, idx]
    
    def get_shift_sdf_raw(self, x):
        sdf_raw = self.forward(x)[:, :self.d_out]
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)

        # shift raw sdf
        pos_min_sdf = -sdf          # other object sdf must bigger than -sdf
        pos_min_sdf_expand = pos_min_sdf.expand_as(sdf_raw)
        shift_mask = (sdf < 0)
        shift_mask_expand = shift_mask.expand_as(sdf_raw)

        shift_sdf_raw = torch.where(shift_mask_expand, torch.max(sdf_raw, pos_min_sdf_expand), sdf_raw)
        shift_sdf_raw[torch.arange(indices.size(0)), indices.squeeze()] = sdf.squeeze()

        return shift_sdf_raw

    def get_outputs_and_indices(self, x):
        x.requires_grad_(True)
        output = self.forward(x)
        sdf_raw = output[:,:self.d_out]

        # if self.sigmoid_optim:
        #     sigmoid_value = torch.exp(self.sigmoid_basis)
        # else:
        sigmoid_value = self.sigmoid

        semantic = sigmoid_value * torch.sigmoid(-sigmoid_value * sdf_raw)
        # sdf = -self.pool(-sdf_raw.unsqueeze(1)).squeeze(-1) # get the minium value of sdf
        sdf, indices = self.pool(-sdf_raw.unsqueeze(1))
        sdf = -sdf.squeeze(-1)
        indices = indices.squeeze(-1)
        feature_vectors = output[:, self.d_out:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return sdf, feature_vectors, gradients, semantic, sdf_raw, indices

    def mlp_parameters(self):
        parameters = []
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            parameters += list(lin.parameters())
        return parameters

    def grid_parameters(self, verbose=False):
        if verbose:
            print("[INFO]: grid parameters", len(list(self.encoding.parameters())))
            for p in self.encoding.parameters():
                print(p.shape)
        return self.encoding.parameters()


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            per_image_code = False
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.per_image_code = per_image_code
        if self.per_image_code:
            # nerf in the wild parameter
            # parameters
            # maximum 1024 images
            self.embeddings = nn.Parameter(torch.empty(1024, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32

        # print("rendering network architecture:")
        # print(dims)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        if self.per_image_code:
            image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1)
            rendering_input = torch.cat([rendering_input, image_code], dim=-1)
            
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
        
        # x.shape [N_rays*pts_per_ray, 7]
        color = self.sigmoid(x[:, :3])
        uncertainty = self.relu(x[:, 3:])
        return color, uncertainty[:, 0], uncertainty[:, 1:]


class PhyReconNetwork(nn.Module):
    def __init__(self,
                  conf,
                  plots_dir=None,
                  physics_sampling_begin_iter=0,
                  physics_loss_begin_iter=10,
                  z_floor=None,
                  ft_folder=None,
                  ):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()
        self.use_bg_reg = conf.get_bool('use_bg_reg', default=False)
        self.render_bg_iter = conf.get_int('render_bg_iter', default=10)
        self.begin_bg_surface_obj_reg_iter = conf.get_int('begin_bg_surface_obj_reg_iter', default=80000)

        Grid_MLP = conf.get_bool('Grid_MLP', default=False)
        self.Grid_MLP = Grid_MLP
        if Grid_MLP: 
            self.implicit_network = ObjectImplicitNetworkGrid(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))    
        else:
            self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        
        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        
        self.num_semantic = conf.get_int('implicit_network.d_out')

        self.uncertainty_field = DenseGrid(fill_data=0.0, resolution=512)               # physical uncertainty grid
        
        self.plots_dir = plots_dir
        self.ft_folder = ft_folder              # if continue training
        self.pc_boundary = conf.get_float('simulator.pc_boundary', default=1.0)

        # init physical simulator
        self.adaptive_spacing = 0.002
        self.pc_resolution = conf.get_int('simulator.pc_resolution', default=64)
        self.mc_level = conf.get_float('simulator.mc_level', default=0.0)
        self.use_sleepstop = conf.get_bool('simulator.use_sleepstop', default=True)
        self.use_simulator_thresh = conf.get_bool('simulator.use_simulator_thresh', default=False)
        self.sim_obj_id = conf.get_int('simulator.sim_obj_id', default=0)
        self.save_pc_path = os.path.join(self.plots_dir, 'simulator_pc')
        os.makedirs(self.save_pc_path, exist_ok=True)

        print(f'pc_resolution: {self.pc_resolution}, pc_boundary: {self.pc_boundary}, use_sleepstop: {self.use_sleepstop}, use_simulator_thresh: {self.use_simulator_thresh}, sim_obj_id: {self.sim_obj_id}')
        
        self.n_max = 100000
        self.sim = Simulator(self.use_simulator_thresh, self.use_sleepstop, n_max=self.n_max)

        # set max simulation time for each object, only use for physical loss
        self.max_sim_time = 100
        self.obj_sim_time_dict = {}

        # simulator floor
        self.z_floor = z_floor          # get from gt mesh
        self.floor_resolution = 64
        self.floor_delta = 0.1

        self.obj_bbox_dic = {}
        self.sim_idx_iter = 0
        self.sim_obj_idx_list = [x for x in range(1, self.num_semantic)]        # exclude background, self.num_semantic = obj_num + 1
        self.stopthreshold = 0.01           # 1cm

        self.physics_sampling_begin_iter = physics_sampling_begin_iter
        self.physics_loss_begin_iter = physics_loss_begin_iter

        assert self.physics_sampling_begin_iter < self.physics_loss_begin_iter, 'physics_sampling_begin_iter should be smaller than physics_loss_begin_iter'



    def forward(self, input, indices, iter_step=-1):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        
        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        depth_scale = ray_dirs_tmp[0, :, 2:]
        
        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        beta_cur = self.density.get_beta()
        sdf, feature_vectors, gradients, semantic, sdf_raw = self.implicit_network.get_outputs(points_flat, beta=None)

        rgb_flat, depth_un, normal_un = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, indices)
        rgb = rgb_flat.reshape(-1, N_samples, 3)
        depth_un = depth_un.reshape(-1, N_samples, 1)
        normal_un = normal_un.reshape(-1, N_samples, 3)

        phy_un = self.uncertainty_field(points_flat)               # [ray_num*N_samples, 1]
        phy_un = phy_un.reshape(-1, N_samples, 1)

        semantic = semantic.reshape(-1, N_samples, self.num_semantic)
        weights, transmittance, dists = self.volume_rendering(z_vals, sdf)

        # rendering the occlusion-awared object opacity
        object_opacity = self.occlusion_opacity(z_vals, transmittance, dists, sdf_raw).sum(-1).transpose(0, 1)


        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        semantic_values = torch.sum(weights.unsqueeze(-1)*semantic, 1)
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values

        # volume rendering for uncertainty values
        depth_un_values = torch.sum(weights.unsqueeze(-1) * depth_un, 1)                        # [ray_num, 1]
        normal_un_values = torch.sum(weights.unsqueeze(-1) * normal_un, 1)                      # [ray_num, 3]
        normal_un_values = torch.sum(normal_un_values, 1, keepdim=True)                         # [ray_num, 1]
        
        phy_un_values = torch.sum(weights.unsqueeze(-1) * phy_un, 1)                            # [ray_num, 1]

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb':rgb,
            'semantic_values': semantic_values, # here semantic value calculated as in ObjectSDF
            'object_opacity': object_opacity, 
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'depth_un_values': depth_un_values,
            'normal_un_values': normal_un_values,
            'phy_un_values': phy_un_values,
        }

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels 
            
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            # add some neighbour points as unisurf
            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01   
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)
            
            grad_theta = self.implicit_network.gradient(eikonal_points)

            sample_sdf = self.implicit_network.get_sdf_raw(eikonal_points)
            sdf_value = self.implicit_network.get_sdf_vals(eikonal_points)
            output['sample_sdf'] = sample_sdf
            output['sample_minsdf'] = sdf_value
            
            # split gradient to eikonal points and heighbour ponits
            output['grad_theta'] = grad_theta[:grad_theta.shape[0]//2]
            output['grad_theta_nei'] = grad_theta[grad_theta.shape[0]//2:]

            # # use bg surface for regularization
            # if iter_step > self.begin_bg_surface_obj_reg_iter:                   # start to use bg surface for regularization
            #     surf_bg_z_vals = self.ray_sampler.ray_marching_surface(self, ray_dirs, cam_loc, idx=0) # [N, 1]
            #     # the sdf value of objects that behind bg surface
            #     bg_surf_back_mask = z_vals > surf_bg_z_vals # [1024, 98]
            #     sdf_all = sdf_raw.reshape(z_vals.shape[0], z_vals.shape[1], -1)
            #     objs_sdfs_bg_back = sdf_all[bg_surf_back_mask][..., 1:]  # [K, num_semantics-1]

            #     output['obj_sdfs_behind_bg'] = objs_sdfs_bg_back
        
        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        
        # transform to local coordinate system
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()
        
        output['normal_map'] = normal_map

        # only for render the background depth and normal
        iter_check = iter_step % self.render_bg_iter
        if self.use_bg_reg and iter_check == 0:

            # construct patch uv
            patch_size = 32
            n_patches = 1

            x0 = np.random.randint(0, 384 - patch_size + 1, size=(n_patches, 1, 1))         # NOTE: fix image resolution as 384
            y0 = np.random.randint(0, 384 - patch_size + 1, size=(n_patches, 1, 1))
            xy0 = np.concatenate([x0, y0], axis=-1)
            patch_idx = xy0 + np.stack(np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),axis=-1).reshape(1, -1, 2)
            uv0 = torch.from_numpy(patch_idx).float().reshape(1, -1, 2).float().cuda()
            ray_dirs0, cam_loc0 = rend_util.get_camera_params(uv0, pose, intrinsics)

            # we should use unnormalized ray direction for depth
            ray_dirs0_tmp, _ = rend_util.get_camera_params(uv0, torch.eye(4).to(pose.device)[None], intrinsics)
            depth_scale0 = ray_dirs0_tmp[0, :, 2:]
            
            batch_size, num_pixels, _ = ray_dirs0.shape

            cam_loc0 = cam_loc0.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
            ray_dirs0 = ray_dirs0.reshape(-1, 3)

            bg_z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs0, cam_loc0, self, idx=0)
            N_samples_bg = bg_z_vals.shape[1]

            bg_points = cam_loc0.unsqueeze(1) + bg_z_vals.unsqueeze(2) * ray_dirs0.unsqueeze(1)
            bg_points_flat = bg_points.reshape(-1, 3)
            scene_sdf, _, bg_gradients, scene_semantic, bg_sdf = self.implicit_network.get_specific_outputs(bg_points_flat, 0)
            
            bg_weight, _, _ = self.volume_rendering(bg_z_vals, bg_sdf)

            # NOTE: semantic should use scene sdf for volume rendering
            scene_weight, _, _ = self.volume_rendering(bg_z_vals, scene_sdf)
            scene_semantic = scene_semantic.reshape(-1, N_samples_bg, self.num_semantic)
            bg_semantic_value = torch.sum(scene_weight.unsqueeze(-1)*scene_semantic, 1)
            bg_mask = torch.argmax(bg_semantic_value, dim=-1, keepdim=True)
            output['bg_mask'] = bg_mask

            bg_depth_values = torch.sum(bg_weight * bg_z_vals, 1, keepdims=True) / (bg_weight.sum(dim=1, keepdims=True) +1e-8)
            bg_depth_values = depth_scale0 * bg_depth_values 
            output['bg_depth_values'] = bg_depth_values

            # compute bg normal map
            bg_normals = bg_gradients / (bg_gradients.norm(2, -1, keepdim=True) + 1e-6)
            bg_normals = bg_normals.reshape(-1, N_samples_bg, 3)
            bg_normal_map = torch.sum(bg_weight.unsqueeze(-1) * bg_normals, 1)
            bg_normal_map = rot @ bg_normal_map.permute(1, 0)
            bg_normal_map = bg_normal_map.permute(1, 0).contiguous()
            output['bg_normal_map'] = bg_normal_map

        use_physical_simulator = False
        if self.training and self.z_floor is not None:
            
            if iter_step > self.physics_loss_begin_iter:            # use physical simulator every iter
                use_physical_simulator = True
            elif iter_step > self.physics_sampling_begin_iter:      # simulate every object once per epoch
                if self.sim_idx_iter < self.num_semantic - 1:       # self.num_semantic = obj_num + 1
                    use_physical_simulator = True
                else:
                    use_physical_simulator = False
            else:
                use_physical_simulator = False

            if iter_step < self.physics_loss_begin_iter:
                use_phyloss = False
            else:
                use_phyloss = True
        
        if use_physical_simulator:
            # get surface points
            pc_grid_boundary = [-self.pc_boundary, self.pc_boundary]

            list_idx = self.sim_idx_iter % (len(self.sim_obj_idx_list))
            obj_idx = self.sim_obj_idx_list[list_idx]

            self.sim_idx_iter = self.sim_idx_iter + 1

            print(f'*********** sim_obj_idx_list = {self.sim_obj_idx_list} ***********')
            print(f'************** sim obj_idx = {obj_idx} **************')

            # get object bbox
            if not os.path.exists(os.path.join(self.plots_dir, 'bbox')):        # use object bbox
                # assert False, "Please provide object bbox, set use_physical_simulator after providing object bbox"
                import shutil       # temp solution
                src_bbox_path = os.path.join(self.ft_folder, 'plots', 'bbox')
                dst_bbox_path = os.path.join(self.plots_dir, 'bbox')
                shutil.copytree(src_bbox_path, dst_bbox_path)

            obj_list = os.listdir(os.path.join(self.plots_dir, 'bbox'))
            obj_list.sort(key=lambda x: int((x.split('.')[0]).split('_')[1]))
            for obj_file_name in obj_list:
                obj_bbox_idx = int((obj_file_name.split('.')[0]).split('_')[1])
                with open(os.path.join(self.plots_dir, 'bbox', obj_file_name), 'r') as f:
                    bbox = json.load(f)
                # expand bbox
                self.obj_bbox_dic[obj_bbox_idx] = bbox

            raw_obj_pc, _ = self.get_surface_points_mc(resolution=self.pc_resolution, grid_boundary=pc_grid_boundary, obj_idx=obj_idx)
            raw_obj_pc = raw_obj_pc.double()        # taichi use float64
            if raw_obj_pc.shape[0] > self.n_max:
                indices = torch.randperm(raw_obj_pc.size(0))[:self.n_max]
                obj_pc = raw_obj_pc[indices]
            else:
                obj_pc = raw_obj_pc

            # import trimesh
            # export_obj_pc = obj_pc.cpu().detach().numpy()
            # export_obj_pc = trimesh.PointCloud(export_obj_pc)
            # export_obj_pc.export(f'{obj_idx}_mc_pc.ply')

            print('object_pc.shape: ', obj_pc.shape)
            if obj_pc.shape[0] == 0:
                print('*************** obj_pc error ***************')
            else:
                # for cut floor_pc, use obj_pc 3D bbox
                x_min, x_max = torch.min(obj_pc[:, 0]), torch.max(obj_pc[:, 0])
                y_min, y_max = torch.min(obj_pc[:, 1]), torch.max(obj_pc[:, 1])
                # expand floor pc x and y range
                x_min, x_max = x_min - self.floor_delta, x_max + self.floor_delta
                y_min, y_max = y_min - self.floor_delta, y_max + self.floor_delta
                
                x = torch.linspace(x_min.item(), x_max.item(), self.floor_resolution, device=obj_pc.device)
                y = torch.linspace(y_min.item(), y_max.item(), self.floor_resolution, device=obj_pc.device)
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                real_floor_pc = torch.zeros((self.floor_resolution, self.floor_resolution, 3), device=obj_pc.device)
                real_floor_pc[:, :, 0] = xx
                real_floor_pc[:, :, 1] = yy
                real_floor_pc[:, :, 2] = self.z_floor
                real_floor_pc = real_floor_pc.reshape(-1, 3).double()

                # simulator
                obj_pc_after, collision_mask = self.sim(obj_pc, real_floor_pc, iter_step, obj_idx)

                # NOTE: update physical uncertainty field
                contact_points_before = obj_pc[collision_mask].clone()
                contact_points_after = obj_pc_after[collision_mask].clone()
                contact_points = self.interpolate_points(contact_points_before, contact_points_after, M=32).reshape(-1, 3)
                contact_points = contact_points.detach().float()

                sampling_factor = self.uncertainty_field(contact_points)
                sampling_loss = self.uncertainty_field.get_loss(sampling_factor)
                output['sampling_loss'] = sampling_loss

                if use_phyloss:
                    # if z movement is too small, object is nearly stable, stop simulation
                    if torch.abs(obj_pc_after[:, 2] - obj_pc[:, 2]).max() < self.stopthreshold and torch.abs(obj_pc_after[:, 2] - obj_pc[:, 2]).max() > 1e-5:
                        self.sim.sleep_status = True
                        print(f'************** {obj_idx} object is nearly stable, stop simulation **************')

                    # if object sim time is larger than max sim time, stop simulation
                    if obj_idx not in self.obj_sim_time_dict:
                        self.obj_sim_time_dict[obj_idx] = 1

                    else:
                        if self.obj_sim_time_dict[obj_idx] > self.max_sim_time:
                            self.sim.sleep_status = True
                            print(f'************** {obj_idx} object sim time is larger than max sim time, stop simulation **************')
                        else:
                            self.obj_sim_time_dict[obj_idx] += 1

                    sim_rate = self.obj_sim_time_dict[obj_idx] * 1.0 / (self.max_sim_time * 1.0)

                    if self.sim.sleep_status:
                        self.sim_obj_idx_list.remove(obj_idx)
                        output['sleep_obj_idx'] = obj_idx
                        output['sleep_pc'] = obj_pc
                        output['sleep_pc_floor'] = real_floor_pc
                    else:
                        print('obj_pc: ', obj_pc[:10])
                        print('obj_pc_after: ', obj_pc_after[:10])
                        print('sim_rate: ', sim_rate)

                        output['obj_pc'] = obj_pc
                        output['obj_pc_after'] = obj_pc_after
                        output['sim_rate'] = sim_rate
                else:
                    self.sim.sleep_status = False           # sleep stop only for phyloss

            if use_phyloss:
                output['sim_obj_idx_list'] = self.sim_obj_idx_list

        
        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights, transmittance, dists

    def occlusion_opacity(self, z_vals, transmittance, dists, sdf_raw):
        obj_density = self.density(sdf_raw).transpose(0, 1).reshape(-1, dists.shape[0], dists.shape[1]) # [#object, #ray, #sample points]       
        free_energy = dists * obj_density
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        object_weight = alpha * transmittance
        return object_weight
    
    def get_surface_points_mc(self, resolution, grid_boundary, obj_idx):
        # get coarse surface points
        with torch.no_grad():
            coarse_point_cloud = get_coarse_surface_points_mc(
                get_outputs=lambda x: self.implicit_network.get_sdf_vals_and_sdfs(x),
                resolution=resolution,
                obj_bbox=self.obj_bbox_dic[obj_idx],
                obj_idx=obj_idx,
                adaptive_spacing=self.adaptive_spacing,
                mc_level = self.mc_level
            )

        # ### visualize object surface point cloud
        # import trimesh
        # for idx in range(self.implicit_network.d_out):
        #     if idx == 0:
        #         continue
        #     export_obj_pc = coarse_point_cloud_dic[idx]
        #     export_obj_pc = export_obj_pc.cpu().detach().numpy()
        #     export_obj_pc = trimesh.PointCloud(export_obj_pc)
        #     export_obj_pc.export(f'{idx}_mc_pc.ply')

        # NOTE: only simulate one object in a simulation
        fine_point_cloud, fine_gradients = get_fine_surface_points_mc(
            get_outputs=lambda x: self.implicit_network.get_outputs_and_indices(x),
            points=coarse_point_cloud.detach(),            # NOTE: detach here
            obj_idx=obj_idx
        )

        return fine_point_cloud, fine_gradients
    
    def get_parameters(self):
        # delete the parameters in uncertainty field
        params = []
        for name, param in self.named_parameters():
            if 'uncertainty_field' not in name:
                params.append(param)
        
        return params
    
    def interpolate_points(self, contact_pc_before, contact_pc_after, M=64):
        """
        Interpolate points between two point clouds.
        Args:
        contact_pc_before (torch.Tensor): Tensor of shape [N, 3]
        contact_pc_after (torch.Tensor): Tensor of shape [N, 3]
        M (int): The number of total points.

        Returns:
        torch.Tensor: Interpolated points of shape [N, M, 3].
        """
        # Calculate the step for each dimension
        steps = (contact_pc_after - contact_pc_before).unsqueeze(1) / (M-1)

        # Create a range of steps for interpolation
        step_range = torch.arange(0, M, device=contact_pc_before.device).view(1, M, 1)

        # Interpolate points
        interpolated_points = contact_pc_before.unsqueeze(1) + steps * step_range

        return interpolated_points
