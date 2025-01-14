import abc
from tkinter.messagebox import NO
import torch

from utils import rend_util

class RaySampler(metaclass=abc.ABCMeta):
    def __init__(self,near, far):
        self.near = near
        self.far = far

    @abc.abstractmethod
    def get_z_vals(self, ray_dirs, cam_loc, model):
        pass

class UniformSampler(RaySampler):
    def __init__(self, scene_bounding_sphere, near, N_samples, take_sphere_intersection=False, far=-1):
        #super().__init__(near, 2.0 * scene_bounding_sphere if far == -1 else far)  # default far is 2*R
        super().__init__(near, 2.0 * scene_bounding_sphere * 1.75 if far == -1 else far)  # default far is 2*R
        self.N_samples = N_samples
        self.scene_bounding_sphere = scene_bounding_sphere
        self.take_sphere_intersection = take_sphere_intersection

    # dtu and bmvs
    def get_z_vals_dtu_bmvs(self, ray_dirs, cam_loc, model):
        if not self.take_sphere_intersection:
            near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0], 1).cuda()
        else:
            sphere_intersections = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()
            far = sphere_intersections[:,1:]

        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()
        z_vals = near * (1. - t_vals) + far * (t_vals)

        if model.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda()

            z_vals = lower + (upper - lower) * t_rand

        return z_vals, near, far
    
    def near_far_from_cube(self, rays_o, rays_d, bound):
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=self.near)
        far = torch.clamp(far, max=self.far)
        return near, far

    # currently this is used for replica scannet and T&T
    def get_z_vals(self, ray_dirs, cam_loc, model):
        if not self.take_sphere_intersection:
            near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0], 1).cuda()
        else:
            _, far = self.near_far_from_cube(cam_loc, ray_dirs, bound=self.scene_bounding_sphere)
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()
        
        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()
        z_vals = near * (1. - t_vals) + far * (t_vals)

        if model.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda()

            z_vals = lower + (upper - lower) * t_rand

        return z_vals, near, far    


class ErrorBoundSampler(RaySampler):
    def __init__(self, scene_bounding_sphere, near, N_samples, N_samples_eval, N_samples_extra,
                 eps, beta_iters, max_total_iters,
                 inverse_sphere_bg=False, N_samples_inverse_sphere=0, add_tiny=1.0e-6):
        #super().__init__(near, 2.0 * scene_bounding_sphere)
        super().__init__(near, 2.0 * scene_bounding_sphere * 1.75)
        
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.take_sphere_intersection = True
        self.uniform_sampler = UniformSampler(scene_bounding_sphere, near, N_samples_eval, take_sphere_intersection=True) # replica scannet and T&T courtroom
        #self.uniform_sampler = UniformSampler(scene_bounding_sphere, near, N_samples_eval, take_sphere_intersection=inverse_sphere_bg)  # dtu and bmvs

        self.N_samples_extra = N_samples_extra

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

        self.inverse_sphere_bg = inverse_sphere_bg
        if inverse_sphere_bg:
            self.inverse_sphere_sampler = UniformSampler(1.0, 0.0, N_samples_inverse_sphere, False, far=1.0)

    def get_z_vals(self, ray_dirs, cam_loc, model, idx=None):
        beta0 = model.density.get_beta().detach()

        # Start with uniform sampling
        z_vals, near, far = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)
        samples, samples_idx = z_vals, None

        # Get maximum beta from the upper bound (Lemma 2)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists ** 2.).sum(-1)
        beta = torch.sqrt(bound)

        total_iters, not_converge = 0, True

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)

            # Calculating the SDF only for the new sampled points
            with torch.no_grad():
                if idx==None:
                    samples_sdf = model.implicit_network.get_sdf_vals(points_flat)
                else:
                    samples_sdf = model.implicit_network.get_object_sdf_vals(points_flat, idx).unsqueeze(-1)
            if samples_idx is not None:
                sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                       samples_sdf.reshape(-1, samples.shape[1])], -1)
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf


            # Calculating the bound d* (Theorem 1)
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1).cuda()
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign


            # Updating beta using line search
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.
                curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max

            # Upsample more points
            density = model.density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

            dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)
            free_energy = dists * density
            shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance  # probability of the ray hits something here

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                ''' Sample more points proportional to the current error bound'''

                N = self.N_samples_eval

                bins = z_vals
                error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * (dists[:,:-1] ** 2.) / (4 * beta.unsqueeze(-1) ** 2)
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (torch.clamp(torch.exp(error_integral),max=1.e6) - 1.0) * transmittance[:,:-1]

                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

            else:
                ''' Sample the final sample set to be used in the volume rendering integral '''

                N = self.N_samples

                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))


            # Invert CDF
            if (not_converge and total_iters < self.max_total_iters) or (not model.training):
                u = torch.linspace(0., 1., steps=N).cuda().unsqueeze(0).repeat(cdf.shape[0], 1)
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N]).cuda()
            u = u.contiguous()

            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = (cdf_g[..., 1] - cdf_g[..., 0])
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])


            # Adding samples if we not converged
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)


        z_samples = samples
        #TODO Use near and far from intersection
        near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0],1).cuda()
        if self.inverse_sphere_bg: # if inverse sphere then need to add the far sphere intersection
            far = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:,1:]

        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1]-1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:,sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        # add some of the near surface points
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],)).cuda()
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))

        if self.inverse_sphere_bg:
            z_vals_inverse_sphere, _, _ = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, model)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1./self.scene_bounding_sphere)
            z_vals = (z_vals, z_vals_inverse_sphere)

        return z_vals, z_samples_eik

    def get_error_bound(self, beta, model, sdf, z_vals, dists, d_star):
        density = model.density(sdf.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), dists * density[:, :-1]], dim=-1)
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists ** 2.) / (4 * beta ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(-integral_estimation[:, :-1])

        return bound_opacity.max(-1)[0]
    
    def near_far_from_cube(self, rays_o, rays_d, bound):
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=self.near)
        far = torch.clamp(far, max=self.far)
        return near, far
        
    def secant(self, model, f_low, f_high, d_low, d_high, n_secant_steps,
                          ray0_masked, ray_direction_masked, tau, idx=0):
        ''' Runs the secant method for interval [d_low, d_high].
        Args:
            d_low (tensor): start values for the interval, actually for sdf this should be > 0
            d_high (tensor): end values for the interval, actually for sdf this should be < 0
            n_secant_steps (int): number of steps
            ray0_masked (tensor): masked ray start points
            ray_direction_masked (tensor): masked ray direction vectors
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code c
            tau (float): threshold value in logits
        '''
        # this guarantee a valid depth no matter f_low > 0 or f_low < 0
        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        for i in range(n_secant_steps):
            p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
            with torch.no_grad():
                if idx >= 0:
                    f_mid = (model.implicit_network.get_object_sdf_vals(p_mid, idx).unsqueeze(-1))[...,0] - tau
                else:
                    f_mid = model.implicit_network.get_sdf_vals(p_mid)[...,0] - tau
            
            # ind_low = f_mid < 0
            # ind_low = ind_low
            ind_low = torch.sign(f_mid * f_low)
            ind_low[ind_low <= 0] = 0
            ind_low = ind_low.long()

            if ind_low.sum() > 0:
                d_low[ind_low == 1] = d_pred[ind_low == 1]
                f_low[ind_low == 1] = f_mid[ind_low == 1]
            if (ind_low == 0).sum() > 0:
                d_high[ind_low == 0] = d_pred[ind_low == 0]
                f_high[ind_low == 0] = f_mid[ind_low == 0]

            d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        return d_pred 
    
    def ray_marching_surface(self, model, ray_dirs, cam_loc, idx=0, n_steps=128, n_secant_steps=8, max_points=3500000):
        '''
        ray_dirs: [N, 3]
        According to indoor scene initialization, sdf should be first > 0 then < 0
        '''
        ray0 = cam_loc.unsqueeze(0)
        ray_direction = ray_dirs.unsqueeze(0)

        batch_size, n_pts, D = ray0.shape
        device = ray_dirs.device

        if not self.take_sphere_intersection:
            near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0], 1).cuda()
        else:
            _, far = self.near_far_from_cube(cam_loc, ray_dirs, bound=self.scene_bounding_sphere)
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()

        near = near.reshape(batch_size, n_pts, 1, 1).to(device)
        far = far.reshape(batch_size, n_pts, 1, 1).to(device)

        d_proposal = torch.linspace(0, 1, steps=n_steps).view(1, 1, n_steps, 1).to(device)
        d_proposal = near * (1. - d_proposal) + far * d_proposal

        # [1, n_pts, n_steps, 3]
        p_proposal = ray0.unsqueeze(2).repeat(1, 1, n_steps, 1) + ray_direction.unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal
        with torch.no_grad():
            if idx >= 0:
                val = torch.cat([(
                    model.implicit_network.get_object_sdf_vals(p_split.view(-1, 3), idx)).unsqueeze(-1)
                    for p_split in torch.split(
                        p_proposal.reshape(batch_size, -1, 3),
                        int(max_points / batch_size), dim=1)], dim=1).view(
                            batch_size, -1, n_steps)
            else:   # use the minimum value to ray_marching the surface
                val = torch.cat([(
                    model.implicit_network.get_sdf_vals(p_split.view(-1, 3)))
                    for p_split in torch.split(
                        p_proposal.reshape(batch_size, -1, 3),
                        int(max_points / batch_size), dim=1)], dim=1).view(
                            batch_size, -1, n_steps)
                
        # val: [1, n_pts, n_steps]
        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = val[:, :, 0] > 0

        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]),
                                 torch.ones(batch_size, n_pts, 1).to(device)],
                                dim=-1)
        cost_matrix = sign_matrix * torch.arange(
            n_steps, 0, -1).float().to(device)

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_pos_to_neg = val[torch.arange(batch_size).unsqueeze(-1),
                              torch.arange(n_pts).unsqueeze(-0), indices] > 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_pos_to_neg & mask_0_not_occupied

        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        n = batch_size * n_pts
        d_low = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
            batch_size, n_pts)[mask]
        indices = torch.clamp(indices + 1, max=n_steps-1)
        d_high = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_high = val.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]

        ray0_masked = ray0[mask]
        ray_direction_masked = ray_direction[mask]

        # Apply surface depth refinement step (e.g. Secant method)
        d_pred = self.secant(
            model, f_low, f_high, d_low, d_high, n_secant_steps, ray0_masked,
            ray_direction_masked, tau=0., idx=idx)

        # for sanity
        d_pred_out = torch.ones(batch_size, n_pts).to(device)
        d_pred_out[mask] = d_pred
        d_pred_out[mask == 0] = far.reshape(1, -1)[mask == 0]
        d_pred_out[mask_0_not_occupied == 0] = far.reshape(1, -1)[mask_0_not_occupied == 0]

        d_pred_out = d_pred_out.reshape(-1, 1)
        return d_pred_out
