import torch
from torch import nn
import utils.general as utils
import math
import torch.nn.functional as F

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


# def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

#     M = torch.sum(mask, (1, 2))
#     res = prediction - target
#     image_loss = torch.sum(mask * res * res, (1, 2))

#     return reduction(image_loss, 2 * M)

def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    res = prediction - target
    image_loss = mask * res * res

    return image_loss


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        # self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        # self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        # if self.__alpha > 0:
        #     total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy
    

class MonoSDFLoss(nn.Module):
    def __init__(self, rgb_loss, 
                 eikonal_weight, 
                 smooth_weight = 0.005,
                 depth_weight = 0.1,
                 normal_l1_weight = 0.05,
                 normal_cos_weight = 0.05,
                 uncertainty_begin_iter = 20000000,           # set a large number to avoid using uncertainty
                 depth_type = 'marigold',
                 phy_un_weight = 50,
                 end_step = -1):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.smooth_weight = smooth_weight
        self.depth_weight = depth_weight
        self.normal_l1_weight = normal_l1_weight
        self.normal_cos_weight = normal_cos_weight
        self.uncertainty_begin_iter = uncertainty_begin_iter
        self.depth_type = depth_type
        self.phy_un_weight = phy_un_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')

        self.use_uncertainty = True
        
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        
        # print(f"using weight for loss RGB_1.0 EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}")
        
        self.step = 0
        self.end_step = end_step

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_smooth_loss(self,model_outputs):
        # smoothness loss as unisurf
        g1 = model_outputs['grad_theta']
        g2 = model_outputs['grad_theta_nei']
        
        normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        smooth_loss =  torch.norm(normals_1 - normals_2, dim=-1).mean()
        return smooth_loss
    
    def get_depth_loss(self, depth_pred, depth_gt, mask, depth_un, phy_un):
        # TODO remove hard-coded scaling for depth

        if self.depth_type == 'marigold':
            depth_loss = self.depth_loss(depth_pred.reshape(1, -1, 32), (depth_gt * 4.0 + 0.5).reshape(1, -1, 32), mask.reshape(1, -1, 32))
        elif self.depth_type == 'omnidata':
            depth_loss = self.depth_loss(depth_pred.reshape(1, -1, 32), (depth_gt * 50.0 + 0.5).reshape(1, -1, 32), mask.reshape(1, -1, 32))
        else:
            raise ValueError(f'{self.depth_type} not implement')

        if self.use_uncertainty:
            # apply depth_un
            depth_un = depth_un.reshape(1, -1, 32)
            phy_un = phy_un.reshape(1, -1, 32)
            depth_loss = (torch.log(torch.abs(depth_un + 1.0)) + depth_loss / (torch.abs(depth_un) + 0.05)) / (self.phy_un_weight * phy_un + 1.0)
        
        depth_loss = torch.sum(depth_loss) / (torch.sum(mask) + 1e-6)

        return depth_loss
        
    def get_normal_loss(self, normal_pred, normal_gt, normal_un_values, phy_un):
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1)
        cos = (1. - torch.sum(normal_pred * normal_gt, dim = -1))

        if self.use_uncertainty:
            # apply normal_un_values
            normal_un_values = normal_un_values.reshape(l1.shape)                         # [1, ray_num]
            phy_un = phy_un.reshape(l1.shape)
            l1 = (0.5 * torch.log(torch.abs(normal_un_values) + 1.0) + l1 / (torch.abs(normal_un_values) + 0.05)) / (self.phy_un_weight * phy_un + 1.0)
            cos = (0.5 * torch.log(torch.abs(normal_un_values) + 1.0) + cos / (torch.abs(normal_un_values) + 0.05)) / (self.phy_un_weight * phy_un + 1.0)
        
        l1 = torch.mean(l1)
        cos = torch.mean(cos)

        return l1, cos
        
    def forward(self, model_outputs, ground_truth):
        
        # import pdb; pdb.set_trace()
        rgb_gt = ground_truth['rgb'].cuda()
        # monocular depth and normal
        depth_gt = ground_truth['depth'].cuda()
        normal_gt = ground_truth['normal'].cuda()
        
        depth_pred = model_outputs['depth_values']
        normal_pred = model_outputs['normal_map'][None]

        depth_un_values = model_outputs['depth_un_values']
        normal_un_values = model_outputs['normal_un_values']
        phy_un_values = model_outputs['phy_un_values'].detach()
        
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        # only supervised the foreground normal
        mask = ((model_outputs['sdf'] > 0.).any(dim=-1) & (model_outputs['sdf'] < 0.).any(dim=-1))[None, :, None]
        # combine with GT
        mask = (ground_truth['mask'] > 0.5).cuda() & mask

        depth_loss = self.get_depth_loss(depth_pred, depth_gt, mask, depth_un_values, phy_un_values) if self.depth_weight > 0 else torch.tensor(0.0).cuda().float()
        if isinstance(depth_loss, float):
            depth_loss = torch.tensor(0.0).cuda().float()    
        
        normal_l1, normal_cos = self.get_normal_loss(normal_pred * mask, normal_gt, normal_un_values, phy_un_values)
        
        smooth_loss = self.get_smooth_loss(model_outputs)
        
        # compute decay weights 
        if self.end_step > 0:
            decay = math.exp(-self.step / self.end_step * 10.)
        else:
            decay = 1.0
            
        self.step += 1

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss +\
               self.smooth_weight * smooth_loss +\
               decay * self.depth_weight * depth_loss +\
               decay * self.normal_l1_weight * normal_l1 +\
               decay * self.normal_cos_weight * normal_cos               
        
        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'smooth_loss': smooth_loss,
            'depth_loss': depth_loss,
            'normal_l1': normal_l1,
            'normal_cos': normal_cos
        }

        return output


class PhyReconLoss(MonoSDFLoss):
    def __init__(self, rgb_loss, 
                 eikonal_weight,
                 semantic_weight = 0.04,
                 smooth_weight = 0.005,
                 semantic_loss = torch.nn.CrossEntropyLoss(ignore_index = -1),
                 depth_weight = 0.1,
                 normal_l1_weight = 0.05,
                 normal_cos_weight = 0.05,
                 reg_vio_weight = 0.1,
                 use_obj_opacity = True,
                 bg_reg_weight = 0.1,
                 bg_surface_obj_reg_weight = 0.1,
                 use_rendering_uncertainty = True,
                 uncertainty_begin_iter = 30000,
                 phy_un_weight = 50,
                 physics_loss_begin_iter=0,
                 depth_type = 'marigold',
                 fix_physics_weight = 60,
                 use_curriculum_phy = True,
                 curri_phy = [50, 1],
                 end_step = -1):
        super().__init__(
                 rgb_loss = rgb_loss, 
                 eikonal_weight = eikonal_weight, 
                 smooth_weight = smooth_weight,
                 depth_weight = depth_weight,
                 normal_l1_weight = normal_l1_weight,
                 normal_cos_weight = normal_cos_weight,
                 uncertainty_begin_iter = uncertainty_begin_iter,
                 depth_type = depth_type,
                 phy_un_weight = phy_un_weight,
                 end_step = end_step)
        self.semantic_weight = semantic_weight
        self.bg_reg_weight = bg_reg_weight
        self.bg_surface_obj_reg_weight = bg_surface_obj_reg_weight
        self.semantic_loss = utils.get_class(semantic_loss)(reduction='none') if semantic_loss is not torch.nn.CrossEntropyLoss else torch.nn.CrossEntropyLoss(ignore_index = -1, reduction='none')
        self.reg_vio_weight = reg_vio_weight
        self.use_obj_opacity = use_obj_opacity
        self.use_rendering_uncertainty = use_rendering_uncertainty

        # for physical loss
        self.phy_loss = nn.MSELoss(reduction='mean')
        self.physics_loss_begin_iter = physics_loss_begin_iter
        self.fix_physics_weight = fix_physics_weight
        self.use_curriculum_phy = use_curriculum_phy
        self.curri_phy = curri_phy

        print(f"[INFO]: using weight for loss RGB_1.0 EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}\
            Semantic_{self.semantic_weight}, semantic_loss_type_{self.semantic_loss} Use_object_opacity_{self.use_obj_opacity} Reg_vio_{self.reg_vio_weight} BG_reg_{self.bg_reg_weight}\
            Use_rendering_uncertainty_{self.use_rendering_uncertainty}, uncertainty_begin_iter_{self.uncertainty_begin_iter}, Depth_type_{depth_type}")

    def get_semantic_loss(self, semantic_value, semantic_gt, phy_un_values):
        semantic_gt = semantic_gt.squeeze()
        semantic_loss = self.semantic_loss(semantic_value, semantic_gt)

        phy_un = phy_un_values.reshape(semantic_loss.shape)
        semantic_loss = semantic_loss / (self.phy_un_weight * phy_un + 1.0)

        semantic_loss = torch.mean(semantic_loss)
        return semantic_loss
    
    def object_distinct_loss(self, sdf_value, min_sdf):
        _, min_indice = torch.min(sdf_value.squeeze(), dim=1, keepdims=True)
        input = -sdf_value.squeeze() - min_sdf.detach()
        res = torch.relu(input).sum(dim=1, keepdims=True) - torch.relu(torch.gather(input, 1, min_indice))
        loss = res.mean()
        return loss

    def object_opacity_loss(self, predict_opacity, gt_opacity, phy_un_values, weight=None):
        target = torch.nn.functional.one_hot(gt_opacity.squeeze(), num_classes=predict_opacity.shape[1]).float()        # [ray_num, obj_num]
        if weight is None:

            loss = F.binary_cross_entropy(predict_opacity.clamp(1e-4, 1-1e-4), target, reduction='none').mean(dim=-1)    # [ray_num]
            phy_un = phy_un_values.reshape(loss.shape)
            loss = loss / (self.phy_un_weight * phy_un + 1.0)
            loss = torch.mean(loss)

        return loss
    
    # modify from RICO
    def get_bg_render_loss(self, bg_depth, bg_normal, mask):

        bg_depth = bg_depth.reshape(1, 32, 32)
        bg_normal = bg_normal.reshape(32, 32, 3).permute(2, 0, 1)

        mask = mask.reshape(1, 32, 32)

        depth_grad = self.compute_grad_error(bg_depth, mask)
        normal_grad = self.compute_grad_error(bg_normal, mask.repeat(3, 1, 1))

        bg_render_loss = depth_grad + normal_grad
        return bg_render_loss
    
    # modify from RICO
    def get_bg_surface_obj_reg(self, obj_sdfs):
        margin_target = torch.ones(obj_sdfs.shape).cuda()
        threshold = 0.05 * torch.ones(obj_sdfs.shape).cuda()
        loss = torch.nn.functional.margin_ranking_loss(obj_sdfs, threshold, margin_target)

        return loss
    
    def get_physical_loss(self, model_outputs):

        phy_loss_value =  -1.0 * self.phy_loss(model_outputs['obj_pc'].clone().detach(), model_outputs['obj_pc_after'])

        return phy_loss_value
    
    def compute_grad_error(self, x, mask):
        scales = 4
        grad_loss = torch.tensor(0.0).cuda().float()
        for i in range(scales):
            step = pow(2, i)

            mask_step = mask[:, ::step, ::step]
            x_step = x[:, ::step, ::step]

            M = torch.sum(mask_step[:1], (1, 2))

            diff = torch.mul(mask_step, x_step)

            grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
            mask_x = torch.mul(mask_step[:, :, 1:], mask_step[:, :, :-1])
            grad_x = torch.mul(mask_x, grad_x)

            grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
            mask_y = torch.mul(mask_step[:, 1:, :], mask_step[:, :-1, :])
            grad_y = torch.mul(mask_y, grad_y)

            image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

            divisor = torch.sum(M)

            if divisor == 0:
                scale_loss = torch.tensor(0.0).cuda().float()
            else:
                scale_loss = torch.sum(image_loss) / divisor

            grad_loss += scale_loss

        return grad_loss
    

    def forward(self, model_outputs, ground_truth, call_reg=False, call_bg_reg=False):
        output = super().forward(model_outputs, ground_truth)

        iter_step = model_outputs['iter_step']
        phy_un_values = model_outputs['phy_un_values'].detach()

        # compute physical loss
        if 'obj_pc' in model_outputs and iter_step >= self.physics_loss_begin_iter:
            phy_loss = self.get_physical_loss(model_outputs)
            if self.use_curriculum_phy:
                sim_rate = model_outputs['sim_rate']

                if sim_rate > 0.5:                      # near to max sim times, increase the weight
                    physics_weight = self.curri_phy[0] + (iter_step - self.physics_loss_begin_iter) * self.curri_phy[1] * sim_rate * 2
                else:
                    physics_weight = self.curri_phy[0] + (iter_step - self.physics_loss_begin_iter) * self.curri_phy[1]
            else:
                physics_weight = self.fix_physics_weight
        else:
            phy_loss = torch.tensor(0.0).cuda().float()
            physics_weight = self.fix_physics_weight

        if 'semantic_values' in model_outputs and not self.use_obj_opacity: # ObjectSDF loss: semantic field + cross entropy
            semantic_gt = ground_truth['segs'].cuda().long()
            semantic_loss = self.get_semantic_loss(model_outputs['semantic_values'], semantic_gt, phy_un_values)
        elif "object_opacity" in model_outputs and self.use_obj_opacity: # ObjectSDF++ loss: occlusion-awared object opacity + MSE
            semantic_gt = ground_truth['segs'].cuda().long()
            semantic_loss = self.object_opacity_loss(model_outputs['object_opacity'], semantic_gt, phy_un_values)
        else:
            semantic_loss = torch.tensor(0.0).cuda().float()
        
        if "sample_sdf" in model_outputs and call_reg:
            sample_sdf_loss = self.object_distinct_loss(model_outputs["sample_sdf"], model_outputs["sample_minsdf"])
        else:
            sample_sdf_loss = torch.tensor(0.0).cuda().float()

        # background_reg_loss = torch.tensor(0.0).cuda().float()        
        if 'bg_depth_values' in model_outputs:
            if 'bg_mask' in model_outputs:
                bg_mask = (model_outputs['bg_mask'] !=0).int()         # only smooth occluded background, i.e. semantic value is not 0
            else:
                bg_mask = (ground_truth['segs'] != 0).cuda()            # use gt mask directly
            background_reg_loss = self.get_bg_render_loss(model_outputs['bg_depth_values'], model_outputs['bg_normal_map'], bg_mask)
        else:
            background_reg_loss = torch.tensor(0.0).cuda().float()

        # # add bg surface object regularization loss
        # if 'obj_sdfs_behind_bg' in model_outputs:
        #     bg_surface_obj_reg = self.get_bg_surface_obj_reg(model_outputs['obj_sdfs_behind_bg'])
        #     if torch.isnan(bg_surface_obj_reg):
        #         bg_surface_obj_reg = torch.tensor(0.0).cuda().float()
        # else:
        #     bg_surface_obj_reg = torch.tensor(0.0).cuda().float()
        bg_surface_obj_reg = torch.tensor(0.0).cuda().float()

        output['phy_loss'] = phy_loss
        output['semantic_loss'] = semantic_loss
        output['collision_reg_loss'] = sample_sdf_loss
        output['background_reg_loss'] = background_reg_loss
        output['bg_surface_obj_reg'] = bg_surface_obj_reg
        output['loss'] = output['loss'] + physics_weight * phy_loss + self.semantic_weight * semantic_loss + self.reg_vio_weight* sample_sdf_loss + self.bg_reg_weight * background_reg_loss + self.bg_surface_obj_reg_weight * bg_surface_obj_reg
        return output