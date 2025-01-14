import imp
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm
import numpy as np
import json
import wandb
from torch.utils.tensorboard import SummaryWriter
import trimesh

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import get_time, refined_obj_bbox
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth
from PIL import Image

class PhyReconTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.description = kwargs['description']
        self.use_wandb = kwargs['use_wandb']
        self.ft_folder = kwargs['ft_folder']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        self.scan_id = scan_id
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        self.finetune_folder = kwargs['ft_folder'] if kwargs['ft_folder'] is not None else None
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        if self.GPU_INDEX == 0:
            utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
            self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
            utils.mkdir_ifnotexists(self.expdir)
            if self.description == "":              # default not use description
                self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            else:
                self.timestamp = f'{self.description}' + '_{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

            self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
            utils.mkdir_ifnotexists(self.plots_dir)

            # create checkpoints dirs
            self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
            utils.mkdir_ifnotexists(self.checkpoints_path)
            self.model_params_subdir = "ModelParameters"
            self.optimizer_params_subdir = "OptimizerParameters"
            self.scheduler_params_subdir = "SchedulerParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        # if (not self.GPU_INDEX == 'ignore'):
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('[INFO]: shell command : {0}'.format(' '.join(sys.argv)))

        print('[INFO]: Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.physics_sampling_begin_iter = self.conf.get_int('train.physics_sampling_begin_iter')                     # begin add physics sampling
        self.physics_loss_begin_iter = self.conf.get_int('train.physics_loss_begin_iter')                     # begin add physical loss

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=200000)
        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))
        # if scan_id < 24 and scan_id > 0: # BlendedMVS, running for 200k iterations
        self.nepochs = int(self.max_total_iters / self.ds_len)
        print('RUNNING FOR {0}'.format(self.nepochs))

        if len(self.train_dataset.label_mapping) > 0:
            # a hack way to let network know how many categories, so don't need to manually set in config file
            self.conf['model']['implicit_network']['d_out'] = len(self.train_dataset.label_mapping)
            print('RUNNING FOR {0} CLASSES'.format(len(self.train_dataset.label_mapping)))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=8,
                                                            pin_memory=True)
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        self.stable_mesh_path = os.path.join(self.plots_dir, 'stable_mesh')
        os.makedirs(self.stable_mesh_path, exist_ok=True)
        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model, plots_dir=self.plots_dir, physics_sampling_begin_iter=self.physics_sampling_begin_iter, physics_loss_begin_iter=self.physics_loss_begin_iter, z_floor=None, ft_folder=self.ft_folder)

        self.Grid_MLP = self.model.Grid_MLP
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'), physics_loss_begin_iter=self.physics_loss_begin_iter)

        # The MLP and hash grid should have different learning rates
        self.lr = self.conf.get_float('train.learning_rate')
        self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)
        
        if self.Grid_MLP:
            self.optimizer = torch.optim.Adam([
                {'name': 'uncertainty_field', 'params': list(self.model.uncertainty_field.grid_parameters()),
                'lr': self.lr * 10},
                {'name': 'encoding', 'params': list(self.model.implicit_network.grid_parameters()), 
                    'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'net', 'params': list(self.model.implicit_network.mlp_parameters()) +\
                    list(self.model.rendering_network.parameters()),
                    'lr': self.lr},
                {'name': 'density', 'params': list(self.model.density.parameters()),
                    'lr': self.lr},
            ], betas=(0.9, 0.99), eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.GPU_INDEX], broadcast_buffers=False, find_unused_parameters=True)
        
        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        # Loading a pretrained model for finetuning, the model path can be provided by self.finetune_folder
        if is_continue or self.finetune_folder is not None:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints') if self.finetune_folder is None\
             else os.path.join(self.finetune_folder, 'checkpoints')

            print('[INFO]: Loading pretrained model from {}'.format(old_checkpnts_dir))
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            # continue training need copy mesh files from old folder
            old_plots_folder = os.path.join(self.finetune_folder, 'plots')
            mesh_str = f'surface_{self.start_epoch}_*'
            cmd = f'cp {old_plots_folder}/{mesh_str} {self.plots_dir}'
            os.system(cmd)
            cmd = f'cp -r {old_plots_folder}/bbox {self.plots_dir}'
            os.system(cmd)

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).cuda()
        
        self.add_objectvio_iter = self.conf.get_int('train.add_objectvio_iter', default=100000)

        self.z_floor = None

        self.n_sem = self.conf.get_int('model.implicit_network.d_out')
        assert self.n_sem == len(self.train_dataset.label_mapping)

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def run(self):

        print("training...")
        if self.GPU_INDEX == 0 :

            if self.use_wandb:
                infos = json.loads(json.dumps(self.conf))
                wandb.init(
                    config=infos,
                    project=self.conf.get_string('wandb.project_name'),
                    name=self.timestamp,
                )

                # # visiualize gradient
                # wandb.watch(self.model, self.optimizer)

            else:
                print('Not using wandb, use tensorboard instead.')
                log_dir = os.path.join(self.expdir, self.timestamp, 'logs')
                os.makedirs(log_dir, exist_ok=True)
                tb_writer = SummaryWriter(log_dir)

        self.iter_step = self.start_epoch * len(self.train_dataset)
        print(f'Start epoch: {self.start_epoch}, iter_step: {self.iter_step}')
        for epoch in range(self.start_epoch, self.nepochs + 1):

            # set sim_idx_iter to 0
            self.model.module.sim_idx_iter = 0

            if self.iter_step >= self.physics_sampling_begin_iter:
                self.train_dataset.set_begin_physics_sampling(True)

                if self.z_floor is None:

                    dataset_type = self.conf.get_string('dataset.data_dir')
                    data_root_dir = self.conf.get_string('dataset.data_root_dir')
                    data_path = os.path.join(data_root_dir, dataset_type, f'scan{self.scan_id}')

                    try:
                        from glob import glob
                        mesh_list = glob(os.path.join(self.plots_dir, '*_whole.ply'))
                        mesh_list.sort(key=lambda x: int(x.split('_')[-2]))
                        mesh_path = mesh_list[-1]
                        print('use mesh_path: ', mesh_path)

                        input_scalemat = os.path.join(data_path, 'cameras.npz')
                        output_mesh = os.path.join(self.plots_dir, 'fuse_mesh.ply')
                        if dataset_type == 'replica':
                            traj = os.path.join(data_path, 'traj.txt')
                            fuse_cmd = f'python utils/fuse_mesh_replica.py --input_mesh {mesh_path} --input_scalemat {input_scalemat} --traj {traj} --output_mesh {output_mesh}'
                        else:               # scannet or scannetpp
                            fuse_cmd = f'python utils/fuse_mesh_scannet_scannetpp.py --input_mesh {mesh_path} --input_scalemat {input_scalemat} --output_mesh {output_mesh}'

                        os.system(fuse_cmd)
                        print(f'export fuse mesh: {output_mesh}')
                        fuse_mesh = trimesh.load(output_mesh)
                        scalemat = np.load(input_scalemat)['scale_mat_0']
                        scale_matrix = np.linalg.inv(scalemat[:3, :3].T)
                        fuse_mesh.vertices = (fuse_mesh.vertices - scalemat[:3, 3]) @ scale_matrix
                        _, _, z_min = np.min(fuse_mesh.vertices, axis=0)
                        self.z_floor = z_min

                        fuse_mesh.export(os.path.join(self.plots_dir, 'fuse_mesh_nerf.ply'))        # in nerf coords
                    except:
                        print('NOTE: Failed to fuse mesh, may caused by useless of pyrender, use default z_floor')
                        zfloor_txt_path = os.path.join(data_path, f'{dataset_type}_scan{self.scan_id}_zfloor.txt')
                        with open(zfloor_txt_path, 'r') as f:
                            self.z_floor = float(f.readline().strip())

                    self.model.module.z_floor = self.z_floor
                    
                    save_zfloor_path = os.path.join(self.plots_dir, 'z_floor.txt')
                    with open(save_zfloor_path, 'w') as f:
                        f.write(f'{self.z_floor:.4f}\n')
                    print(f'z_floor: {self.z_floor}')

                    # NOTE: floaters make a great negative impact on physical simulation
                    # filter some floaters and update object bbox for physical simulation
                    refined_obj_bbox(self.plots_dir, self.z_floor)

            if self.iter_step >= self.physics_loss_begin_iter:
                self.plot_freq = 1
                self.checkpoint_freq = 1

            if self.GPU_INDEX == 0 and epoch % self.checkpoint_freq == 0 and epoch != 0:
                self.save_checkpoints(epoch)

            if self.GPU_INDEX == 0 and self.do_vis and epoch % self.plot_freq == 0 and epoch != 0:
                self.model.eval()

                self.train_dataset.change_sampling_idx(-1)

                indices, model_input, ground_truth = next(iter(self.plot_dataloader))
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                
                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                res = []
                for s in tqdm(split):
                    out = self.model(s, indices)
                    d = {'rgb_values': out['rgb_values'].detach(),
                         'normal_map': out['normal_map'].detach(),
                         'depth_values': out['depth_values'].detach(),
                         'depth_un_values': out['depth_un_values'].detach(),
                         'normal_un_values': out['normal_un_values'].detach(),
                         'phy_un_values': out['phy_un_values'].detach(),}
                    if 'rgb_un_values' in out:
                        d['rgb_un_values'] = out['rgb_un_values'].detach()
                    if 'semantic_values' in out:
                        d['semantic_values'] = torch.argmax(out['semantic_values'].detach(),dim=1)
                    res.append(d)

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'], ground_truth['segs'])

                obj_bbox_dict = None
                if os.path.exists(os.path.join(self.plots_dir, 'bbox')):        # use object bbox
                    obj_bbox_dict = {}
                    obj_list = os.listdir(os.path.join(self.plots_dir, 'bbox'))
                    for obj in obj_list:
                        obj_idx = int((obj.split('.')[0]).split('_')[1])
                        with open(os.path.join(self.plots_dir, 'bbox', obj), 'r') as f:
                            bbox = json.load(f)
                        obj_bbox_dict[obj_idx] = bbox
                
                plt.plot(self.model.module.implicit_network,
                        indices,
                        plot_data,
                        self.plots_dir,
                        epoch,
                        self.iter_step,         # iter
                        self.img_res,
                        **self.plot_conf,
                        obj_bbox_dict=obj_bbox_dict
                        )

                self.model.train()
            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                
                self.optimizer.zero_grad()

                # visulize sampling pixels
                if self.iter_step % 1000 == 0:
                    vis_uv = model_input['uv'].squeeze(0).cpu().detach().numpy()        # [N_rays, 2]
                    vis_uv = vis_uv.astype(np.int32)
                    vis_rgb = ground_truth['full_rgb'].clone().squeeze(0).cpu().detach().numpy()     # [N_rays, 3]
                    vis_rgb = vis_rgb.reshape(self.img_res[0], self.img_res[1], 3)
                    vis_rgb = (vis_rgb * 255.0).astype(np.uint8)
                    vis_rgb[vis_uv[:, 1], vis_uv[:, 0], :] = [255, 0, 0]                # add red mask
                    vis_rgb = Image.fromarray(vis_rgb)
                    # create vis_pixels dir
                    vis_pixels_dir = os.path.join(self.plots_dir, 'vis_pixels')
                    os.makedirs(vis_pixels_dir, exist_ok=True)
                    vis_pixels_path = os.path.join(vis_pixels_dir, f'{epoch}_{indices[0]}.png')
                    vis_rgb.save(vis_pixels_path)
                
                model_outputs = self.model(model_input, indices, iter_step=self.iter_step)
                model_outputs['iter_step'] = self.iter_step

                # update physical uncertainty map for sampling
                if self.train_dataset.begin_physics_sampling:
                    self.train_dataset.update_physical_uncertainty_map(indices[0], model_input['sampling_idx'][0], model_outputs['phy_un_values'].detach().cpu().reshape(-1))

                # export stable mesh
                if self.model.module.sim.sleep_status:
                    sleep_obj_idx = model_outputs['sleep_obj_idx']
                    print(f'********** {sleep_obj_idx} sleep! **********')
                    sleep_obj_pc = model_outputs['sleep_pc']
                    sleep_floor_pc = model_outputs['sleep_pc_floor']
                    sleep_obj_pc = sleep_obj_pc.cpu().detach().numpy()
                    sleep_floor_pc = sleep_floor_pc.cpu().detach().numpy()
                    export_obj_pc = trimesh.PointCloud(sleep_obj_pc)
                    export_obj_pc.export(os.path.join(self.stable_mesh_path, f'e{epoch}_i{self.iter_step}_{sleep_obj_idx}_pc.ply'))
                    export_floor_pc = trimesh.PointCloud(sleep_floor_pc)
                    export_floor_pc.export(os.path.join(self.stable_mesh_path, f'e{epoch}_i{self.iter_step}_{sleep_obj_idx}_floor_pc.ply'))
                    print(f'********** {sleep_obj_idx} sleep obj pc export over **********')

                    # export mesh
                    print(f'********** {sleep_obj_idx} sleep obj mesh export begin **********')

                    sleep_bbox_file_path = os.path.join(self.plots_dir, 'bbox', f'bbox_{sleep_obj_idx}.json')
                    with open(sleep_bbox_file_path) as f:
                        obj_bbox = json.load(f)
                    _ = plt.get_object_surface_trace(
                        path = self.plots_dir,
                        epoch = epoch,
                        iter = self.iter_step,
                        sdf = lambda x: self.model.module.implicit_network.get_shift_sdf_raw(x),
                        resolution = 256,
                        obj_idx = sleep_obj_idx,
                        obj_bbox = obj_bbox,
                    )
                    print(f'********** {sleep_obj_idx} sleep obj mesh export over **********')

                    # reset sleep_status
                    self.model.module.sim.sleep_status = False

                
                loss_output = self.loss(model_outputs, ground_truth, call_reg=True) if\
                        self.iter_step >= self.add_objectvio_iter else self.loss(model_outputs, ground_truth, call_reg=False)
                # if change the pixel sampling pattern to patch, then you can add a TV loss to enforce some smoothness constraint
                loss = loss_output['loss']
                if 'sampling_loss' in model_outputs:
                    loss += model_outputs['sampling_loss']
                loss.backward()

                # calculate gradient norm
                total_norm = 0
                parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                self.optimizer.step()
                
                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))
                
                self.iter_step += 1                
                
                if self.GPU_INDEX == 0 and data_index %20 == 0:
                    print(
                        '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}, bete={9}, alpha={10}, semantic_loss = {11}, reg_loss = {12}'
                            .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
                                    loss_output['rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    psnr.item(),
                                    self.model.module.density.get_beta().item(),
                                    1. / self.model.module.density.get_beta().item(),
                                    loss_output['semantic_loss'].item(),
                                    loss_output['collision_reg_loss'].item()))
                    
                    if self.use_wandb:
                        for k, v in loss_output.items():
                            wandb.log({f'Loss/{k}': v.item()}, self.iter_step)

                        if 'sampling_loss' in model_outputs:
                            wandb.log({'Loss/sampling_loss': model_outputs['sampling_loss'].item()}, self.iter_step)

                        wandb.log({'Statistics/beta': self.model.module.density.get_beta().item()}, self.iter_step)
                        wandb.log({'Statistics/alpha': 1. / self.model.module.density.get_beta().item()}, self.iter_step)
                        wandb.log({'Statistics/psnr': psnr.item()}, self.iter_step)
                        wandb.log({'Statistics/total_norm': total_norm}, self.iter_step)
                        
                        if self.Grid_MLP:
                            wandb.log({'Statistics/lr0': self.optimizer.param_groups[0]['lr']}, self.iter_step)
                            wandb.log({'Statistics/lr1': self.optimizer.param_groups[1]['lr']}, self.iter_step)
                            wandb.log({'Statistics/lr2': self.optimizer.param_groups[2]['lr']}, self.iter_step)

                    else:
                        for k, v in loss_output.items():
                            tb_writer.add_scalar(f'Loss/{k}', v.item(), self.iter_step)

                        if 'sampling_loss' in model_outputs:
                            tb_writer.add_scalar('Loss/sampling_loss', model_outputs['sampling_loss'].item(), self.iter_step)

                        tb_writer.add_scalar('Statistics/beta', self.model.module.density.get_beta().item(), self.iter_step)
                        tb_writer.add_scalar('Statistics/alpha', 1. / self.model.module.density.get_beta().item(), self.iter_step)
                        tb_writer.add_scalar('Statistics/psnr', psnr.item(), self.iter_step)
                        tb_writer.add_scalar('Statistics/total_norm', total_norm, self.iter_step)

                        if self.Grid_MLP:
                            tb_writer.add_scalar('Statistics/lr0', self.optimizer.param_groups[0]['lr'], self.iter_step)
                            tb_writer.add_scalar('Statistics/lr1', self.optimizer.param_groups[1]['lr'], self.iter_step)
                            tb_writer.add_scalar('Statistics/lr2', self.optimizer.param_groups[2]['lr'], self.iter_step)
                
                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()

                if len(self.model.module.sim_obj_idx_list) == 0:
                    print('********** all object stable, over this epoch **********')
                    break


            if len(self.model.module.sim_obj_idx_list) == 0:
                print('********** all object stable, training over **********')
                break

        if self.GPU_INDEX == 0:
            self.save_checkpoints(epoch)

        if self.GPU_INDEX == 0:
            print('********** finish export mesh **********')
            self.model.eval()
            # surface_traces = plt.get_surface_sliding(path=self.plots_dir,
            #                     epoch=epoch + 1,
            #                     sdf=lambda x: self.model.module.implicit_network.get_sdf_vals(x).reshape(-1),
            #                     resolution=1024,
            #                     grid_boundary=self.plot_conf['grid_boundary'],
            #                     level=0
            #                     )
            plot_data = None

            obj_bbox_dict = None
            if os.path.exists(os.path.join(self.plots_dir, 'bbox')):        # use object bbox
                obj_bbox_dict = {}
                obj_list = os.listdir(os.path.join(self.plots_dir, 'bbox'))
                for obj in obj_list:
                    obj_idx = int((obj.split('.')[0]).split('_')[1])
                    with open(os.path.join(self.plots_dir, 'bbox', obj), 'r') as f:
                        bbox = json.load(f)
                    obj_bbox_dict[obj_idx] = bbox
            plt.plot(
                self.model.module.implicit_network,
                None,
                plot_data,
                self.plots_dir,
                'finish',
                self.iter_step,         # iter
                self.img_res,
                **self.plot_conf,
                obj_bbox_dict=obj_bbox_dict
            )
        
        if self.use_wandb:
            wandb.finish()
        print('training over')

        
    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt, seg_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.
      
        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        depth_map = depth_map * scale + shift
        
        seg_map = model_outputs['semantic_values'].reshape(batch_size, num_samples)
        seg_gt = seg_gt.to(seg_map.device)

        # save point cloud
        depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        pred_points = self.get_point_cloud(depth, model_input, model_outputs)

        gt_depth = depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        gt_points = self.get_point_cloud(gt_depth, model_input, model_outputs)

        # vis uncertainty map
        depth_un_map = model_outputs['depth_un_values'].reshape(batch_size, num_samples, 1)
        normal_un_map = model_outputs['normal_un_values'].reshape(batch_size, num_samples, 1)
        phy_un_map = model_outputs['phy_un_values'].reshape(batch_size, num_samples, 1)
        
        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'seg_gt': seg_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'seg_map': seg_map,
            "pred_points": pred_points,
            "gt_points": gt_points,
            "depth_un_map": depth_un_map,
            "normal_un_map": normal_un_map,
            "phy_un_map": phy_un_map
        }

        return plot_data
    
    def get_point_cloud(self, depth, model_input, model_outputs):
        color = model_outputs["rgb_values"].reshape(-1, 3)
        
        K_inv = torch.inverse(model_input["intrinsics"][0])[None]
        points = self.backproject(depth, K_inv)[0, :3, :].permute(1, 0)
        points = torch.cat([points, color], dim=-1)
        return points.detach().cpu().numpy()
