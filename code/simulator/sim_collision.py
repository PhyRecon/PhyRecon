import torch
import torch.nn as nn
import numpy as np
import taichi as ti

torch.set_printoptions(precision=10)


import os
import sys
sys.path.append(os.getcwd())

from simulator.RigidBody import RigidBody
from simulator.CollisionDetection import proximity_detection_with_fix_particles

import time

# init taichi
arch = ti.cuda
ti.init(arch=arch)              # NOTE: debug=True will check array overflow, but is slower


class Simulator(torch.nn.Module):

    def __init__(self, use_simulator_thresh=False, use_sleepstop=False, n_max=100000):
        super(Simulator, self).__init__()

        self.device = 'cuda'
        self.particles_radius = 0.001
        if use_simulator_thresh:
            self.delta_thresh = 1e-4
        else:
            self.delta_thresh = 0.0         # not use threshold

        self.use_sleepstop = use_sleepstop

        self.dt = 0.01                      # simulator dt
        self.steps = 100                    # simulator steps

        self.sleepThreshold = self.dt * 9.8
        self.sleep_status = False

        dim = 3
        self.pmass = 0.01

        self.obj = RigidBody(dim, self.pmass, [0.0, 0.0, -9.8], n_max)
        self.floor = RigidBody(dim, self.pmass, [0.0, 0.0, -9.8], n_max)

        print('simulator init......')

        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(ctx, obj_pc, floor_pc, iter_step, obj_idx):       # obj_pc: [n, 3], torch.tensor (cuda)

                # NOTE: for every sim, init sleep_status = False
                self.sleep_status = False

                ctx.input_size = obj_pc.shape
                self.device = obj_pc.device

                obj_n_particles = obj_pc.shape[0]
                ctx.obj_n_particles = obj_n_particles
                output_p_x = torch.zeros_like(obj_pc, device=self.device, requires_grad=True)
                print('obj_n_particles', obj_n_particles)

                floor_n_particles = floor_pc.shape[0]
                print('floor_n_particles', floor_n_particles)

                self.obj.n_particles[0] = obj_n_particles
                self.obj.set_init_rigid()
                obj_pc_padded = torch.ones((n_max, 3), device='cuda') * 10000           # make sure not in contact boundary
                obj_pc_padded[:obj_n_particles] = obj_pc
                self.obj.p_x.from_torch(obj_pc_padded)

                # init the first collision point as the original position
                self.obj.p_x_first_collision_pos.from_torch(obj_pc_padded)

                self.floor.n_particles[0] = floor_n_particles
                self.floor.set_init_rigid()
                floor_pc_padded = torch.ones((n_max, 3), device='cuda') * 10000
                floor_pc_padded[:floor_n_particles] = floor_pc
                self.floor.p_x.from_torch(floor_pc_padded)

                # initialized rigid body state
                self.obj.set_rigid_property()
                self.obj.sleep[0] = -1              # not sleep
                
                self.floor.set_rigid_property()     # floor not need to set init state

                count_list = []
                for step_idx in range(self.steps):
                    # print('step', step_idx)

                    # simulator
                    self.obj.pre_simulation()
                    self.obj.compute_rigid_motion_wo_contact(self.dt)
                    self.obj.update_particles_state()

                    # calculate contact
                    count = 0
                    while True:
                        self.obj.pre_contact()
                        proximity_detection_with_fix_particles(self.obj, self.floor, self.particles_radius, self.dt)
                        self.obj.post_contact()
                        count = count + 1
                        # print('contact number', self.obj.n_contact[0])
                        if (self.obj.n_contact[0] == 0 or count > 150):
                            break

                    # print('count', count)
                    print(f'step {step_idx} count {count}')
                    count_list.append(count)

                    self.obj.check_sleep(self.sleepThreshold)
                    if (self.obj.sleep[0] == 1):                    # break if sleep
                        print('sleep status', self.obj.sleep[0])
                        break

                ctx.sim_step = step_idx + 1                 # simulation step, not include sleep step
                ctx.count_list = count_list
                ctx.sleep_status = self.obj.sleep[0]

                output_p_x_padded = self.obj.p_x_first_collision_pos.to_torch(device=self.device)
                output_p_x = output_p_x_padded[:obj_n_particles]

                delta_points = output_p_x - obj_pc
                # print('delta_points', delta_points[:10])
                delta_mask = torch.abs(delta_points) > self.delta_thresh
                ctx.delta_mask = delta_mask

                if self.use_sleepstop and ctx.sim_step == 1:
                    self.sleep_status = True

                p_x_collision = self.obj.p_x_collision.to_torch(device=self.device)
                p_x_collision = p_x_collision[:obj_n_particles]                         # [obj_n_particles, 1]
                collision_mask = (p_x_collision > 0.5).squeeze(1)
                # print('collision points: ', sum(collision_mask))

                return output_p_x, collision_mask
            
            @staticmethod
            def backward(ctx, dL_dpx_after, dL_dcm):        # dL_dcm is None

                input_size = ctx.input_size
                obj_n_particles = ctx.obj_n_particles
                delta_mask = ctx.delta_mask

                sim_step = ctx.sim_step
                count_list = ctx.count_list
                sleep_status = ctx.sleep_status

                self.zero_grad()

                input_grad = torch.zeros(*input_size, dtype=dL_dpx_after.dtype, device=self.device)

                if sim_step == 1:                           # rest object in simulator, just 1 step to sleep
                    print('input_grad', input_grad[:10])
                    print('max abs input_grad: ', torch.max(torch.abs(input_grad)))

                    return input_grad, None, None, None

                dL_dpx_after = dL_dpx_after * delta_mask

                dL_dpx_after_padded = torch.zeros((n_max, 3), device='cuda')
                dL_dpx_after_padded[:obj_n_particles] = dL_dpx_after
                self.obj.p_x.grad.from_torch(dL_dpx_after_padded.contiguous())

                # print('before backward', self.obj.p_x.grad.to_torch(device=self.device)[:10])

                for step_idx in range(sim_step):
                    grad_step_idx = sim_step - step_idx - 1
                    count = count_list[grad_step_idx]

                    for i in range(count):
                        self.obj.post_contact.grad()
                        proximity_detection_with_fix_particles.grad(self.obj, self.floor, self.particles_radius, self.dt)
                        self.obj.pre_contact.grad()

                    self.obj.update_particles_state.grad()
                    self.obj.compute_rigid_motion_wo_contact.grad(self.dt)
                    self.obj.pre_simulation.grad()

                    # print(f'{grad_step_idx} backward', self.obj.p_x.grad.to_torch(device=self.device)[:10])

                self.obj.set_rigid_property.grad()

                input_grad_padded = self.obj.p_x.grad.to_torch(device=self.device)
                input_grad = input_grad_padded[:obj_n_particles]

                # only add collision points grad
                p_x_collision = self.obj.p_x_collision.to_torch(device=self.device)
                p_x_collision = p_x_collision[:obj_n_particles]                         # [obj_n_particles, 1]
                input_grad = input_grad * p_x_collision
                print('p_x_collision sum', sum(p_x_collision))

                print('input_grad', input_grad[:10])
                print('max abs input_grad: ', torch.max(torch.abs(input_grad)))

                # remove nan grad
                input_grad[input_grad != input_grad] = 0.0

                return input_grad, None, None, None

        self._module_function = _module_function.apply


    def zero_grad(self):
        self.obj.p_x.grad.fill(0.)
        self.obj.p_x_pre.grad.fill(0.)
        self.obj.p_x_local.grad.fill(0.)

        self.floor.p_x.grad.fill(0.)
        self.floor.p_x_pre.grad.fill(0.)
        self.floor.p_x_local.grad.fill(0.)
        

    def forward(self, obj_pc, floor_pc, iter_step, obj_idx):

        return self._module_function(obj_pc, floor_pc, iter_step, obj_idx)

    def get_contact_points(self):
        p_x_collision = self.obj.p_x_collision.to_torch(device=self.device)
        obj_n_particles = self.obj.n_particles[0]
        p_x_collision = p_x_collision[:obj_n_particles]                         # [obj_n_particles, 1]

