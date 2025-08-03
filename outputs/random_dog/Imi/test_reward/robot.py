from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import nn
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .bipedal_dog_config_baseline import BipedalBaseCfg

from termcolor import cprint

import numpy as np

class Bipedaldog(LeggedRobot):
    cfg: BipedalBaseCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):

        self.morphology_const_info_dict = {
            'base_mass': (0, 1),   # base
            'limb_mass': (1, 4),  # hip, thigh, calf
            'stiffness': (4, 5),  # p_gain
            'damping': (5, 6),  # d_gain
            'motor_strength': (6, 18),  # motor strength
            'friction': (18, 19),
        }
        # self.mass = 0

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.pen_vec = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                   requires_grad=False)

        # self.bipedal_data = {"feet_normal_force":[]
                             
        # }

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

    def _init_buffers(self):
        super()._init_buffers()

    def _compute_poses(self, actions):
        return super()._compute_poses(actions)

    def _compute_torques(self, actions):
        return super()._compute_torques(actions)

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_vel = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        if self.cfg.env.train_type == "standard":
            noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
            noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            noise_vec[6:9] = noise_scales.gravity * noise_level
            noise_vec[9:12] = 0.  # commands
            noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[36:48] = 0.  # previous actions
        else:
            noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.gravity * noise_level
            noise_vec[6:9] = 0.  # commands
            noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[33:45] = 0.  # previous actions
            # noise_vec[42:] = 0.  # reference motion clip

        noise_vel[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vel[3:7] = 0.
        noise_vel[7:11] = 0.
        if self.enable_priv_measured_height:
            noise_vel[11:198] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        if self.cfg.env.measure_obs_heights:
            noise_vec[48:235] = (noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements)
        return noise_vec, noise_vel

    def compute_observations(self):
        """ Computes observations
        """

        if self.cfg.env.train_type == "standard":
            self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                      self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.projected_gravity,
                                      self.commands[:, :3] * self.commands_scale,
                                      (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                      self.dof_vel * self.obs_scales.dof_vel,
                                      self.actions,
                                      ), dim=-1)


        else:
            self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.projected_gravity,
                                      self.commands[:, :3] * self.commands_scale,
                                      (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                      self.dof_vel * self.obs_scales.dof_vel,
                                      self.actions,
                                    #   (self.ref_qt_1 - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    #   (self.ref_qt_2 - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    #   (self.ref_qt_3 - self.default_dof_pos) * self.obs_scales.dof_pos,
                                      ), dim=-1)

        if self.enable_priv_enableMeasuredVel:
            self.privileged_obs_buf = self.base_lin_vel * self.obs_scales.lin_vel
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.foot_height,
                                                 contact,
                                                 ), dim=-1)
        if self.enable_priv_measured_height:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 heights,
                                                 ), dim=-1)
            if self.cfg.env.measure_obs_heights:
                self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        if self.enable_priv_disturbance_force:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.disturbance_force,
                                                 ), dim=-1)

        if self.enable_priv_bodys_mass:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.morph_priv_info_buf[:, 0:4],
                                                 ), dim=-1)

        if self.enable_priv_bodys_enableMotorStrength:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.morph_priv_info_buf[:, 4:18],
                                                 ), dim=-1)

        if self.enable_priv_friction_center:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.morph_priv_info_buf[:, 18:19],
                                                 ), dim=-1)

        self.rigid_body_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,0:3]

        self.feet_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3] - self.root_states[:, :3].unsqueeze(1) # - torch.tensor([[[0., 0., 0.02]]], device=self.device)
        self.feet_pos[:, 0] = quat_rotate_inverse(self.root_states[:, 3:7], self.feet_pos[:, 0])
        self.feet_pos[:, 1] = quat_rotate_inverse(self.root_states[:, 3:7], self.feet_pos[:, 1])
        self.feet_pos[:, 2] = quat_rotate_inverse(self.root_states[:, 3:7], self.feet_pos[:, 2])
        self.feet_pos[:, 3] = quat_rotate_inverse(self.root_states[:, 3:7], self.feet_pos[:, 3])
        self.feet_normal_force = self.contact_forces[:, self.feet_indices, 2] + torch.tensor([[0, 0, 1e-6, 1e-6]] * self.num_envs, device=self.device)
        # self.feet_normal_force = self.contact_forces[:, self.feet_indices, 2] + torch.tensor([[1e-6, 1e-6, 0, 0]] * self.num_envs, device=self.device)
        # self.feet_normal_force = self.contact_forces[:, self.feet_indices, 2] + torch.tensor([[1e-6, 1e-6, 1e-6, 1e-6]] * self.num_envs, device=self.device)

        # if self.episode_length_buf[0] == 500:
        #     force_norm = 3.
        #     force_direction = 0. * np.pi
        #     # print(torch.tensor([[force_norm * np.cos(force_direction), force_norm * np.sin(force_direction)]]*self.num_envs, device=self.device))
        #     self.root_states[:, 7:9] = torch.tensor([[force_norm *  np.cos(force_direction), force_norm * np.sin(force_direction)]]*self.num_envs, device=self.device)
        #     self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        
        # self.bipedal_data["feet_normal_force"] += self.feet_normal_force.clone().detach().cpu().numpy().tolist()
        # np.savetxt("feet_normal_force.txt", np.array(self.bipedal_data["feet_normal_force"]))

        self.cop_pos = (1./torch.sum(self.feet_normal_force, dim=-1)).unsqueeze(-1) * torch.sum(self.feet_pos * self.feet_normal_force.unsqueeze(-1), dim=1)
        self.com_pos = torch.sum(self.rigid_body_pos * self.mass.unsqueeze(-1),dim=1)/torch.sum(self.mass)
        self.com_pos = quat_rotate_inverse(self.root_states[:, 3:7], self.com_pos-self.root_states[:, 0:3])

        self.com_cop = self.com_pos - self.cop_pos
        self.pen_vec = quat_apply(self.root_states[:, 3:7], self.com_cop)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                             self.com_cop,
                                             ), dim=-1)
        # self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                      self.com_pos,
        #                                      ), dim=-1)
        # if self.episode_length_buf[0] % 100 == 0:
        #     self.gym.clear_lines(self.viewer)
        #     self.gym.refresh_rigid_body_state_tensor(self.sim)
        #     np.set_printoptions(precision=4)

        # com_geom = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0.85, 0.1, 0.1))
        # com_proj_geom = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0, 1, 0))
        # contact_geom = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0, 0, 1))
        # cop_geom = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0.85, 0.5, 0.1))
        # for i in range(self.num_envs):
        #     # draw COM(= base pose) and its projection
        #     base_pos = (self.root_states[i, :3]).cpu().numpy()
        #     com = self.coms[i] + base_pos
        #     com_pose = gymapi.Transform(gymapi.Vec3(com[0], com[1], com[2]), r=None)
        #     com_proj_pose = gymapi.Transform(gymapi.Vec3(com[0], com[1], 0), r=None)
        #     gymutil.draw_lines(com_geom, self.gym, self.viewer, self.envs[i], com_pose)
        #     # gymutil.draw_lines(com_proj_geom, self.gym, self.viewer, self.envs[i], com_proj_pose)

        #     # draw contact point and COP projection
        #     eef_state = self.rigid_body_state[i, self.feet_indices, :3]
        #     contact_idxs = (self.contact_forces[i, self.feet_indices, 2] > 1.).nonzero(as_tuple=False).flatten()
        #     contact_state = eef_state[contact_idxs]
        #     contact_force = self.contact_forces[i, self.feet_indices[contact_idxs], 2]

        #     for i_feet in range(contact_state.shape[0]):
        #         contact_pose = contact_state[i_feet, :]
        #         sphere_pose = gymapi.Transform(gymapi.Vec3(contact_pose[0], contact_pose[1], 0), r=None)
        #         gymutil.draw_lines(contact_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        #     # calculate and draw COP
        #     cop_sum = torch.sum(contact_state * contact_force.view(contact_force.shape[0], 1), dim=0)
        #     cop = cop_sum / torch.sum(contact_force)
        #     sphere_pose = gymapi.Transform(gymapi.Vec3(cop[0], cop[1], 0), r=None)
        #     gymutil.draw_lines(cop_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        #     # # draw inverted pendulum
        #     cop = cop.cpu().numpy()
        #     self.gym.add_lines(self.viewer, self.envs[i], 1, [cop[0], cop[1], 0, com[0], com[1], com[2]], [0.85, 0.1, 0.5])

        #     # draw connected line between contact point (fault case and normal case)
        #     self._draw_contact_polygon(contact_state, self.envs[i])

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_noise_vel


        if self.cfg.env.train_type == "Dream" or "RMA" or  "GenHis" or "GenMorphVel"  or  "Imi":
            # deal with normal observation, do sliding window
            prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
            # concatenate to get full history
            cur_obs_buf = self.obs_buf.clone().unsqueeze(1)
            self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)
            # refill the initialized buffers
            # if reset, then the history buffer are all filled with the current observation
            at_reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.obs_buf_lag_history[at_reset_env_ids, :, :] = self.obs_buf[at_reset_env_ids].unsqueeze(1)

            self.proprio_hist_buf = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()

    def _process_rigid_body_props(self, props, env_id, robot_type_id):

        # randomize base mass

        com_sum, mass_sum = np.zeros(3), 0.
        for i, p in enumerate(props):
            self.mass[i] = p.mass
            # calculate COM pose
            mass_sum += p.mass
            com_sum += p.mass * np.array([p.com.x, p.com.y, p.com.z])
        com = com_sum / mass_sum
        self.coms.append(com)

        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        self._update_morph_priv_buf(env_id=env_id, name='base_mass', value=props[0].mass,
                                        lower=4., upper=15.0)
        # add limb mass into priv info
        limb_mass = [props[1].mass, props[2].mass, props[3].mass]  # take FL's hip, thigh, calf for example

        self._update_morph_priv_buf(env_id=env_id, name='limb_mass', value=limb_mass,
                                    lower=0.1, upper=4.0)


        return props

    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            friction_range = self.cfg.domain_rand.friction_range
            if env_id == 0:
                # prepare friction randomization
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1], (num_buckets, 1), device="cpu"
                )
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

            if self.cfg.env.train_type == "RMA":
                self._update_priv_buf(env_id=env_id, name='friction', value=props[0].friction,
                                      lower=friction_range[0], upper=friction_range[1])
            if self.cfg.env.train_type == "GenMorphVel":
                self._update_morph_priv_buf(env_id=env_id, name='friction', value=props[0].friction,
                                            lower=friction_range[0], upper=friction_range[1])

        return props




    def _process_dof_props(self, props, env_id, robot_type_id):
        for i in range(len(props)):
            self.dof_pos_limits[env_id][i, 0] = props["lower"][i].item()
            self.dof_pos_limits[env_id][i, 1] = props["upper"][i].item()
            self.dof_vel_limits[env_id][i] = props["velocity"][i].item()
            self.torque_limits[env_id][i] = props["effort"][i].item()
            # soft limits
            m = (self.dof_pos_limits[env_id][i, 0] + self.dof_pos_limits[env_id][i, 1]) / 2
            r = self.dof_pos_limits[env_id][i, 1] - self.dof_pos_limits[env_id][i, 0]
            self.dof_pos_limits[env_id][i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            self.dof_pos_limits[env_id][i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit


        if self.cfg.domain_rand.randomize_motor_strength:
            motor_strength = []

            stiffness = self.cfg.control.stiffness
            damping = self.cfg.control.damping

            for i in range(self.num_dofs):
                name = self.dof_names[i]

                angle = self.cfg.init_state.default_joint_angles[name]

                self.default_dof_pos[env_id][i] = angle

                found = False
                rand_motor_strength = np.random.uniform(self.cfg.domain_rand.added_motor_strength[0],
                                                        self.cfg.domain_rand.added_motor_strength[1])
                for dof_name in stiffness.keys():

                    if dof_name in name:
                        if self.cfg.control.control_type == "POSE":
                            props['driveMode'][i] = gymapi.DOF_MODE_POS
                            props['stiffness'][i] = stiffness[dof_name] * rand_motor_strength  # self.Kp
                            props['damping'][i] = damping[dof_name] * rand_motor_strength  # self.Kd
                        elif self.cfg.control.control_type == "actuator_net":

                            rand_motor_offset = np.random.uniform(self.cfg.domain_rand.added_Motor_OffsetRange[0],
                                                                  self.cfg.domain_rand.added_Motor_OffsetRange[1])

                            self.motor_offsets[env_id][i] = rand_motor_offset
                            self.motor_strengths[env_id][i] = rand_motor_strength
                        else:
                            self.p_gains[env_id][i] = stiffness[dof_name] * rand_motor_strength
                            self.d_gains[env_id][i] = damping[dof_name] * rand_motor_strength
                        found = True
                if not found:
                    self.p_gains[i] = 0.0
                    self.d_gains[i] = 0.0
                    if self.cfg.control.control_type in ["P", "V"]:
                        print(f"PD gain of joint {name} were not defined, setting them to zero")
                motor_strength.append(rand_motor_strength)

            if self.cfg.env.train_type == "RMA":
                self._update_priv_buf(env_id=env_id, name='motor_strength', value=motor_strength,
                                      lower=self.cfg.domain_rand.added_motor_strength[0],
                                      upper=self.cfg.domain_rand.added_motor_strength[1])


            self._update_morph_priv_buf(env_id=env_id, name='stiffness', value=stiffness['joint'],
                                        lower=20., upper=80.)

            self._update_morph_priv_buf(env_id=env_id, name='damping', value=damping['joint'],
                                        lower=0.5,
                                        upper=2.0)

            self._update_morph_priv_buf(env_id=env_id, name='motor_strength', value=motor_strength,
                                        lower=self.cfg.domain_rand.added_motor_strength[0],
                                        upper=self.cfg.domain_rand.added_motor_strength[1])
        return props

    def _allocate_task_buffer(self, num_envs):
        super()._allocate_task_buffer(num_envs)
        self.num_env_morph_priv_obs = self.cfg.env.num_env_morph_priv_obs
        self.morph_priv_info_buf = torch.zeros((num_envs, self.num_env_morph_priv_obs), device=self.device,
                                               dtype=torch.float)
    def _update_morph_priv_buf(self, env_id, name, value, lower=None, upper=None):
        # normalize to -1, 1
        s, e = self.morphology_const_info_dict[name]
        if type(value) is list:
            value = to_torch(value, dtype=torch.float, device=self.device)
        if type(lower) is list or type(upper) is list:
            lower = to_torch(lower, dtype=torch.float, device=self.device)
            upper = to_torch(upper, dtype=torch.float, device=self.device)
        if lower is not None and upper is not None:
            value = ( value -  lower) / (upper - lower)
        self.morph_priv_info_buf[env_id, s:e] = value

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        # print("lin_vel_z", self.base_lin_vel)
        return torch.square(self.base_lin_vel[:, 0])
        # return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        # return torch.sum(torch.square(self.base_ang_vel[:, 1].unsqueeze(1)), dim=1)
        # print("ang_vel_xy:", self.base_ang_vel)
        return torch.sum(torch.square(self.base_ang_vel[:, 1:3]), dim=1)
        # return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        # print(torch.square(self.projected_gravity[:, :2]))
        # print(torch.sign(self.projected_gravity[:, 0]))
        # print("projected_gravity:", self.projected_gravity)
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation_3(self):
        # Penalize non flat base orientation
        # print(torch.square(self.projected_gravity[:, :2]))
        # print(torch.sign(self.projected_gravity[:, 0]))
        return torch.sum(torch.square(self.projected_gravity[:, 2]).unsqueeze(1), dim=1)

    def _reward_fl_contact_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        # print("-------------------------")
        # print(self.contact_forces[:, self.feet_indices, :])
        # print(foot_forces[:, 0])
        fl_force = foot_forces[:, 0]
        return fl_force

    def _reward_fr_contact_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        # print("-------------------------")
        # print(self.contact_forces[:, self.feet_indices, :])
        # print(foot_forces[:, 0])
        fr_force = foot_forces[:, 1]
        return fr_force

    def _reward_f_contact_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        # print("-------------------------")
        # print(self.contact_forces[:, self.feet_indices, :])
        # print(foot_forces[:, 0])
        f_force = foot_forces[:, 0] + foot_forces[:, 1]
        return f_force

    def _reward_rl_contact_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        # print("-------------------------")
        # print(self.contact_forces[:, self.feet_indices, :])
        # print(foot_forces[:, 0])
        rl_force = foot_forces[:, 2]
        return rl_force

    def _reward_rr_contact_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        # print("-------------------------")
        # print(self.contact_forces[:, self.feet_indices, :])
        # print(foot_forces[:, 0])
        rr_force = foot_forces[:, 3]
        return rr_force

    def _reward_r_contact_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        # print("-------------------------")
        # print(self.contact_forces[:, self.feet_indices, :])
        # print(foot_forces[:, 0])
        r_force = foot_forces[:, 2] + foot_forces[:, 3]
        return r_force

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)return torch.square(self.base_lin_vel[:, 2])
        # command_rearrange_indices = torch.tensors([2,1], device=self.device)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] + self.base_lin_vel[:, 1:3]), dim=1)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        # jacobian = [[],[],[]]

        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 0])
        # ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_thigh_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [1, 4, 7, 10]] - (self.default_dof_pos[:, [1, 4, 7, 10]] + torch.tensor([[0., 0., 1.57, 1.57]]*self.num_envs, dtype=torch.float, device=self.device))), dim=1)
        # return torch.sum(torch.abs(self.dof_pos[:, [1, 4, 7, 10]] - (self.default_dof_pos[:, [1, 4, 7, 10]] + torch.tensor([[-1.57, -1.57, 0., 0.]]*self.num_envs, dtype=torch.float, device=self.device))), dim=1)

        # return torch.sum(torch.abs(self.dof_pos[:, [1, 4, 7, 10]] - (self.default_dof_pos[:, [1, 4, 7, 10]] )), dim=1)

    def _reward_inv_pendulum(self):

        self.pen_len = torch.norm(self.pen_vec, dim=1)

        self.pen_ang = torch.acos(self.pen_vec[:,2] / (1e-6 * torch.ones(self.num_envs, device=self.device, requires_grad=False) + self.pen_len))


        return torch.pow(self.pen_ang, 2)

    def _reward_inv_pendulum_acc(self):


        self.pen_len = torch.norm(self.pen_vec, dim=1)

        self.pen_ang_acc = (torch.ones(self.num_envs,device=self.device, requires_grad=False) - torch.pow(self.pen_vec[:,2] / (1e-6 * torch.ones(self.num_envs, device=self.device, requires_grad=False) + self.pen_len), 2)) / (1e-6 * torch.ones(self.num_envs, device=self.device, requires_grad=False) + torch.pow(self.pen_len, 2))

        return self.pen_ang_acc 

    def _reward_inv_pendulum_both(self):

        return torch.pow(self.pen_ang, 2) + torch.abs(self.pen_ang_acc)


    def _reward_cart_table_len_xy(self):
        self.cart_len_xy = torch.norm(self.pen_vec[:, :2], dim=1)

        return self.cart_len_xy

    def _reward_cart_table_len_z(self):
        self.cart_len_z = torch.norm(self.pen_vec[:, 2], dim=1)

        return self.cart_len_z