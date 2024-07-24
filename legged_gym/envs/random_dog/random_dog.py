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
from .random_dog_config_baseline import RandomBaseCfg

from termcolor import cprint


class Randomdog(LeggedRobot):
    cfg: RandomBaseCfg

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

    def post_physics_step(self):
        super().post_physics_step()

        self.ref_vt[:] = self.ref_body_vel[self.episode_length_buf]

        self.ref_qt_1[:] = self.ref_joint_pos[self.episode_length_buf + 1]
        self.ref_qt_2[:] = self.ref_joint_pos[self.episode_length_buf + 2]
        self.ref_qt_3[:] = self.ref_joint_pos[self.episode_length_buf + 3]
        self.ref_qt_5[:] = self.ref_joint_pos[self.episode_length_buf + 5]
        self.ref_qt_10[:] = self.ref_joint_pos[self.episode_length_buf + 10]
        self.ref_qt_20[:] = self.ref_joint_pos[self.episode_length_buf + 20]

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
            noise_vec[6:8] = 0.  # commands
            noise_vec[8:20] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[20:32] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[32:44] = 0.  # previous actions
            noise_vec[44:] = 0.  # reference motion clip

        noise_vel[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vel[3:7] = 0.
        noise_vel[7:11] = 0.
        if self.enable_priv_measured_height:
            noise_vel[11:198] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        if self.cfg.env.measure_obs_heights:
            noise_vec[48:235] = (noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements)
        return noise_vec, noise_vel

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
            self.ref_qt_2 = (self.commands[:, 0] < 0.0).unsqueeze(-1) * ((self.ref_qt_2 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) + 0.5 * self.commands[:, 0]).unsqueeze(-1) + \
                                                                         (self.ref_qt_1 - self.default_dof_pos) * 0.5 * (-self.commands[:, 0]).unsqueeze(-1)) + \
                            (self.commands[:, 0] >= 0.0).unsqueeze(-1)  * ((self.ref_qt_2 - self.default_dof_pos)* (torch.ones(self.num_envs, device=self.device) - self.commands[:, 0]).unsqueeze(-1) + \
                                                                           (self.ref_qt_3 - self.default_dof_pos) * self.commands[:, 0].unsqueeze(-1)) 
            self.ref_qt_3 = (self.commands[:, 0] < 0.0).unsqueeze(-1)  * ((self.ref_qt_3 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) + 0.5 * self.commands[:, 0]).unsqueeze(-1) + 
                                                                          (self.ref_qt_2 - self.default_dof_pos)* 0.5 * (-self.commands[:, 0]).unsqueeze(-1)) + \
                            (self.commands[:, 0] >= 0.0).unsqueeze(-1)  * ((self.ref_qt_3 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) - self.commands[:, 0]).unsqueeze(-1) + 
                                                                           (self.ref_qt_5 - self.default_dof_pos) * self.commands[:, 0].unsqueeze(-1)) 
            self.ref_qt_5 = (self.commands[:, 0] < 0.0).unsqueeze(-1)  * ((self.ref_qt_5 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) + 0.5 * self.commands[:, 0]).unsqueeze(-1) +
                                                                          (self.ref_qt_3 - self.default_dof_pos) * 0.5 * (-self.commands[:, 0]).unsqueeze(-1)) + \
                            (self.commands[:, 0] >= 0.0).unsqueeze(-1)  * ((self.ref_qt_5 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) - self.commands[:, 0]).unsqueeze(-1) +
                                                                           (self.ref_qt_10 - self.default_dof_pos) * self.commands[:, 0].unsqueeze(-1)) 
            self.ref_qt_10 =  (self.commands[:, 0] < 0.0).unsqueeze(-1)  * ((self.ref_qt_10 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) + 0.5 * self.commands[:, 0]).unsqueeze(-1) + 
                                                                            (self.ref_qt_5 - self.default_dof_pos) * 0.5 * (-self.commands[:, 0]).unsqueeze(-1)) + \
                              (self.commands[:, 0] >= 0.0).unsqueeze(-1)  * ((self.ref_qt_10 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) - self.commands[:, 0]).unsqueeze(-1) + 
                                                                             (self.ref_qt_20 - self.default_dof_pos) * self.commands[:, 0].unsqueeze(-1)) 
            self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.projected_gravity,
                                      self.commands[:, 0].unsqueeze(-1) * self.commands_scale[0],
                                      (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                      self.dof_vel * self.obs_scales.dof_vel,
                                      self.actions,
                                      self.ref_qt_1 * self.obs_scales.dof_pos,
                                      self.ref_qt_2 * self.obs_scales.dof_pos,
                                      self.ref_qt_3 * self.obs_scales.dof_pos,
                                      self.ref_qt_5 * self.obs_scales.dof_pos,
                                      self.ref_qt_10 * self.obs_scales.dof_pos,
                                    #   ((self.ref_qt_1 - self.default_dof_pos)) * self.obs_scales.dof_pos,
                                    #   ((self.ref_qt_2 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) - 0.5 * self.commands[:, 0]).unsqueeze(-1) + (self.ref_qt_1 - self.default_dof_pos) * 0.5 * self.commands[:, 0].unsqueeze(-1)) * self.obs_scales.dof_pos,
                                    #   ((self.ref_qt_3 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) - 0.5 * self.commands[:, 0]).unsqueeze(-1) + (self.ref_qt_2 - self.default_dof_pos) * 0.5 * self.commands[:, 0].unsqueeze(-1)) * self.obs_scales.dof_pos,
                                    #   ((self.ref_qt_5 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) - 0.5 * self.commands[:, 0]).unsqueeze(-1) + (self.ref_qt_3 - self.default_dof_pos) * 0.5 * self.commands[:, 0].unsqueeze(-1)) * self.obs_scales.dof_pos,
                                    #   ((self.ref_qt_10 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) - 0.5 * self.commands[:, 0]).unsqueeze(-1) + (self.ref_qt_5 - self.default_dof_pos) * 0.5 * self.commands[:, 0].unsqueeze(-1)) * self.obs_scales.dof_pos,
                                    #   ((self.ref_qt_1 - self.default_dof_pos)) * self.obs_scales.dof_pos,
                                    #   ((self.ref_qt_2 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) - self.commands[:, 0]).unsqueeze(-1) + (self.ref_qt_3 - self.default_dof_pos) * self.commands[:, 0].unsqueeze(-1)) * self.obs_scales.dof_pos,
                                    #   ((self.ref_qt_3 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) - self.commands[:, 0]).unsqueeze(-1) + (self.ref_qt_5 - self.default_dof_pos) * self.commands[:, 0].unsqueeze(-1)) * self.obs_scales.dof_pos,
                                    #   ((self.ref_qt_5 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) - self.commands[:, 0]).unsqueeze(-1) + (self.ref_qt_10 - self.default_dof_pos) * self.commands[:, 0].unsqueeze(-1)) * self.obs_scales.dof_pos,
                                    #   ((self.ref_qt_10 - self.default_dof_pos) * (torch.ones(self.num_envs, device=self.device) - self.commands[:, 0]).unsqueeze(-1) + (self.ref_qt_20 - self.default_dof_pos) * self.commands[:, 0].unsqueeze(-1)) * self.obs_scales.dof_pos,
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

        # print(self.ref_vt)
        # print(self.commands[:, 0])
        self.ref_vt = (self.commands[:, 0] < 0.0).unsqueeze(-1)  * self.ref_vt * (torch.ones(self.num_envs, device=self.device) + 0.5 * self.commands[:, 0]).unsqueeze(-1) + (self.commands[:, 0] >= 0.0).unsqueeze(-1) * self.ref_vt * (torch.ones(self.num_envs, device=self.device) + self.commands[:, 0]).unsqueeze(-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                             self.ref_vt,
                                            # self.ref_vt * (torch.ones(self.num_envs, device=self.device) + self.commands[:, 0]).unsqueeze(-1),
                                             ), dim=-1)
        # add noise if needed)
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