

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
from termcolor import cprint
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from scipy.spatial.transform import Rotation as R
import pickle as pkl
from urdfpy import URDF

class Agent:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.robot = URDF.load(urdf_path)

    def get_body_size(self):
        for jdx, link in enumerate(self.robot.links):
            if link.name in ["trunk"]:
                trunk_size = self.robot.links[jdx].visuals[0].geometry.box.size

        leg_indices = []
        leg_indices1 = []
        for link_idx, link in enumerate(self.robot.links):
            if link.name.split("_")[-1] == "thigh":
                leg_indices.append(link_idx)

        for link_idx, link in enumerate(self.robot.links):
            if link.name.split("_")[-1] == "calf":
                leg_indices1.append(link_idx)

        for i in leg_indices:
            thigh_size = self.robot.links[i].visuals[0].geometry.box.size[0]


        for i in leg_indices:
            calf_size = self.robot.links[i].visuals[0].geometry.box.size[0]


        return list(trunk_size), thigh_size, calf_size
        # return list(trunk_size), list(thigh_size), list(calf_size)


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training
        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.num_calls = 0
        self.cfg = cfg


        self._setup_priv_option_config(self.cfg.privInfo)


        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = self.cfg.sim.enable_debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)

        # ************************ RMA specific ************************
        self.priv_info_dict = {
                'base_mass': (0, 1),
                'center': (1, 3),
                'motor_strength': (3, 15),  # weaken or strengthen
                'friction': (15, 16),
            }

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        # load actuator network
        if self.cfg.control.control_type == "actuator_net":
            print("********* Load actuator net from *********", self.cfg.control.actuator_net_file)
            # print("----------------------------------------------", "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_go1_join_brick_stairs_it550.pt")
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            actuator_network = torch.jit.load(actuator_network_path).to(self.device)

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                torques = actuator_network(xs.view(self.num_envs * 12, 6))
                return torques.view(self.num_envs, 12)

            self.actuator_network = eval_actuator_network
        self.init_base_pos = torch.clone(self.root_states[:,:3])
        self.init_base_quat = torch.clone(self.root_states[:,3:7])
        self.imi_data = {
                    "joint_pos":[],
                    "joint_vel":[],
                    "body_pos":[],
                    "body_vel":[],
                    "feet_pos":[], 

                    "body_quat":[],
                    "body_omega":[],

                    "foot_forces":[],
                    "foot_velocities":[],
                    "desired_contact":[],

                    "com_pos":[],
                    "cop_pos":[],
                    }
        self.ref_log = False
        # f = open("imitation_data_set/go1_isaac/wtw_ref_go1_troting.pkl", "rb")
        # f = open("imitation_data_set/go1_isaac/wtw_ref_go1_pacing.pkl", "rb")
        f = open("imitation_data_set/go1_isaac/wtw_ref_go1_bounding.pkl", "rb")
        # f = open("imitation_data_set/go1_isaac/wtw_ref_go1_pronking.pkl", "rb")
        # f = open("imitation_data_set/go1_pybullet/mpc_ref_go1_walking.pkl", "rb")
        # f = open("imitation_data_set/go1_isaac/rl_ref_go1_bipedal_walking.pkl", "rb")
        data = pkl.load(f)
        # print("**************************************", len(data["joint_pos"]))
        self.ref_joint_pos = torch.tensor(data["joint_pos"][:][::], dtype=torch.float, device=self.device)
        self.ref_joint_vel = torch.tensor(data["joint_vel"][:][::], dtype=torch.float, device=self.device)
        self.ref_body_pos = torch.tensor(data["body_pos"][:][::], dtype=torch.float, device=self.device)
        self.ref_body_vel = torch.tensor(data["body_vel"][:][::], dtype=torch.float, device=self.device)
        self.ref_feet_pos = torch.tensor(data["feet_pos"][:][::], dtype=torch.float, device=self.device)
        self.ref_body_quat = torch.tensor(data["body_quat"][:][::], dtype=torch.float, device=self.device)
        self.ref_body_omega = torch.tensor(data["body_omega"][:][::], dtype=torch.float, device=self.device)
        self.ref_foot_forces = torch.tensor(data["foot_forces"][:][::], dtype=torch.float, device=self.device)
        self.ref_foot_velocities = torch.tensor(data["foot_velocities"][:][::], dtype=torch.float, device=self.device)
        self.ref_desired_contact = torch.tensor(data["desired_contact"][:][::], dtype=torch.float, device=self.device)
        self.ref_com_pos = torch.tensor(data["com_pos"][:][::], dtype=torch.float, device=self.device)
        self.ref_cop_pos = torch.tensor(data["cop_pos"][:][::], dtype=torch.float, device=self.device)

    def make_handle_trans(self, res, env_num, trans, rot, hfov=None):
        # TODO Add camera sensors here?
        camera_props = gymapi.CameraProperties()
        # print("FOV: ", camera_props.horizontal_fov)
        # camera_props.horizontal_fov = 75.0
        # 1280 x 720
        width, height = res
        camera_props.width = width
        camera_props.height = height
        camera_props.enable_tensors = True
        if hfov is not None:
            camera_props.horizontal_fov = hfov
        # print("envs[i]", self.envs[i])
        # print("len envs: ", len(self.envs))
        camera_handle = self.gym.create_camera_sensor(
            self.envs[env_num], camera_props
        )
        # print("cam handle: ", camera_handle)

        local_transform = gymapi.Transform()
        # local_transform.p = gymapi.Vec3(75.0, 75.0, 30.0)
        # local_transform.r = gymapi.Quat.from_euler_zyx(0, 3.14 / 2, 3.14)
        x, y, z = trans
        local_transform.p = gymapi.Vec3(x, y, z)
        a, b, c = rot
        local_transform.r = gymapi.Quat.from_euler_zyx(a, b, c)

        return camera_handle, local_transform

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        self.log_ref_data(self.ref_log)
        for _ in range(self.cfg.control.decimation):

            ### torques change 1 ############
            if self.cfg.control.control_type == "POSE":
                # Note: Position control
                self.target_poses = self._compute_poses(self.actions).view(self.target_poses.shape)
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_poses))
            else:
                # Note: Torques control
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        # print(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1))
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        self.obs_dict['obs'] = self.obs_buf
        self.obs_dict['privileged_info'] = self.privileged_obs_buf.to(self.device)
        if self.cfg.env.train_type == "RMA":
            self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.device)
            self.obs_dict['priv_info'] = self.priv_info_buf.to(self.device)
        else:
            self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.device).flatten(1)

        # self.desired_contact_states = self.step_gait_planner(0., 0.5, 0.) # bound
        # self.desired_contact_states = self.step_gait_planner(0.5, 0., 0.)  # trot
        self.desired_contact_states = self.step_gait_planner(0., 0., 0.5) # pace
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def step_gait_planner(self, phases, offsets, bounds):
        gait_frequency = 1
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * gait_frequency, 1.0)
        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]
        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
        # print((torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)<0.8)*1.)
        return (torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)<0.75)*1.
        # return torch.tensor([[0., 0., 1., 1.]]*self.num_envs, device=self.device)

    def log_ref_data(self, log=False):
        if log:
            self.imi_data["joint_pos"] += (self.dof_pos.detach().cpu().numpy()).tolist() 
            self.imi_data["joint_vel"] += (self.dof_vel.detach().cpu().numpy()).tolist() 
            self.imi_data["body_vel"] += (self.base_lin_vel.detach().cpu().numpy()).tolist()
            self.imi_data["body_omega"] += (self.base_ang_vel.detach().cpu().numpy()).tolist()

            self.local_base_pos = quat_rotate_inverse(self.init_base_quat, self.root_states[:,:3] - self.init_base_pos)
            self.start_T = R.from_quat(self.init_base_quat.detach().cpu().numpy()[0]).as_matrix()
            self.base_T = R.from_quat(self.base_quat.detach().cpu().numpy()[0]).as_matrix()
            self.imi_data["body_pos"] += (self.local_base_pos.detach().cpu().numpy()).tolist()
            self.imi_data["body_quat"] += [R.from_matrix((self.start_T).T @ self.base_T).as_quat().tolist()]
            # print("body_pos", self.local_base_pos.detach().cpu().numpy())
            # print("body_quat", R.from_matrix((self.start_T).T @ self.base_T).as_quat().tolist())
            self.local_eef_state = self.rigid_body_state[:, self.feet_indices, :3] - self.root_states[:, :3].unsqueeze(1)
            self.local_eef_state[:, 0] = quat_rotate_inverse(self.root_states[:, 3:7], self.local_eef_state[:, 0])
            self.local_eef_state[:, 1] = quat_rotate_inverse(self.root_states[:, 3:7], self.local_eef_state[:, 1])
            self.local_eef_state[:, 2] = quat_rotate_inverse(self.root_states[:, 3:7], self.local_eef_state[:, 2])
            self.local_eef_state[:, 3] = quat_rotate_inverse(self.root_states[:, 3:7], self.local_eef_state[:, 3])
            self.foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
            self.foot_velocities = self.rigid_body_state[:, self.feet_indices, 7:10]
            self.foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
            self.imi_data["feet_pos"] += (self.local_eef_state.detach().cpu().numpy()).tolist()
            self.imi_data["foot_forces"] += self.foot_forces.detach().cpu().numpy().tolist()
            self.imi_data["foot_velocities"] += self.foot_velocities.detach().cpu().numpy().tolist()
            # print("feet_pos", self.local_eef_state.detach().cpu().numpy())
            # print("foot_forces", self.foot_forces.detach().cpu().numpy())
            # print("foot_velocities", self.foot_velocities.detach().cpu().numpy())
            f = open("imitation_data_set/go1_isaac/rl_ref_go1_bipedal_walking.pkl", "wb")
            data = pkl.dump(self.imi_data, f)
            f.close()
        else:
            pass

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg.control.control_type == "POSE":
            # Note: Position control
            self.gym.refresh_dof_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()


        foot_height  = self.rigid_body_state[:, self.feet_indices, 2:3]
        foot_vel = self.rigid_body_state[:, self.feet_indices, 7:9]

        self.foot_height = torch.cat((foot_height[:, 0, ],
                                      foot_height[:, 1, ],
                                      foot_height[:, 2, ],
                                      foot_height[:, 3, ],
                                      ), dim=-1)
        self.foot_vel = torch.cat((foot_vel[:, 0, ],
                                   foot_vel[:, 1, ],
                                   foot_vel[:, 2, ],
                                   foot_vel[:, 3, ],
                                      ), dim=-1)
        # print('foot', foot_height)
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions_2[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contact_forces[:] = self.contact_forces[:]
        self.last_torques[:] =  self.torques[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers
        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_actions_2[env_ids] = 0.

        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_eval(self):
        """Compute evals
        Reports eval function values, not used in training.
        """
        for i in range(len(self.eval_functions)):
            name = self.eval_names[i]
            rew = self.eval_functions[i]()
            self.eval_sums[name] += rew

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

        noise_vel[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vel[3:7] = 0
        noise_vel[7:11] = 0
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
                                      # self.commands[:, :3] * self.commands_scale,
                                      (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                      self.dof_vel * self.obs_scales.dof_vel,
                                      self.actions,
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

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield",  "stair", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type == "stair":
            self._create_stairs()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain meshes type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
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
            if self.cfg.env.train_type == "RMA":
                self._update_priv_buf(env_id=env_id, name='base_mass', value=props[0].mass,
                                      lower=rng[0], upper=rng[1])


        if self.cfg.domain_rand.randomize_center:
            center = self.cfg.domain_rand.added_center_range
            com = [np.random.uniform(center[0], center[1]), np.random.uniform(center[0], center[1])]
            props[0].com.x, props[0].com.y = com
            if self.cfg.env.train_type == "RMA":
                self._update_priv_buf(env_id=env_id, name='center', value=com,
                                      lower=center[0], upper=center[1])
        return props

    def _process_rigid_body_size_props(self, trunk_size, thigh_size, calf_size, env_id):
        # todo: get the base and limb size and update the buf, hardcode now
        pass


    def _process_rigid_com_props(self, props, env_id):
        # randomize center
        if self.cfg.domain_rand.randomize_center:
            center = self.cfg.domain_rand.added_center_range
            com = [np.random.uniform(center[0], center[1]), np.random.uniform(center[0], center[1])]
            props[0].com.x, props[0].com.y = com
            if self.cfg.env.train_type == "RMA":
                self._update_priv_buf(env_id=env_id, name='center', value=com,
                                      lower=center[0], upper=center[1])
        return props


    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
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
            for i in range(self.num_dofs):
                name = self.dof_names[i]
                angle = self.cfg.init_state.default_joint_angles[name]
                self.default_dof_pos[env_id][i] = angle
                found = False
                rand_motor_strength = np.random.uniform(self.cfg.domain_rand.added_motor_strength[0],
                                                        self.cfg.domain_rand.added_motor_strength[1])


                for dof_name in self.cfg.control.stiffness.keys():

                    if dof_name in name:
                        if self.cfg.control.control_type == "POSE":
                            props['driveMode'][i] = gymapi.DOF_MODE_POS
                            props['stiffness'][i] = self.cfg.control.stiffness[dof_name] * rand_motor_strength  # self.Kp
                            props['damping'][i] = self.cfg.control.damping[dof_name] * rand_motor_strength  # self.Kd
                        elif self.cfg.control.control_type == "actuator_net":
                            props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
                            rand_motor_offset = np.random.uniform(self.cfg.domain_rand.added_Motor_OffsetRange[0],
                                              self.cfg.domain_rand.added_Motor_OffsetRange[1])

                            self.motor_offsets[env_id][i] = rand_motor_offset
                            self.motor_strengths[env_id][i] = rand_motor_strength

                        else:
                            self.p_gains[env_id][i] = self.cfg.control.stiffness[dof_name] * rand_motor_strength
                            self.d_gains[env_id][i] = self.cfg.control.damping[dof_name] * rand_motor_strength
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



        # self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # cprint(f"unsqueeze: {self.default_dof_pos.shape,  self.default_dof_pos}", 'red', attrs=['bold'])

        return props


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)


        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self.disturbance_force = self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1],
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1],
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if control_type == "P":
            torques = (
                    self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                    - self.d_gains * self.dof_vel
            )


        elif control_type == "actuator_net":

            if self.cfg.domain_rand.added_lag_timesteps:
                self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
                self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
            else:
                self.joint_pos_target = actions_scaled + self.default_dof_pos

            self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets
            self.joint_vel = self.dof_vel

            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)

            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
            # scale the output
            torques = torques * self.motor_strengths

        elif control_type == "V":
            torques = (
                    self.p_gains * (actions_scaled - self.dof_vel)
                    - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _compute_poses(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale  # wo - current pos is better than w - current pos
        target_poses = actions_scaled + self.default_dof_pos
        return torch.clip(target_poses, self.dof_pos_limits[:, :, 0], self.dof_pos_limits[:, :, 1])

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:cfg.control.1.5 x default positions.
        Velocities are set to zero.
        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32),
                                              len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1., 1., (len(env_ids), 2),device=self.device)
            # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32),
                                                     len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2),
                                                    device=self.device)  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        return self.root_states[:, 7:9]



    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                          -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        if self.cfg.control.control_type == "POSE":
            # Note: Position control
            torques = self.gym.acquire_dof_force_tensor(self.sim)
            self.target_poses = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
            self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_actions)

            # self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            # self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        else:
            self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                       requires_grad=False)

        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0

        self.extras = {}
        self.noise_scale_vec,self.noise_noise_vel = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # Note: torque get directly from gym

        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_actions_2 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)

        self.ref_vt = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                      requires_grad=False)

        self.ref_qt_1 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.ref_qt_2 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.ref_qt_3 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.ref_qt_5 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.ref_qt_10 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.ref_qt_20 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        #### add last_information ##################
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_contact_forces = torch.zeros_like(self.contact_forces)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])


        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # print(self.gravity_vec)
        # print(self.projected_gravity)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.disturbance_force = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)  # x vel, y vel



        foot_height  = self.rigid_body_state[:, self.feet_indices, 2:3]
        foot_vel = self.rigid_body_state[:, self.feet_indices, 7:9]

        self.foot_height = torch.cat((foot_height[:, 0, ],
                                      foot_height[:, 1, ],
                                      foot_height[:, 2, ],
                                      foot_height[:, 3, ],
                                      ), dim=-1)
        self.foot_vel = torch.cat((foot_vel[:, 0, ],
                                   foot_vel[:, 1, ],
                                   foot_vel[:, 2, ],
                                   foot_vel[:, 3, ],
                                      ), dim=-1)

        # Additionally initialize actuator network hidden state tensors
        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.added_lag_timesteps + 1)]

        # if self.cfg.control.control_type == "POSE":
        # init pos err and vel buffer(12 DOF)
        self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)
        self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)
        self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)
        self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))


        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
    def _prepare_eval_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales
        to_remove = []
        for name, scale in self.eval_scales.items():
            if not scale:
                to_remove.append(name)
        for name in to_remove:
            self.eval_scales.pop(name)

        # prepare list of functions
        self.eval_functions = []
        self.eval_names = []
        for name, scale in self.eval_scales.items():
            print(name, scale)
            if name == "termination":
                continue
            self.eval_names.append(name)
            name = "_eval_" + name
            self.eval_functions.append(getattr(self, name))

        # reward episode sums
        self.eval_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.eval_scales.keys()
        }

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle meshes terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)


    def _create_stairs(self):
        """ Adds a triangle meshes terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.asset_list = []
        self.asset_size_list = []
        if self.cfg.asset.name == "morph_dog":
            cprint('General Policy', 'blue', attrs=['bold'])
            asset_files = os.listdir(asset_path)
            # asset_files.sort(reverse=False)

            # cprint(f"unsqueeze: {asset_path, asset_files}", 'red', attrs=['bold'])

            for asset_file in asset_files:
                asset = self.gym.load_asset(self.sim, asset_path, asset_file, asset_options)
                self.asset_list.append(asset)
                self.asset_size_list.append(asset_file)
            # cprint(f"unsqueeze: {self.asset_size_list}", 'red', attrs=['bold'])

            robot_asset = self.asset_list[0]
        elif self.cfg.asset.name == "random_dog":# train a general policy. all morph in the morph list
            self.asset_name = self.cfg.asset.asset_name

            for j in range(len(self.asset_name)):
                asset_root = os.path.join(asset_path,  self.asset_name[j], 'urdf')
                asset_file = self.asset_name[j] + '.urdf'
                robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
                self.asset_list.append(robot_asset)

        else:
            asset_root = os.path.dirname(asset_path)
            asset_file = os.path.basename(asset_path)
            robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            self.asset_list.append(robot_asset) # still append specific config to the list

        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        # dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        # rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)

        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        self.foot_name = feet_names
        self.body_names = body_names
        self.mass = torch.zeros(self.num_bodies, device=self.device, requires_grad=False)



        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.coms = []
        # if self.cfg.control.control_type != "actuator_network":
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)

        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                               requires_grad=False)

        self.default_dof_pos = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)

        self.dof_pos_limits = torch.zeros(self.num_envs, self.num_actions, 2, dtype=torch.float, device=self.device,
                                          requires_grad=False)

        self.dof_vel_limits = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.torque_limits = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                         requires_grad=False)

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # ! For Morphology --> group ID
            if self.cfg.asset.name == "morph_dog":
                robot_type_id = i % len(self.asset_list)
            else:
                robot_type_id = i % len(self.asset_list)


            rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.asset_list[robot_type_id])
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)

            self.gym.set_asset_rigid_shape_properties(self.asset_list[robot_type_id], rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, self.asset_list[robot_type_id], start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)


            if self.cfg.asset.name == "morph_dog":
                path = os.path.join(asset_path, self.asset_size_list[robot_type_id])
                trunk_size, thigh_size, calf_size = Agent(path).get_body_size()
                self._process_rigid_body_size_props(trunk_size, thigh_size, calf_size, i)


            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i, robot_type_id)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            dof_props_asset = self.gym.get_asset_dof_properties(self.asset_list[robot_type_id])
            dof_props = self._process_dof_props(dof_props_asset, i, robot_type_id)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)  # Note: important to read torque !!!!



            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)


        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh", "trimesh_on_stair"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        np.set_printoptions(precision=4)

        # draw height lines
        if self.cfg.terrain.measure_heights:
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                               self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        com_geom = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0.85, 0.1, 0.1))
        com_proj_geom = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0, 1, 0))
        contact_geom = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0, 0, 1))
        cop_geom = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0.85, 0.5, 0.1))
        for i in range(self.num_envs):
            # draw COM(= base pose) and its projection
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            com = self.coms[i] + base_pos
            com_pose = gymapi.Transform(gymapi.Vec3(com[0], com[1], com[2]), r=None)
            com_proj_pose = gymapi.Transform(gymapi.Vec3(com[0], com[1], 0), r=None)
            gymutil.draw_lines(com_geom, self.gym, self.viewer, self.envs[i], com_pose)
            gymutil.draw_lines(com_proj_geom, self.gym, self.viewer, self.envs[i], com_proj_pose)

            # draw contact point and COP projection
            eef_state = self.rigid_body_state[i, self.feet_indices, :3]
            contact_idxs = (self.contact_forces[i, self.feet_indices, 2] > 1.).nonzero(as_tuple=False).flatten()
            contact_state = eef_state[contact_idxs]
            contact_force = self.contact_forces[i, self.feet_indices[contact_idxs], 2]

            for i_feet in range(contact_state.shape[0]):
                contact_pose = contact_state[i_feet, :]
                sphere_pose = gymapi.Transform(gymapi.Vec3(contact_pose[0], contact_pose[1], 0), r=None)
                gymutil.draw_lines(contact_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

            # calculate and draw COP
            cop_sum = torch.sum(contact_state * contact_force.view(contact_force.shape[0], 1), dim=0)
            cop = cop_sum / torch.sum(contact_force)
            sphere_pose = gymapi.Transform(gymapi.Vec3(cop[0], cop[1], 0), r=None)
            gymutil.draw_lines(cop_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

            # # draw inverted pendulum
            # cop = cop.cpu().numpy()
            # self.gym.add_lines(self.viewer, self.envs[i], 1, [cop[0], cop[1], 0, com[0], com[1], com[2]], [0.85, 0.1, 0.5])

            # draw connected line between contact point (fault case and normal case)
            self._draw_contact_polygon(contact_state, self.envs[i])

    def _draw_contact_polygon(self, contact_state, env_handle):
        contact_num = contact_state.shape[0]
        if contact_num >= 2:
            if contact_num == 4:  # switch the order of rectangle
                contact_state = contact_state[[0, 1, 3, 2], :]
            polygon_start = contact_state[0].cpu().numpy()

            width, n_lines = 0.01, 10  # make it thicker
            polygon_starts = []
            for i_line in range(n_lines):
                polygon_starts.append(polygon_start.copy())
                polygon_start += np.array([0, 0, width / n_lines])
            for i_feet in range(contact_num):
                polygon_end = contact_state[(i_feet + 1) % contact_num, :].cpu().numpy()

                polygon_ends = []
                polygon_vecs = []
                for i_line in range(n_lines):
                    polygon_ends.append(polygon_end.copy())
                    polygon_end += np.array([0, 0, width / n_lines])
                    polygon_vecs.append(
                        [polygon_starts[i_line][0], polygon_starts[i_line][1], polygon_starts[i_line][2],
                         polygon_ends[i_line][0], polygon_ends[i_line][1], polygon_ends[i_line][2]])
                self.gym.add_lines(self.viewer, env_handle, n_lines,
                                   polygon_vecs,
                                   n_lines * [0.85, 0.1, 0.1])

                polygon_starts = polygon_ends

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)
        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw
        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.
        Raises:
            NameError: [description]
        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain meshes type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
            self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        # print("lin_vel_z", self.base_lin_vel)
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        # return torch.sum(torch.square(self.base_ang_vel[:, 1].unsqueeze(1)), dim=1)
        # print("ang_vel_xy:", self.base_ang_vel)
        return torch.sum(torch.square(self.base_ang_vel[:, 0:2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        # print(torch.square(self.projected_gravity[:, :2]))
        # print(torch.sign(self.projected_gravity[:, 0]))
        # print("projected_gravity:", self.projected_gravity)
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation_1(self):
        # Penalize non flat base orientation
        # print(torch.square(self.projected_gravity[:, :2]))
        # print(torch.sign(self.projected_gravity[:, 0]))
        return torch.sum(torch.square(self.projected_gravity[:, 0]).unsqueeze(1), dim=1)

    def _reward_orientation_23(self):
        # Penalize non flat base orientation
        # print(torch.square(self.projected_gravity[:, :2]))
        # print(torch.sign(self.projected_gravity[:, 0]))
        return torch.sum(torch.square(self.projected_gravity[:, 1:3]), dim=1)

    def _reward_orientation_2(self):
        # Penalize non flat base orientation
        # print(torch.square(self.projected_gravity[:, :2]))
        # print(torch.sign(self.projected_gravity[:, 0]))
        return torch.sum(torch.square(self.projected_gravity[:, 1]).unsqueeze(1), dim=1)

    def _reward_orientation_3(self):
        # Penalize non flat base orientation
        # print(torch.square(self.projected_gravity[:, :2]))
        # print(torch.sign(self.projected_gravity[:, 0]))
        return torch.sum(torch.square(self.projected_gravity[:, 2]).unsqueeze(1), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        # print('***********', self.root_states[:, 2].unsqueeze(1).shape, self.measured_heights.shape, (self.root_states[:, 2].unsqueeze(1) - self.measured_heights).shape)
        # base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1), dim=1)
        # print(base_height)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_motion(self):
        # cosmetic penalty for motion
        return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)

    # ***************** energy disspation ***************
    def _reward_energy(self):
        # Penalize energy
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)return torch.square(self.base_lin_vel[:, 2])
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time_base(self):
        # Reward long steps
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        first_contact = (self.feet_air_time > 0.) * contact
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact,
                                dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact
        return rew_airTime

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
                torch.norm(self.commands[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
                    torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_motion_base(self):
        # cosmetic penalty for motion
        return torch.sum(torch.square(self.dof_pos[:,: ] - self.default_dof_pos[:, :]), dim=1)

    def _reward_hip_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)

    def _reward_thigh_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [1, 4, 7, 10]] - (self.default_dof_pos[:, [1, 4, 7, 10]] )), dim=1)
        # return torch.sum(torch.abs(self.dof_pos[:, [1, 4, 7, 10]] - (self.default_dof_pos[:, [1, 4, 7, 10]])), dim=1)

    def _reward_calf_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [2, 5, 8, 11]] - self.default_dof_pos[:, [2, 5, 8, 11]]), dim=1)

    ############## Motion Functions ########

    def _reward_f_hip_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [0, 3]] - self.default_dof_pos[:, [0, 3]]), dim=1)

    def _reward_r_hip_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [6, 9]] - self.default_dof_pos[:, [6, 9]]), dim=1)

    def _reward_f_thigh_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [1, 4]] - self.default_dof_pos[:, [1, 4]]), dim=1)

    def _reward_r_thigh_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [7, 10]] - self.default_dof_pos[:, [7, 10]]), dim=1)

    def _reward_f_calf_motion(self):
        # cosmetic penalty for hip motion
        return torch.sum(torch.abs(self.dof_pos[:, [2, 5]] - self.default_dof_pos[:, [2, 5]]), dim=1)

    def _reward_r_calf_motion(self):
        # cosmetic penalty for hip motion
        # return torch.sum(torch.abs(self.dof_pos[:, [8, 11]] - self.default_dof_pos[:, [8, 11]] + torch.tensor([[1.57, 1.57]] * self.num_envs, dtype=torch.float, device=self.device)), dim=1)
        return torch.sum(torch.abs(self.dof_pos[:, [8, 11]] - self.default_dof_pos[:, [8, 11]]), dim=1)
    ############## Trackong Gait Command ########

    def _reward_flrl_gait_diff(self): # left joints difference
        return torch.sum(torch.square(self.dof_pos[:, [0, 1, 2]] - self.dof_pos[:, [6, 7, 8]]), dim=1)

    def _reward_frrr_gait_diff(self): # right joints difference
        return torch.sum(torch.square(self.dof_pos[:, [3, 4, 5]] - self.dof_pos[:, [9, 10, 11]]), dim=1)

    def _reward_flrl_gait_diff_2(self): # left joints difference
        return torch.sum(torch.square(self.dof_vel[:, [0, 1, 2]] - self.dof_vel[:, [6, 7, 8]]), dim=1)

    def _reward_frrr_gait_diff_2(self): # right joints difference
        return torch.sum(torch.square(self.dof_vel[:, [3, 4, 5]] - self.dof_vel[:, [9, 10, 11]]), dim=1)

    def _reward_flfr_gait_diff(self): # left joints difference
        return torch.sum(torch.square(self.dof_pos[:, [0, 1, 2]] - self.dof_pos[:, [3, 4, 5]]), dim=1)

    def _reward_rlrr_gait_diff(self): # right joints difference
        return torch.sum(torch.square(self.dof_pos[:, [6, 7, 8]] - self.dof_pos[:, [9, 10, 11]]), dim=1)

    def _reward_flfr_gait_diff_2(self): # left joints difference
        return torch.sum(torch.square(self.dof_vel[:, [0, 1, 2]] - self.dof_vel[:, [3, 4, 5]]), dim=1)

    def _reward_rlrr_gait_diff_2(self): # right joints difference
        return torch.sum(torch.square(self.dof_vel[:, [6, 7, 8]] - self.dof_vel[:, [9, 10, 11]]), dim=1)

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
        # for i in [2,3]:
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.cfg.rewards.tracking_sigma))
        return reward / 4

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

    def _reward_fl_soft_constraint(self):
        # foot_heights = self.rigid_body_state[:, self.feet_indices, 2]
        fl_soft_constraint = torch.sum(torch.tensor([0.01, 0.01, 0.98],dtype=torch.float, device=self.device)*torch.square(torch.tensor([[0., 0.9, -2.1]]*self.num_envs, dtype=torch.float, device=self.device) - self.dof_pos[:, 0:3]))
        # fl_height = (foot_heights[:, 0] > 0.2) * 0.02

        return fl_soft_constraint

    def _reward_tracking_contacts_shaped_vel(self):
        self.foot_velocities = self.rigid_body_state[:, self.feet_indices, 7:10]
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
        # for i in [2,3]:
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.cfg.rewards.tracking_sigma)))
        return reward / 4

    ############## Imitation Learning ########

    def _reward_tracking_ref_joint_pos(self):
        ref_last_q_err = torch.sum(torch.abs(self.last_dof_pos - self.ref_joint_pos[self.episode_length_buf-1]), dim=1)
        ref_q_err = torch.sum(torch.abs(self.dof_pos - self.ref_joint_pos[self.episode_length_buf]), dim=1)
        ref_q_err = (ref_q_err + ref_last_q_err) / 2
        return torch.exp(- 1e-2 * ref_q_err / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ref_joint_vel(self):
        ref_last_qd_err = torch.sum(torch.abs(self.last_dof_vel - self.ref_joint_vel[self.episode_length_buf-1]), dim=1)
        ref_qd_err = torch.sum(torch.abs(self.dof_vel - self.ref_joint_vel[self.episode_length_buf]), dim=1)
        ref_qd_err = (ref_qd_err + ref_last_qd_err) / 2
        return torch.exp(- 1e-2 *ref_qd_err / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ref_body_pos(self):
        self.local_base_pos = quat_rotate_inverse(self.init_base_quat, self.root_states[:,:3] - self.init_base_pos)
        ref_pos_error = torch.sum(torch.abs(self.ref_body_pos[self.episode_length_buf][:, :3] - self.local_base_pos[:, :3]), dim=1)
        return torch.exp(- 1e-3 *ref_pos_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ref_body_quat(self):
        ref_quat_error = torch.sum(torch.abs(self.root_states[:,3:7] - self.ref_body_quat[self.episode_length_buf]), dim=1)
        return torch.exp(- 1e-3 *ref_quat_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ref_body_vel(self):
        # ref_vel_error = torch.sum(torch.abs(self.ref_body_vel[self.episode_length_buf][:, :3] * (torch.ones(self.num_envs, device=self.device) - 0.5 * self.commands[:, 0]).unsqueeze(-1)- self.base_lin_vel[:, :3]), dim=1)
        ref_vel_error = torch.sum(torch.abs(self.ref_body_vel[self.episode_length_buf][:, :3] * (torch.ones(self.num_envs, device=self.device) + self.commands[:, 0]).unsqueeze(-1)- self.base_lin_vel[:, :3]), dim=1)
        return torch.exp(- 1e-1 *ref_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ref_body_omega(self):
        ref_vel_error = torch.sum(torch.abs(self.ref_body_omega[self.episode_length_buf][:, :3] - self.base_ang_vel[:, :3]), dim=1)
        return torch.exp(- 1e-1 *ref_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ref_both_pos(self):
        self.local_base_pos = quat_rotate_inverse(self.init_base_quat, self.root_states[:, :3] - self.init_base_pos)
        both_pos_error = torch.sum(torch.abs(self.ref_body_pos[self.episode_length_buf][:, :3] - self.local_base_pos[:, :3]), dim=1) + 0.5* torch.sum(torch.abs(self.dof_pos - self.ref_joint_pos[self.episode_length_buf]), dim=1)
        return torch.exp(- 1e-3 *both_pos_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ref_both_vel(self):
        both_vel_error = torch.sum(torch.abs(self.ref_body_vel[self.episode_length_buf][:, :3] - self.base_lin_vel[:, :3]), dim=1) + 0.1* torch.sum(torch.abs(self.dof_vel - self.ref_joint_vel[self.episode_length_buf]), dim=1)
        return torch.exp(- 1e-3 *both_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ref_feet_pos(self):
        # self.quat_local_world = self.root_states[:, 3:7] * torch.tensor([-1, -1, -1, 1], device=self.device)
        self.local_eef_state = self.rigid_body_state[:, self.feet_indices, :3] - self.root_states[:, :3].unsqueeze(1)
        self.local_eef_state[:, 0] = quat_rotate_inverse(self.root_states[:, 3:7], self.local_eef_state[:, 0])
        self.local_eef_state[:, 1] = quat_rotate_inverse(self.root_states[:, 3:7], self.local_eef_state[:, 1])
        self.local_eef_state[:, 2] = quat_rotate_inverse(self.root_states[:, 3:7], self.local_eef_state[:, 2])
        self.local_eef_state[:, 3] = quat_rotate_inverse(self.root_states[:, 3:7], self.local_eef_state[:, 3])
        # print(self.local_eef_state)
        ref_feet_pos_err = torch.sum(torch.abs(self.local_eef_state - self.ref_feet_pos[self.episode_length_buf]))
        return torch.exp(- 1e-6*ref_feet_pos_err / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ref_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        # print(foot_forces)
        reward = torch.exp(-1e-9 * torch.sum((foot_forces[:, :]-self.ref_foot_forces[self.episode_length_buf]) ** 2) / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_tracking_ref_shaped_vel(self):
        self.foot_velocities = self.rigid_body_state[:, self.feet_indices, 7:10]
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        reward = torch.exp(- 1e-6 *torch.sum((foot_velocities[:, :]-self.ref_foot_velocities[self.episode_length_buf]) ** 2) / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_tracking_com_pos(self):
        self.rigid_body_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,0:3]
        self.com_pos = torch.sum(self.rigid_body_pos * self.mass.unsqueeze(-1),dim=1)/torch.sum(self.mass)
        self.com_pos = quat_rotate_inverse(self.root_states[:, 3:7], self.com_pos-self.root_states[:, 0:3])
        com_err = self.com_pos-self.ref_com_pos[self.episode_length_buf]
        return torch.exp( -torch.sum(( self.com_pos-self.ref_com_pos[self.episode_length_buf])**2)/ self.cfg.rewards.tracking_sigma)               

    def _reward_tracking_body_roll_vel(self):
        body_roll_vel_err = torch.sum(torch.square(self.base_ang_vel[:, 0] - self.ref_roll_vel[self.episode_length_buf]))

        return torch.exp(-1e-5 *body_roll_vel_err / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_body_pitch_vel(self):
        body_pitch_vel_err =  torch.sum(torch.square(self.base_ang_vel[:, 1] - self.ref_pitch_vel[self.episode_length_buf]))
        return torch.exp(-1e-5 *body_pitch_vel_err / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_body_yaw_vel(self):
        body_yaw_vel_err =  torch.sum(torch.square(self.base_ang_vel[:, 0] - self.ref_body_omega[self.episode_length_buf][:, 0]))
        # print("-----------------------")
        # print(self.base_ang_vel[:, 0])
        # print(self.ref_body_omega[self.episode_length_buf][:, 0])
        return torch.exp(-1e-0 *body_yaw_vel_err / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_body_x_vel(self):
        body_x_vel_err =  torch.sum(torch.square(self.base_lin_vel[:, 0] - self.ref_body_vel[self.episode_length_buf]))
        return torch.exp(-1e-0 *body_x_vel_err / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_body_y_vel(self):
        body_y_vel_err =  torch.sum(torch.square(self.base_lin_vel[:, 1] - self.ref_body_vel[self.episode_length_buf][:, 1]))
        return torch.exp(-1e-0 *body_y_vel_err / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_body_z_vel(self):
        body_z_vel_err =  torch.sum(torch.square(self.base_lin_vel[:, 2] - self.ref_body_vel[self.episode_length_buf][:, 2]))
        return torch.exp(-1e-0 *body_z_vel_err / self.cfg.rewards.tracking_sigma)

    ############## Aggressive Maneuver ########

    def _reward_body_roll_vel(self):
        # return torch.abs(self.base_ang_vel[:, 0])
        return torch.abs(self.base_ang_vel[:, 0])

    def _reward_body_pitch_vel(self):
        # return -(self.base_ang_vel[:, 1])
        return -(self.base_ang_vel[:, 1])

    def _reward_body_yaw_vel(self):
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_body_roll_yaw_vel(self):
        return torch.square(torch.abs(self.base_ang_vel[:, 0]) + torch.abs(self.base_ang_vel[:, 2]))

    def _reward_body_pitch_yaw_vel(self):
        return torch.square(torch.abs(self.base_ang_vel[:, 1]) + torch.abs(self.base_ang_vel[:, 2]))

    def _reward_body_lin_vel_z(self):
        return +(self.base_lin_vel[:, 2])

    def _reward_body_lin_vel_y(self):
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_body_lin_vel_x(self):
        return torch.square(self.base_lin_vel[:, 0])

    # ------------ _change_cmds functions----------------

    def _change_cmds(self, vx, vy, vang):
        # change command_ranges with the input
        self.commands[:, 0] = vx
        self.commands[:, 1] = vy
        self.commands[:, 2] = vang
        # self.commands[:, 3] = heading

    # ************************ RMA specific ************************
    def reset(self):
        super().reset()
        return self.obs_dict



    def _setup_priv_option_config(self, p_config):
        self.enable_priv_enableMeasuredVel = p_config.enableMeasuredVel
        self.enable_priv_measured_height = p_config.enableMeasuredHeight
        self.enable_priv_disturbance_force = p_config.enableForce
        self.enable_priv_friction_center = p_config.enableFrictionCenter

        # enable property info (constant value during training)
        self.enable_priv_bodys_mass = p_config.enableBodysMass
        self.enable_priv_bodys_size = p_config.enableBodysSize
        self.enable_priv_bodys_enableMotorStrength = p_config.enableMotorStrength

    def _update_priv_buf(self, env_id, name, value, lower=None, upper=None):
        # normalize to -1, 1
        s, e = self.priv_info_dict[name]
        if eval(f'self.cfg.domain_rand.randomize_{name}'):
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            if type(lower) is list or upper is list:
                lower = to_torch(lower, dtype=torch.float, device=self.device)
                upper = to_torch(upper, dtype=torch.float, device=self.device)
            if lower is not None and upper is not None:
                value = (2.0 * value - upper - lower) / (upper - lower)
            self.priv_info_buf[env_id, s:e] = value
        else:
            self.priv_info_buf[env_id, s:e] = 0



    def _allocate_task_buffer(self, num_envs):
        super()._allocate_task_buffer(num_envs)

    ############## Dream ########
    def _reward_power_joint(self):
        a = torch.norm(self.torques[:, ], p=1, dim=1)
        b = torch.norm(self.dof_vel[:, ], p=1, dim=1)
        r = a * b
        return r

    def _reward_dream_smoothness(self):
        return torch.sum(torch.square(self.actions - 2 * self.last_actions + self.last_actions_2), dim=1)

    def _reward_power_distribution(self):
        r = torch.mul(self.torques[:, ], self.dof_vel[:, ])
        # d = torch.square(r)

        c = torch.var(r, dim=1)
        d = torch.sum(torch.abs(r), dim=1)
        return d


    def _reward_foot_clearance(self):
        self.re_foot = torch.zeros_like(self.foot_height)
        for i in range(0, 4):
            real_foot_height = torch.mean(self.foot_height[:, i].unsqueeze(1) - self.measured_heights, dim=1)
            c = torch.sum(torch.abs(self.foot_vel[:, 2 * i: 2 * i + 2]), dim=1)
            b = torch.norm(self.foot_vel[:, 2 * i: 2 * i + 2], p=1, dim=1)
            self.re_foot[:, i] = torch.mul(torch.square(real_foot_height - self.cfg.rewards.foot_height_target), c)

        foot_reward = torch.sum(self.re_foot, dim=1)
        return foot_reward

    def _reward_foot_height(self):
        self.re_foot = torch.zeros_like(self.foot_height)
        for i in range(0, 4):
            real_foot_height = torch.mean(self.foot_height[:, i].unsqueeze(1) - self.measured_heights, dim=1)

            self.re_foot[:, i] = torch.square(real_foot_height - self.cfg.rewards.foot_height_target)

        b = torch.sum(torch.square(self.re_foot[:, ]), dim=1)
        foot_reward = torch.exp(-10 * b)

        return foot_reward


    def _reward_orientation_base(self):
        return torch.sum(torch.square(self.projected_gravity), dim=1)

    def _reward_f_eef_height(self):
        eef_state = self.rigid_body_state[:, self.feet_indices, :3]
        # print(torch.sum(eef_state[:, 0, 2] + eef_state[:, 1, 2]))
        return (eef_state[:, 0, 2] + eef_state[:, 1, 2])

    def _reward_r_eef_height(self):
        eef_state = self.rigid_body_state[:, self.feet_indices, :3]
        # print(torch.sum(eef_state[:, 0, 2] + eef_state[:, 1, 2]))
        return (eef_state[:, 2, 2] + eef_state[:, 3, 2])

    ############## FT-Net ########
    def _reward_inv_pendulum(self):
        eef_state = self.rigid_body_state[:, self.feet_indices, :3]
        contact_force = self.contact_forces[:, self.feet_indices, 2]
        # cop = torch.sum(eef_state * contact_force.unsqueeze(-1),dim=1) * torch.tensor([[1,1,0]]*self.num_envs,device=self.device, requires_grad=False)/ \
        #       (0.00001*torch.ones(self.num_envs, device=self.device, requires_grad=False) + torch.sum(contact_force, dim=1)).unsqueeze(-1)
        # print(cop)
        cop = (eef_state[:,2,:] + eef_state[:,3,:])*torch.tensor([[1,1,0]]*self.num_envs,device=self.device, requires_grad=False)/2
        # cop = (eef_state[:,0,:] + eef_state[:,1,:])*torch.tensor([[1,1,0]]*self.num_envs,device=self.device, requires_grad=False)/2
        com = torch.sum(self.rigid_body_state[:,:,:3]*self.mass.unsqueeze(-1),dim=1) / \
              torch.sum(self.mass,dim=0)
        pendulum_vecs = com-cop
        pendulum_vec_norms = torch.norm(com-cop,dim=1)
        # print(torch.sign(pendulum_vecs[:,2]/( 0.00001*torch.ones(self.num_envs, device=self.device, requires_grad=False)+ pendulum_vec_norms))*torch.pow(torch.abs(pendulum_vecs[:,2]/( 0.00001*torch.ones(self.num_envs, device=self.device, requires_grad=False)+ pendulum_vec_norms)), 0.6))
        # return torch.sign(pendulum_vecs[:,2]/( 0.00001*torch.ones(self.num_envs, device=self.device, requires_grad=False)+ pendulum_vec_norms))*torch.pow(torch.abs(pendulum_vecs[:,2]/( 0.00001*torch.ones(self.num_envs, device=self.device, requires_grad=False)+ pendulum_vec_norms)), 1.6)
        # print("pendulum ang", torch.pow(torch.acos(pendulum_vecs[:,2]/(0.00001*torch.ones(self.num_envs, device=self.device, requires_grad=False)+pendulum_vec_norms)),2))
        self.pen_len = pendulum_vec_norms
        self.pen_ang = torch.acos(pendulum_vecs[:,2]/(0.00001*torch.ones(self.num_envs, device=self.device, requires_grad=False)+pendulum_vec_norms))

        return torch.pow(torch.acos(pendulum_vecs[:,2]/(0.00001*torch.ones(self.num_envs, device=self.device, requires_grad=False)+pendulum_vec_norms)),2)

    def _reward_inv_pendulum_acc(self):
        eef_state = self.rigid_body_state[:, self.feet_indices, :3]
        contact_force = self.contact_forces[:, self.feet_indices, 2]
        # cop = torch.sum(eef_state * contact_force.unsqueeze(-1),dim=1)* torch.tensor([[1,1,0]]*self.num_envs,device=self.device, requires_grad=False) / \
        #       (0.00001*torch.ones(self.num_envs, device=self.device, requires_grad=False) + torch.sum(contact_force, dim=1)).unsqueeze(-1)
        cop = (eef_state[:,2,:] + eef_state[:,3,:])*torch.tensor([[1,1,0]]*self.num_envs,device=self.device, requires_grad=False)/2
        # cop = (eef_state[:,0,:] + eef_state[:,1,:])*torch.tensor([[1,1,0]]*self.num_envs,device=self.device, requires_grad=False)/2
        com = torch.sum(self.rigid_body_state[:,:,:3]*self.mass.unsqueeze(-1),dim=1) / \
              torch.sum(self.mass,dim=0)
        # print(com)
        pendulum_vecs = com-cop
        pendulum_vec_norms = torch.norm(com-cop,dim=1)
        # return torch.sin(torch.acos(pendulum_vecs[:,2]/pendulum_vec_norms))/pendulum_vec_norms
        # print("pendulum acc",(torch.ones(self.num_envs,device=self.device, requires_grad=False)-torch.pow(pendulum_vecs[:,2]/pendulum_vec_norms, 2))/torch.pow(pendulum_vec_norms, 2))
        self.pen_ang_acc = (torch.ones(self.num_envs,device=self.device, requires_grad=False)-torch.pow(pendulum_vecs[:,2]/pendulum_vec_norms, 2))/torch.pow(pendulum_vec_norms, 2)

        return (torch.ones(self.num_envs,device=self.device, requires_grad=False)-torch.pow(pendulum_vecs[:,2]/pendulum_vec_norms, 2))/torch.pow(pendulum_vec_norms, 2)

    def _reward_inv_pendulum_both(self):
        return torch.abs(self.pen_ang) + torch.abs(self.pen_ang_acc)

    def _reward_support_polygon(self):
        eef_state = self.rigid_body_state[:, self.feet_indices, :3]
        base_pos = self.root_states[:, :3]
        contact_bool = (self.contact_forces[:, self.feet_indices, 2] > 1.)#.nonzero(as_tuple=False)
        contact_force = self.contact_forces[:, self.feet_indices, 2]
        cop = torch.sum(eef_state * contact_force.unsqueeze(-1),dim=1)/torch.sum(contact_force, dim=1).unsqueeze(-1)
        # print(eef_state)
        # print("-----------------------")
        # print(base_pos)
        # print("--------------------------------")
        # print(contact_force)
        # print("-------------------------------")
        com = torch.sum(self.rigid_body_state[:,:,:3]*self.mass.unsqueeze(-1),dim=1)/torch.sum(self.mass,dim=0)
        eef_state = self.rigid_body_state[:, self.feet_indices, :3]
        base_pos = self.root_states[:, :3]
        contact_bool = (self.contact_forces[:, self.feet_indices, 2] > 1.)#.nonzero(as_tuple=False)
        # is_contact = torch.any(self.contact_forces[:, self.feet_indices, 2] > 1., dim=1)
        # rt = torch.ones(self.num_envs) * is_contact
        contact_num = torch.sum(contact_bool,dim=1)
        rew_mask = torch.pow(((contact_num[:]>=3).to(torch.int32)+(contact_num[:]>=2).to(torch.int32)-(contact_num[:]>=4).to(torch.int32)).to(torch.int32),3)*(0.125)

        clw_idx = torch.tensor([0,1,3,2], device=self.device, requires_grad=False)
        nst_idx = torch.tensor([1,2,3,0], device=self.device, requires_grad=False)
        leg_mask = self.total_fault_leg_mask
        # print("#########eef_vec_norms", eef_vec_norms)
        # print("#########oef_vertors",oef_vertors)
        # print("#########oef_vertors_nst",oef_vertors_nst)
        # print("--- leg_mask", leg_mask.to(torch.int32))
        leg_not_mask = (torch.ones(self.num_envs,4, device=self.device, requires_grad=False) - leg_mask).to(torch.int32)
        # print("--- leg_not_mask_stage_1", leg_not_mask*clw_idx)
        prv_idx = torch.tensor([3,0,1,2], device=self.device, requires_grad=False)
        # print("--- leg_not_mask_stage_2", (leg_mask[:, prv_idx]*clw_idx)[:,nst_idx])
        # print("--actual_idx", leg_not_mask*clw_idx+(leg_mask[:, prv_idx]*clw_idx)[:,nst_idx])
        actual_idx =  (leg_not_mask*clw_idx+(leg_mask[:, prv_idx]*clw_idx)[:,nst_idx]).to(torch.long)
        # eef_state_clw = eef_state[:, :,actual_idx]
        # print("--eef_state", eef_state)
        eef_state_clw = eef_state[:, clw_idx]
        eef_whole_state =   eef_state[:, actual_idx]
        # print("============", eef_state[:, actual_idx])
        for i in range(self.num_envs):
            if i == 0:
                eef_state_clw_new = eef_whole_state[i][i]
            else:
                eef_state_clw_new = torch.cat((eef_state_clw_new, eef_whole_state[i][i]), 0)
        # print("-------------", eef_state_clw_new.reshape(self.num_envs, 4, 3))
        eef_state_clw = eef_state_clw_new.reshape(self.num_envs, 4, 3)
        eef_state_nst = eef_state_clw[:, nst_idx]
        # print(eef_state_nst)
        eef_vectors = eef_state_nst - eef_state_clw
        # print("#########origin", eef_state)
        # print("#########clw",eef_state_clw)
        # print("#########nst",eef_state_nst)
        # print("--- eef_vectors", eef_vectors)
        # print("------------------------------------------------")
        eef_vec_norms = torch.norm(eef_vectors,dim=2)
        oef_vertors = eef_state_clw - com.unsqueeze(1).expand(self.num_envs, 4,3)
        oef_vertors_nst = oef_vertors[:,nst_idx,:]
        distances = torch.norm(torch.cross(oef_vertors_nst,oef_vertors,dim=2),dim=2)/(0.00001*torch.ones(4, device=self.device, requires_grad=False) + eef_vec_norms.unsqueeze(0))
        # print(torch.norm(torch.cross(oef_vertors_nst,oef_vertors,dim=2),dim=2)/( eef_vec_norms.unsqueeze(0)))
        min_distances = torch.max(distances,dim=2)[0]
        # print(min_distances)
        # print("polygon dist", min_distances.squeeze(0))
        self.poly_dist = distances.squeeze(0)[:, [0,2,3]]

        return min_distances.squeeze(0)

    ############## RMA ########
    def _reward_RMA_work(self):
        return torch.norm(torch.mul(self.torques, (self.dof_pos - self.last_dof_pos)),  dim=-1)

    def _reward_RMA_ground_impact(self):
        a = self.contact_forces[:, self.feet_indices, :]
        b = self.last_contact_forces[:, self.feet_indices, :]
        f1 = torch.sum(torch.sum(torch.square(a-b), dim=-1), dim=1)
        return f1

    def _reward_RMA_smoothness(self):

        a = self.torques
        b = self.last_torques
        f1 = torch.sum(torch.square(a-b), dim=1)

        return f1
    def _reward_RMA_foot_slip(self):
        g = self.contact_forces[:, self.feet_indices, 2] > 1.
        a = g.long()
        b = self.foot_vel
        f1 = torch.sum(torch.square(torch.mul(a, b)), dim=1)
        return f1

    def _reward_RMA_action_magnitude(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_feet_step(self):
        self.num_calls += 1

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)

        rb_states_3d = rb_states.reshape(self.num_envs, -1, rb_states.shape[-1])
        feet_heights = rb_states_3d[
            :, self.feet_indices, 2
        ]  # proper way to get feet heights, don't use global feet ind
        feet_heights = feet_heights.view(-1)

        xy_forces = torch.norm(
            self.contact_forces[:, self.feet_indices, :2], dim=2
        ).view(-1)
        z_forces = self.contact_forces[:, self.feet_indices, 2].view(-1)
        z_forces = torch.abs(z_forces)

        contact = torch.logical_or(
            self.contact_forces[:, self.feet_indices, 2] > 1.0,
            self.contact_forces[:, self.feet_indices, 1] > 1.0,
        )
        contact = torch.logical_or(
            contact, self.contact_forces[:, self.feet_indices, 0] > 1.0
        )

        self.last_contacts = contact
        xy_forces[feet_heights < 0.05] = 0
        z_forces[feet_heights < 0.05] = 0
        z_ans = z_forces.view(-1, 4).sum(dim=1)
        z_ans[z_ans > 1] = 1

        return z_ans


    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        ans = torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

        return ans




    # Eval Functions:
    def _eval_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        ans = torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

        return ans  # (num_robots)

    def _eval_feet_step(self):
        self.num_calls += 1

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)

        rb_states_3d = rb_states.reshape(self.num_envs, -1, rb_states.shape[-1])
        feet_heights = rb_states_3d[
                       :, self.feet_indices, 2
                       ]  # proper way to get feet heights, don't use global feet ind
        feet_heights = feet_heights.view(-1)

        xy_forces = torch.norm(
            self.contact_forces[:, self.feet_indices, :2], dim=2
        ).view(-1)
        z_forces = self.contact_forces[:, self.feet_indices, 2].view(-1)
        z_forces = torch.abs(z_forces)

        contact = torch.logical_or(
            self.contact_forces[:, self.feet_indices, 2] > 1.0,
            self.contact_forces[:, self.feet_indices, 1] > 1.0,
        )
        contact = torch.logical_or(
            contact, self.contact_forces[:, self.feet_indices, 0] > 1.0
        )

        contact_filt = torch.logical_or(contact, self.last_contacts).view(-1)
        self.last_contacts = contact
        # print("contact filt shape: ", contact_filt.shape)

        xy_forces[feet_heights < 0.05] = 0
        xy_ans = xy_forces.view(-1, 4).sum(dim=1)

        # print("lowest contacting foot: ", feet_heights[z_forces > 1].min())
        # print("highest contacting foot: ", feet_heights[z_forces > 50.0].max())
        z_forces[feet_heights < 0.05] = 0
        z_ans = z_forces.view(-1, 4).sum(dim=1)
        z_ans[z_ans > 1] = 1  # (num_robots)

        # ans = torch.ones(1024).cuda()
        return z_ans

    def _eval_crash_freq(self):
        reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )

        # print("reset buff: ", reset_buf)

        return reset_buf

    def _eval_any_contacts(self):
        stumble = self._eval_feet_stumble()
        step = self._eval_feet_step()

        return torch.logical_or(stumble, step)