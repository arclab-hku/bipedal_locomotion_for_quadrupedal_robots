

import sys
from termcolor import cprint

from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

import gym
from gym import spaces


# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1
        self.cfg = cfg
        self.num_envs = cfg.env.num_envs
        self.num_actions = cfg.env.num_actions

        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_histroy_obs = cfg.env.num_histroy_obs

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs,
                self.num_privileged_obs,
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.privileged_obs_buf = None
        self.extras = {}

        self._allocate_buffers()  # his specific buffers
        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            cprint('Enable Visualization', 'green', attrs=['bold'])
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

        self.obs_space = spaces.Box(np.ones(self.num_obs, dtype=np.float32) * -np.Inf, np.ones(self.num_obs, dtype=np.float32) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions, dtype=np.float32) * -1., np.ones(self.num_actions, dtype=np.float32) * 1.)
        self.clip_obs = cfg.normalization.clip_observations
        self.obs_dict = {}

    def get_observations(self):
        return self.obs_dict

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """Reset the environment.
        Returns:
            Observation dictionary
        """
        env_ids = self.reset_buf.nonzero().squeeze(-1)
        self.reset_idx(env_ids)
        zero_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        # step the simulator
        self.step(zero_actions)
        self.obs_dict['obs'] = torch.clip(self.obs_buf, -self.clip_obs, self.clip_obs)
        if self.num_privileged_obs is not None:
            self.obs_dict['privileged_info'] = torch.clip(self.privileged_obs_buf, -self.clip_obs, self.clip_obs)
        if self.cfg.env.train_type == "RMA":
            self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.device)
            self.obs_dict['priv_info'] = self.priv_info_buf.to(self.device)

        else:
            self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.device).flatten(1)

        return self.obs_dict

    def step(self, actions):
        raise NotImplementedError



    def _allocate_buffers(self):
        # additional buffer
        self.obs_buf_lag_history = torch.zeros((self.num_envs, self.num_histroy_obs, self.num_obs), device=self.device, dtype=torch.float)
        self._allocate_task_buffer(self.num_envs)
    def _allocate_task_buffer(self, num_envs):
        self.prop_hist_len = self.cfg.env.num_histroy_obs
        self.num_env_priv_obs = self.cfg.env.num_env_priv_obs
        if self.cfg.env.train_type == "RMA" :
            self.priv_info_buf = torch.zeros((num_envs, self.num_env_priv_obs), device=self.device, dtype=torch.float)
        self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, self.num_obs), device=self.device,
                                            dtype=torch.float)

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)