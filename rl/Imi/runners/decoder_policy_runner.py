import os
import time
import copy
import torch
import numpy as np
from termcolor import cprint
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
from rl.Imi.utils.utils import tprint, export_policy_as_jit

from rl.Imi.modules import ActorCritic
from rl.Imi.env import VecEnv
from rl.Imi.algorithms import PPO

from rl.Imi.modules.actor_critic import Decoder
from rl.Imi.modules.actor_critic import DmEncoder
import torch.optim as optim
import torch.nn.functional as F

import torch.nn as nn


class ImiDecoderPolicyRunner(object):

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.encoder_cfg = train_cfg["Encoder"]
        self.device = device
        self.env = env
        self.encoder_mlp = train_cfg["Encoder"]['encoder_mlp_units']
        self.HistoryLen = train_cfg["Encoder"]['HistoryLen']
        self.num_encoder_input = self.env.num_obs * self.HistoryLen

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(self.env.num_obs,
                                                       self.env.num_actions,
                                                       **self.policy_cfg,
                                                       **self.encoder_cfg).to(self.device)

        dm_encoder = DmEncoder(self.num_encoder_input, self.encoder_mlp)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic,
                                  dm_encoder,
                                  device=self.device,
                                  **self.alg_cfg)

        ### decoder ###
        self.decoder_mlp = train_cfg["Encoder"]['decoder_mlp_units']
        self.num_decoder_input = 210
        ### decoder ###


        self.decoder = Decoder(self.num_decoder_input, self.decoder_mlp)
        self.decoder.to(self.device)
        self.optimizer_decoder = optim.Adam(self.decoder.parameters(), lr=3e-4)

        cprint(f"Decoder MLP: {self.decoder}", 'green', attrs=['bold'])

        # decoder_name = train_cfg["Encoder"]['decoder_name']
        # ---- Output Dir ----
        self.log_dir = log_dir
        self.nn_dir = os.path.join(self.log_dir, 'stage1_nn')
        self.tb_dir = os.path.join(self.log_dir, 'stage1_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)
        self.direct_info = {}
        # ---- Misc ----
        self.batch_size = self.env.num_envs

        self.best_rewards = -1000
        self.agent_steps = 0

        # ---- Decoder loss ----
        self.decoder_loss_fn = torch.nn.GaussianNLLLoss()

        # ---- Training Misc
        self.internal_counter = 0
        self.latent_loss_stat = 0
        self.loss_stat_cnt = 0

        _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        _t = time.time()
        _last_t = time.time()
        cprint(f"checkpoint_model1: {self.encoder_cfg['checkpoint_model']}", 'green', attrs=['bold'])
        obs_dict = self.env.get_observations()

        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        entrin_buf = torch.zeros(self.env.num_envs, 5, 6, device=self.device)
        vel_buf = torch.zeros(self.env.num_envs, 2, 3, device=self.device)
        while self.agent_steps <= 1e9:
            input_dict = {
                'obs': obs_dict['obs'].detach(),
                'privileged_info': obs_dict['privileged_info'],
                'proprio_hist': obs_dict['proprio_hist'].detach(),
            }

            ### decoder ###
            e = self.alg.actor_critic.extrin_loss(input_dict)
            prev_extrin = entrin_buf[:, 1:].clone()
            cur_entrin = e.detach().clone().unsqueeze(1)
            entrin_buf = torch.cat([prev_extrin, cur_entrin], dim=1)
            actions = self.alg.actor_critic.act_inference(input_dict)
            real_vel = obs_dict['privileged_info'][:, 0:3]
            prev_vel = vel_buf[:, 1:].clone()
            cur_vel = real_vel.detach().clone().unsqueeze(1)
            vel_buf = torch.cat([prev_vel, cur_vel], dim=1)
            acc = cur_vel - prev_vel

            decoder_obs = torch.flatten(entrin_buf, start_dim=1).clone()
            pre_acc = 4 * self.decoder(torch.cat([decoder_obs, obs_dict['proprio_hist']], dim=1))
            ### decoder ###

            # loss_Gauss = nn.GaussianNLLLoss()
            # loss = loss_Gauss(pre_est_mean, real_foot_height, pre_est_var)
            # loss = self.decoder_loss_fn(pre_est_mean, real_foot_height, pre_est_var)
            loss = F.mse_loss(pre_acc, acc.squeeze(1).detach())
            # loss = F.mse_loss(pre_est_mean, real_base_linear_velocity.detach())

            self.optimizer_decoder.zero_grad()
            loss.backward()
            self.optimizer_decoder.step()

            actions = actions.detach()
            obs_dict, r, done, info = self.env.step(actions)

            # obs_dict, r, done, info = self.env.step(mu)
            self.agent_steps += self.batch_size

            # ---- statistics
            cur_reward_sum += r
            cur_episode_length += 1
            new_ids = (done > 0).nonzero(as_tuple=False)
            self.rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

            self.log_tensorboard(loss)

            if self.agent_steps % 1e6 == 0:
                self.save(os.path.join(self.nn_dir, '{}0m.pt'.format(self.agent_steps // 1e6)))
                self.save(os.path.join(self.nn_dir, 'last.pt'))

            if len(self.rewbuffer) > 0:
                mean_rewards = statistics.mean(self.rewbuffer)
                if mean_rewards > self.best_rewards:
                    self.save(os.path.join(self.nn_dir, 'best.pt'))
                    self.best_rewards = mean_rewards

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Current Best: {self.best_rewards:.2f}'
            tprint(info_string)

    def log_tensorboard(self, loss):
        if len(self.rewbuffer) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(self.rewbuffer),
                                   int(self.agent_steps))
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(self.lenbuffer),
                                   int(self.agent_steps))
            for k, v in self.direct_info.items():
                self.writer.add_scalar(f'{k}/frame', v, self.agent_steps)
        # regression loss
        self.writer.add_scalar('Loss/decoder_loss', loss.item(), self.agent_steps)

    def restore_train(self, path):
        loaded_dict = torch.load(path)
        cprint('careful, using non-strict matching', 'red', attrs=['bold'])
        self.alg.actor_critic.load_state_dict(loaded_dict['actor_state_dict'])
        self.alg.dm_encoder.load_state_dict(loaded_dict["dm_encoder_state_dict"])

        self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.alg.optimizer_encoder.load_state_dict(loaded_dict['optimizer_encoder_state_dict'])

    def save(self, path, infos=None):
        torch.save({
            'actor_state_dict': self.alg.actor_critic.state_dict(),
            "dm_encoder_state_dict": self.alg.dm_encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),

            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'optimizer_encoder_state_dict': self.alg.optimizer_encoder.state_dict(),

            'optimizer_decoder_state_dict': self.optimizer_decoder.state_dict(),

            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['actor_state_dict'])
        self.alg.dm_encoder.load_state_dict(loaded_dict["dm_encoder_state_dict"])
        ### decoder ###
        self.decoder.load_state_dict(loaded_dict["decoder_state_dict"])
        ### decoder ###
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.optimizer_encoder.load_state_dict(loaded_dict['optimizer_encoder_state_dict'])
            self.optimizer_decoder.load_state_dict(loaded_dict['optimizer_decoder_state_dict'])

        # export policy as a jit module (used to run it from C++)
        if self.cfg['export_policy']:
            cprint('Exporting policy to jit module(C++)', 'green', attrs=['bold'])
            jit_save_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'exported_s1')
            export_policy_as_jit(self.alg.actor_critic.actor, jit_save_path, 'actor.pt')
            export_policy_as_jit(self.alg.actor_critic.dm_encoder, jit_save_path, 'encoder.pt')
            export_policy_as_jit(self.decoder, jit_save_path, 'decoder.pt')

            print(f"actor: {self.alg.actor_critic.actor}")
            print(f"encoder: {self.alg.actor_critic.dm_encoder}")
        ### decoder ###
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference, self.alg.actor_critic.extrin_loss, self.decoder