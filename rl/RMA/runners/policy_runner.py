
import os
import time
import copy
import torch
import numpy as np
from termcolor import cprint
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
from rl.RMA.utils.utils import tprint, export_policy_as_jit

from rl.RMA.modules import ActorCritic
from rl.RMA.env import VecEnv


class ProprioAdaptPolicyRunner(object):

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.policy_cfg = train_cfg["policy"]
        self.encoder_cfg = train_cfg["Encoder"]
        self.device = device
        self.env = env

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        self.actor_critic: ActorCritic = actor_critic_class(self.env.num_obs,
                                                       self.env.num_actions,
                                                       **self.policy_cfg,
                                                       **self.encoder_cfg).to(self.device)

        self.actor_critic.to(self.device)
        self.actor_critic.eval()

        # ---- Output Dir ----
        self.log_dir = log_dir
        self.nn_dir = os.path.join(self.log_dir, 'stage2_nn')
        self.tb_dir = os.path.join(self.log_dir, 'stage2_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)
        self.direct_info = {}
        # ---- Misc ----
        self.batch_size = self.env.num_envs
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.best_rewards = -10000
        self.agent_steps = 0
        # ---- Optim ----
        adapt_params = []
        for name, p in self.actor_critic.named_parameters():
            if 'adapt_tconv' in name:
                adapt_params.append(p)
            else:
                p.requires_grad = False
        self.optim = torch.optim.Adam(adapt_params, lr=3e-4)
        # ---- Training Misc
        self.internal_counter = 0
        self.latent_loss_stat = 0
        self.loss_stat_cnt = 0

        _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):

        cprint(f"foot: {self.encoder_cfg['checkpoint_model']}", 'green', attrs=['bold'])

        # restore the training
        self.restore_train(self.encoder_cfg['checkpoint_model'])

        _t = time.time()
        _last_t = time.time()

        obs_dict = self.env.get_observations()

        cur_reward_sum = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)

        while self.agent_steps <= 1e9:
            input_dict = {
                'obs': obs_dict['obs'].detach(),
                'priv_info': obs_dict['priv_info'],
                'privileged_info': obs_dict['privileged_info'],
                'proprio_hist': obs_dict['proprio_hist'].detach(),
            }
            mu, _, _, e, e_gt,_ = self.actor_critic._actor_critic(input_dict)

            # cprint(f"Encoder loss: {e.shape, e_gt.shape}", 'green', attrs=['bold'])

            loss = ((e - e_gt.detach()) ** 2).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            mu = mu.detach()
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
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

            if self.agent_steps % 1e7 == 0:
                self.save(os.path.join(self.nn_dir, '{}00m.pt'.format(self.agent_steps // 1e7)))
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
            self.writer.add_scalar('Train/mean_reward', statistics.mean(self.rewbuffer), int(self.agent_steps/self.batch_size))
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(self.lenbuffer), int(self.agent_steps/self.batch_size))
            for k, v in self.direct_info.items():
                self.writer.add_scalar(f'{k}/frame', v, self.agent_steps)
        # regression loss
        self.writer.add_scalar('Loss/padapt_regression_loss', loss.item(), self.agent_steps)

    def restore_train(self, path):
        loaded_dict = torch.load(path)
        cprint('careful, using non-strict matching', 'red', attrs=['bold'])
        self.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)

    def save(self, path):
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
        }, path)

    def load(self, path):
        loaded_dict = torch.load(path)
        self.actor_critic.load_state_dict(loaded_dict['model_state_dict'])

        # export policy as a jit module (used to run it from C++)
        if self.cfg['export_policy']:
            cprint('Exporting policy to jit module(C++)', 'green', attrs=['bold'])
            path = os.path.join(os.path.dirname(os.path.dirname(path)), 'exported_s2')
            # two of the network in actor critic
            export_policy_as_jit(self.actor_critic.actor, path, 'actor.pt')
            export_policy_as_jit(self.actor_critic.adapt_tconv, path, 'adapt_tconv.pt')
            print(f"actor: {self.actor_critic.actor}")
            print(f"adapt_tconv: {self.actor_critic.adapt_tconv}")

    def get_inference_policy(self, device=None):
        self.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference