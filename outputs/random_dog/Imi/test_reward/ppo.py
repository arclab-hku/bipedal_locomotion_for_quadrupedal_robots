import torch
import torch.nn as nn
import torch.optim as optim
from termcolor import cprint
from rl.Imi.modules import ActorCritic
from rl.Imi.storage import RolloutStorage

from rl.Imi.modules.actor_critic import DmEncoder
import torch.nn.functional as F


class PPO:
    actor_critic: ActorCritic
    dm_encoder: DmEncoder

    def __init__(self,
                 actor_critic,
                 dm_encoder,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)

        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        ### MLP ###
        self.dm_encoder = dm_encoder
        self.dm_encoder.to(self.device)
        self.optimizer_encoder = optim.Adam(self.dm_encoder.parameters(), lr=learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape,
                     Hist_info_shape):
        print("**********  Hist_info_shape  ", Hist_info_shape)
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
                                      action_shape, Hist_info_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs_dict):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs_dict).detach()
        self.transition.values = self.actor_critic.evaluate(obs_dict).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        ##############add loss #############
        self.transition.extrin_loss = self.actor_critic.extrin_loss(obs_dict).detach()

        # need to record obs before env.step()
        self.transition.observations = obs_dict['obs']

        # privileged_info: need to record / update the vel info
        self.transition.privileged_info = obs_dict['privileged_info']

        # proprio_hist: need to record / update the his of the obs
        self.transition.proprio_hist = obs_dict['proprio_hist']

        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_vel_loss = 0
        mean_com_cop_loss = 0
        # mean_morph_loss = 0
        mean_contact_loss = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, privileged_info_batch, proprio_hist_batch, extrin_loss in generator:

            obs_dict_batch = {
                'obs': obs_batch,
                'privileged_info': privileged_info_batch,
                'proprio_hist': proprio_hist_batch,
            }
            self.actor_critic.act(obs_dict_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_dict_batch, masks=masks_batch,
                                                     hidden_states=hid_states_batch[1])

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            extrin_batch = self.actor_critic.extrin_loss(obs_dict_batch, masks=masks_batch,
                                                         hidden_states=hid_states_batch[1])

            real_vel = privileged_info_batch[:, 0:3]
            real_morph = privileged_info_batch[:, 200:209]
            pre_morph = extrin_batch[:, 3:]

            real_com_cop = privileged_info_batch[:, 219:222]
            real_vel_com_cop = torch.cat((real_vel,
                                          real_com_cop,
                                         ), dim=-1)

            vel_loss = F.mse_loss(extrin_batch[:, 0:3], real_vel.detach())
            com_cop_loss = F.mse_loss(extrin_batch[:, 3:6], real_com_cop.detach())
            # vel_loss = F.huber_loss(extrin_batch[:, 0:6], real_vel_com_cop.detach())
            # vel_loss = F.mse_loss(extrin_batch[:, 0:3], real_vel.detach())


            loss_encoder = vel_loss + com_cop_loss

            # Gradient step
            self.optimizer.zero_grad()
            self.optimizer_encoder.zero_grad()

            loss.backward()
            loss_encoder.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.dm_encoder.parameters(), self.max_grad_norm)
            self.optimizer_encoder.step()
            self.optimizer.step()

            mean_vel_loss += vel_loss.item()
            mean_com_cop_loss += com_cop_loss.item()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_vel_loss, mean_com_cop_loss, mean_contact_loss
