import numpy as np
from termcolor import cprint

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class DmEncoder(nn.Module):
    def __init__(self, num_encoder_obs, encoder_hidden_dims):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_encoder_obs, encoder_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[0], encoder_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[1], encoder_hidden_dims[2]),
        )

    def forward(self, dm):
        """
        Encodes depth map
        Input:
            dm: a depth map usually shape (187)
        """
        return self.encoder(dm)


class Critic_DmEncoder(nn.Module):
    def __init__(self, num_encoder_obs, encoder_hidden_dims):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_encoder_obs, encoder_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[0], encoder_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dims[1], encoder_hidden_dims[2]),
        )

    def forward(self, dm):
        """
        Encodes depth map
        Input:
            dm: a depth map usually shape (187)
        """
        return self.encoder(dm)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        # ---- Priv Info ----
        self.priv_info = kwargs['priv_info']
        self.priv_info_dim = kwargs['priv_info_dim']
        self.HistoryLen = kwargs['HistoryLen']

        num_encoder_input = num_obs * self.HistoryLen
        self.encoder_mlp = kwargs['encoder_mlp_units']

        self.dm_encoder = DmEncoder(num_encoder_input, self.encoder_mlp)

        cprint(f"Encoder MLP: {self.dm_encoder}", 'green', attrs=['bold'])

        num_actor_input = num_obs

        num_critic_input = num_obs + self.priv_info_dim  ##### 45 + 3 + 187 +19

        activation = get_activation(activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_input, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_input, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, obs_dict, **kwargs):
        # self.update_distribution(observations)
        mean, std, _, e = self._actor_critic(obs_dict)

        self.distribution = Normal(mean, mean * 0. + std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        # actions_mean = self.actor(observations)
        # used for testing
        actions_mean, _, _, _ = self._actor_critic(obs_dict)
        return actions_mean

    def evaluate(self, obs_dict, **kwargs):
        # value = self.critic(critic_observations)
        _, _, value, extrin = self._actor_critic(obs_dict)
        return value

    def extrin_loss(self, obs_dict, **kwargs):
        # value = self.critic(critic_observations)
        _, _, _, extrin = self._actor_critic(obs_dict)
        return extrin

    def _actor_critic(self, obs_dict):

        obs = obs_dict['obs']
        obs_vel = obs_dict['privileged_info'][:, 0:3]
        obs_hight = obs_dict['privileged_info'][:, 3:198]
        obs_proprio_hist = obs_dict['proprio_hist']
        obs_his = obs_proprio_hist
        obs_push = obs_dict['privileged_info'][:, 198:200]
        obs_mass = obs_dict['privileged_info'][:, 200:204]
        obs_kp_kd = obs_dict['privileged_info'][:, 204:218]
        obs_friction = obs_dict['privileged_info'][:, 218:219]

        extrin_en = self.dm_encoder(obs_his)
        pre_obs_vel = extrin_en[:, 0:3]

        actor_obs = torch.cat([obs], dim=-1)  ## 45 + 3
        critic_obs = torch.cat([obs_vel, obs, obs_hight, obs_mass, obs_push, obs_friction, obs_kp_kd],
                               dim=-1)  ## 45+3+187+19 = 219


        mu = self.actor(actor_obs)
        value = self.critic(critic_obs)
        sigma = self.std

        return mu, mu * 0 + sigma, value, extrin_en


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


