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




class ProprioAdaptTConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(ProprioAdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * 3, output_size)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 50, 32)
        x = x.permute((0, 2, 1))  # (N, 32, 50)
        x = self.temporal_aggregation(x)  # (N, 32, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x


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

        mlp_input_shape = num_obs

        # ---- Priv Info ----
        self.priv_mlp = kwargs['priv_mlp_units']
        self.priv_info = kwargs['priv_info']
        self.priv_info_dim = kwargs['priv_info_dim']
        self.priv_info_stage2 = kwargs['proprio_adapt']

        self.proprio_adapt_input = num_obs
        self.proprio_adapt_output = kwargs['proprio_adapt_out_dim']
        mlp_input_shape += self.proprio_adapt_output

        if self.priv_info:
            # self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs['priv_info_dim'])
            self.env_mlp = DmEncoder(self.priv_info_dim, self.priv_mlp)
            cprint(f"Encoder MLP: {self.env_mlp}", 'green', attrs=['bold'])

            if self.priv_info_stage2:
                self.adapt_tconv = ProprioAdaptTConv(input_size=self.proprio_adapt_input,
                                                     output_size=self.proprio_adapt_output)
                cprint(f"Adaptation Conv: {self.adapt_tconv}", 'green', attrs=['bold'])
        else:
            cprint('Vanilla Actor Critic', 'green', attrs=['bold'])

        activation = get_activation(activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(53, actor_hidden_dims[0]))
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
        critic_layers.append(nn.Linear(53, critic_hidden_dims[0]))
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

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

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

    # def update_distribution(self, observations):
    #     mean = self.actor(observations)
    #     self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, obs_dict, **kwargs):
        # self.update_distribution(observations)
        mean, std, _, _, _ , _= self._actor_critic(obs_dict)
        self.distribution = Normal(mean, mean * 0. + std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        # actions_mean = self.actor(observations)
        # used for testing
        actions_mean, _, _, _, _, _ = self._actor_critic(obs_dict)
        return actions_mean

    def evaluate(self, obs_dict, **kwargs):
        # value = self.critic(critic_observations)
        _, _, value, _, _ , _= self._actor_critic(obs_dict)
        return value

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        obs_vel = obs_dict['privileged_info'][:, 0:3]
        obs_hight = obs_dict['privileged_info'][:, 11:198]
        obs_priv = obs_dict['priv_info']
        actor_obs = obs
        extrin, extrin_gt = None, None
        if self.priv_info:
            if self.priv_info_stage2:
                extrin = self.adapt_tconv(obs_dict['proprio_hist'])

                extrin_gt = self.env_mlp(obs_priv) if 'priv_info' in obs_dict else extrin

                # extrin_gt = torch.tanh(extrin_gt)
                # extrin = torch.tanh(extrin)

                actor_obs = torch.cat([obs, extrin], dim=-1)

            else:

                extrin = self.env_mlp(obs_priv)

                actor_obs = torch.cat([obs, extrin], dim=-1)



        mu = self.actor(actor_obs)
        value = self.critic(actor_obs)
        sigma = self.std

        mu1 = self.actor(actor_obs)
        return mu, mu * 0 + sigma, value, extrin, extrin_gt, mu1


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


