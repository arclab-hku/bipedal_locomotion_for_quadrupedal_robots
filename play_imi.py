from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from termcolor import cprint
import torch.nn.functional as F


def get_real_morph( value, lower=None, upper=None):
    real_value = value
    if lower is not None and upper is not None:
        real_value = value*(upper - lower)+lower
    return real_value

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2

    # env_cfg.terrain.terrain_length = 10.
    # env_cfg.terrain.terrain_width = 10.
    # env_cfg.terrain.num_rows = 10  # number of terrain rows (levels)
    # env_cfg.terrain.num_cols = 20  # number of terrain cols (types)


    max_init_terrain_level = 10  # starting curriculum state
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_limb_mass = False

    # fixed velocity direction evaluation (make sure the value is within the training range)
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.5, 0.5]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]


    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs_dict = env.reset()
    # load policy
    train_cfg.runner.resume = True # set the mode to be evalution
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy, pre_extrin_en = ppo_runner.get_inference_policy(device=env.device)
    # pre_extrin_en = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    # if EXPORT_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    #     print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 5 # which joint is used for logging
    stop_state_log = 500 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    pos = [0.0, 0.0, 0.45]  # x,y,z [m]
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([0.5, 0., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # import pickle as pkl

    # obs_replay = {}
    # file = open('obs_replay.pkl', 'rb')
    # obs_replay = pkl.load(file)

    for i in range(1000 * int(env.max_episode_length)):

        # obs_replay["T={}_obs".format(i)] = obs_dict["obs"].detach().cpu().numpy()
        # obs_replay["T={}_obs_hist".format(i)] = obs_dict["proprio_hist"].detach().cpu().numpy()
        # obs_replay["T={}_privileged_info".format(i)] = obs_dict["privileged_info"].detach().cpu().numpy()
        # print(obs_replay["T={}_obs".format(i)])

        actions = policy(obs_dict)
        extrin_en = pre_extrin_en(obs_dict)
        # actions = policy(obs_dict, fake_info = [obs_replay["T={}_obs".format(i)], obs_replay["T={}_obs_hist".format(i)]], device=env.device)
        # extrin_en = pre_extrin_en(obs_dict, fake_info = [obs_replay["T={}_obs".format(i)], obs_replay["T={}_obs_hist".format(i)]], device=env.device)

        # force_norm = 2.8

        # force_direction = [0., 0.78539816, 1.57079633, 2.35619449, 3.14159265, 3.92699082, 4.71238898, 5.49778714, 6.28318531]
        # if i == 150:
        #     env._push_robots_debug(force_norm, force_direction[-4])

        obs_dict, rews, dones, infos = env.step(actions.detach())

        # obs_morph = obs_dict['privileged_info'][:, 200:209]

        # pre_obs_morph = extrin_en[:, 3:]

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        a = obs_dict['privileged_info'][robot_index, 0:3]
        b = extrin_en[robot_index, 0:3]

        # obs_dict["obs"] = torch.tensor(obs_replay["T={}_obs".format(i)], device=env.device)
        # obs_dict["proprio_hist"] = torch.tensor(obs_replay["T={}_obs_hist".format(i)], device=env.device)
        # obs_dict["privileged_info"] = torch.tensor(obs_replay["T={}_privileged_info".format(i)], device=env.device)

        if i < stop_state_log:
            logger.log_states(
                {
                    'command_z': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),

                    # 'command_x': env.commands[robot_index, 0].item(),
                    # 'command_y': env.commands[robot_index, 1].item(),
                    'base_ang_x': obs_dict['obs'][robot_index, 2].item(),

                    'base_vel_x':  obs_dict['privileged_info'][robot_index, 0:1].item(),
                    'base_vel_y':  obs_dict['privileged_info'][robot_index, 1:2].item(),
                    'base_vel_z':  obs_dict['privileged_info'][robot_index, 2:3].item(),

                    'pre_vel_x':   extrin_en[robot_index, 0:1].item(),
                    'pre_vel_y':   extrin_en[robot_index, 1:2].item(),
                    'pre_vel_z':   extrin_en[robot_index, 2:3].item(),

                    'base_com_cop_x':  obs_dict['privileged_info'][robot_index, 219:220].item(),
                    'base_com_cop_y':  obs_dict['privileged_info'][robot_index, 220:221].item(),
                    'base_com_cop_z':  obs_dict['privileged_info'][robot_index, 221:222].item(),

                    'pre_com_cop_x':   extrin_en[robot_index, 3:4].item(),
                    'pre_com_cop_y':   extrin_en[robot_index, 4:5].item(),
                    'pre_com_cop_z':   extrin_en[robot_index, 5:6].item(),

                    # 'base_acc_x':  acc[robot_index, 0:1].item(),
                    # 'base_acc_y':  acc[robot_index, 1:2].item(),
                    # 'base_acc_z':  acc[robot_index, 2:3].item(),

                    # 'pre_acc_x':   pre_acc[robot_index, 0:1].item(),
                    # 'pre_acc_y':   pre_acc[robot_index, 1:2].item(),
                    # 'pre_acc_z':   pre_acc[robot_index, 2:3].item(),      
                }
            )

        # elif i == stop_state_log:
        #     logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        # elif i == stop_rew_log:
        #     logger.print_rewards()

    # file = open('obs_replay.pkl', 'wb')
    # pkl.dump(obs_replay, file)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)