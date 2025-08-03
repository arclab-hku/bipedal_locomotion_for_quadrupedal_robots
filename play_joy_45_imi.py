from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
from termcolor import cprint
class PlayJoy():
    def __init__(self, args):
        self.args = args

        self.joy_cmd_velx = 0.4
        self.joy_cmd_vely = 0.4
        self.joy_cmd_heading = 0.0
        self.joy_cmd_ang_vel = 0.0
        self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback, queue_size=1)

        self.env_cfg, self.train_cfg = task_registry.get_cfgs(name=args.task)
        # override some parameters for testing
        self.env_cfg.env.num_envs = min(self.env_cfg.env.num_envs, 1)
        self.env_cfg.terrain.num_rows = 3
        self.env_cfg.terrain.num_cols = 1
        # self.env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # self.env_cfg.terrain.terrain_proportions =  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


        self.env_cfg.terrain.curriculum = False
        self.env_cfg.noise.add_noise = False
        self.env_cfg.domain_rand.randomize_friction = False
        self.env_cfg.domain_rand.push_robots = False
        self.env_cfg.domain_rand.randomize_base_mass = False
        self.env_cfg.domain_rand.randomize_limb_mass = False

        # fixed velocity direction evaluation (make sure the value is within the training range)
        self.env_cfg.commands.heading_command = False
        self.env_cfg.commands.ranges.lin_vel_x = [-0.9, self.joy_cmd_velx]
        self.env_cfg.commands.ranges.lin_vel_y = [-0.9, self.joy_cmd_vely]
        self.env_cfg.commands.ranges.ang_vel_yaw = [self.joy_cmd_ang_vel, self.joy_cmd_ang_vel]
        self.env_cfg.commands.ranges.heading = [self.joy_cmd_heading, self.joy_cmd_heading]

        # init ros publisher
        self.obs_pub = rospy.Publisher('/legged_gym/observation', Float64MultiArray, queue_size=10)
        self.time_pub = rospy.Publisher('/legged_gym/time', Float64, queue_size=10)

        # self.lin_vel_x_pub = rospy.Publisher('/legged_gym/lin_vel_x', Float64, queue_size=10)
        # self.lin_vel_y_pub = rospy.Publisher('/legged_gym/lin_vel_y', Float64, queue_size=10)
        # self.lin_vel_z_pub = rospy.Publisher('/legged_gym/lin_vel_z', Float64, queue_size=10)

        self.ang_vel_x_pub = rospy.Publisher('/legged_gym/ang_vel_x', Float64, queue_size=10)
        self.ang_vel_y_pub = rospy.Publisher('/legged_gym/ang_vel_y', Float64, queue_size=10)
        self.ang_vel_z_pub = rospy.Publisher('/legged_gym/ang_vel_z', Float64, queue_size=10)

        self.gravity_x_pub = rospy.Publisher('/legged_gym/gravity_x', Float64, queue_size=10)
        self.gravity_y_pub = rospy.Publisher('/legged_gym/gravity_y', Float64, queue_size=10)
        self.gravity_z_pub = rospy.Publisher('/legged_gym/gravity_z', Float64, queue_size=10)

        self.cmd_vel_x_pub = rospy.Publisher('/legged_gym/cmd_vel_x', Float64, queue_size=10)
        self.cmd_vel_y_pub = rospy.Publisher('/legged_gym/cmd_vel_y', Float64, queue_size=10)
        self.cmd_vel_z_pub = rospy.Publisher('/legged_gym/cmd_vel_z', Float64, queue_size=10)

        self.FL_hip_pos_pub = rospy.Publisher('/legged_gym/FL_hip_pos', Float64, queue_size=10)
        self.FL_thigh_pos_pub = rospy.Publisher('/legged_gym/FL_thigh_pos', Float64, queue_size=10)
        self.FL_calf_pos_pub = rospy.Publisher('/legged_gym/FL_calf_pos', Float64, queue_size=10)
        self.FR_hip_pos_pub = rospy.Publisher('/legged_gym/FR_hip_pos', Float64, queue_size=10)
        self.FR_thigh_pos_pub = rospy.Publisher('/legged_gym/FR_thigh_pos', Float64, queue_size=10)
        self.FR_calf_pos_pub = rospy.Publisher('/legged_gym/FR_calf_pos', Float64, queue_size=10)
        self.RL_hip_pos_pub = rospy.Publisher('/legged_gym/RL_hip_pos', Float64, queue_size=10)
        self.RL_thigh_pos_pub = rospy.Publisher('/legged_gym/RL_thigh_pos', Float64, queue_size=10)
        self.RL_calf_pos_pub = rospy.Publisher('/legged_gym/RL_calf_pos', Float64, queue_size=10)
        self.RR_hip_pos_pub = rospy.Publisher('/legged_gym/RR_hip_pos', Float64, queue_size=10)
        self.RR_thigh_pos_pub = rospy.Publisher('/legged_gym/RR_thigh_pos', Float64, queue_size=10)
        self.RR_calf_pos_pub = rospy.Publisher('/legged_gym/RR_calf_pos', Float64, queue_size=10)

        self.FL_hip_vel_pub = rospy.Publisher('/legged_gym/FL_hip_vel', Float64, queue_size=10)
        self.FL_thigh_vel_pub = rospy.Publisher('/legged_gym/FL_thigh_vel', Float64, queue_size=10)
        self.FL_calf_vel_pub = rospy.Publisher('/legged_gym/FL_calf_vel', Float64, queue_size=10)
        self.FR_hip_vel_pub = rospy.Publisher('/legged_gym/FR_hip_vel', Float64, queue_size=10)
        self.FR_thigh_vel_pub = rospy.Publisher('/legged_gym/FR_thigh_vel', Float64, queue_size=10)
        self.FR_calf_vel_pub = rospy.Publisher('/legged_gym/FR_calf_vel', Float64, queue_size=10)
        self.RL_hip_vel_pub = rospy.Publisher('/legged_gym/RL_hip_vel', Float64, queue_size=10)
        self.RL_thigh_vel_pub = rospy.Publisher('/legged_gym/RL_thigh_vel', Float64, queue_size=10)
        self.RL_calf_vel_pub = rospy.Publisher('/legged_gym/RL_calf_vel', Float64, queue_size=10)
        self.RR_hip_vel_pub = rospy.Publisher('/legged_gym/RR_hip_vel', Float64, queue_size=10)
        self.RR_thigh_vel_pub = rospy.Publisher('/legged_gym/RR_thigh_vel', Float64, queue_size=10)
        self.RR_calf_vel_pub = rospy.Publisher('/legged_gym/RR_calf_vel', Float64, queue_size=10)

        self.FL_hip_act_pub = rospy.Publisher('/legged_gym/FL_hip_act', Float64, queue_size=10)
        self.FL_thigh_act_pub = rospy.Publisher('/legged_gym/FL_thigh_act', Float64, queue_size=10)
        self.FL_calf_act_pub = rospy.Publisher('/legged_gym/FL_calf_act', Float64, queue_size=10)
        self.FR_hip_act_pub = rospy.Publisher('/legged_gym/FR_hip_act', Float64, queue_size=10)
        self.FR_thigh_act_pub = rospy.Publisher('/legged_gym/FR_thigh_act', Float64, queue_size=10)
        self.FR_calf_act_pub = rospy.Publisher('/legged_gym/FR_calf_act', Float64, queue_size=10)
        self.RL_hip_act_pub = rospy.Publisher('/legged_gym/RL_hip_act', Float64, queue_size=10)
        self.RL_thigh_act_pub = rospy.Publisher('/legged_gym/RL_thigh_act', Float64, queue_size=10)
        self.RL_calf_act_pub = rospy.Publisher('/legged_gym/RL_calf_act', Float64, queue_size=10)
        self.RR_hip_act_pub = rospy.Publisher('/legged_gym/RR_hip_act', Float64, queue_size=10)
        self.RR_thigh_act_pub = rospy.Publisher('/legged_gym/RR_thigh_act', Float64, queue_size=10)
        self.RR_calf_act_pub = rospy.Publisher('/legged_gym/RR_calf_act', Float64, queue_size=10)


    def play(self):
        # prepare environment
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg)
        obs_dict = env.reset()
        # load policy
        self.train_cfg.runner.resume = True  # set the mode to be evalution
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=self.args.task, args=self.args, train_cfg=self.train_cfg)
        policy, pre_extrin_en = ppo_runner.get_inference_policy(device=env.device)

        while not rospy.is_shutdown():
            actions = policy(obs_dict)
            obs_dict, rews, dones, infos = env.step(actions.detach())

            # cmd's linear velocity changing module
            env._change_cmds(self.joy_cmd_velx, self.joy_cmd_vely, self.joy_cmd_ang_vel)
            obs = obs_dict['obs']
            real_morph_par = obs_dict['privileged_info'][:, 200:209]
            real_foot_height = obs_dict['privileged_info'][:, 224:]
            # cprint(f"est_mean: {est_mean}", 'red', attrs=['bold'])
            # cprint(f"real_foot_height: {real_foot_height}", 'green', attrs=['bold'])
            # cprint(f"est_var: {est_var}", 'blue', attrs=['bold'])

            # ************ for debug ************
            obs_float = obs.cpu().numpy().squeeze().astype(np.float32)
            # obs_msg = Float64MultiArray(data=obs_float)
            # self.obs_pub.publish(obs_msg)
            self.observatioon_pub(obs_float)

            time_msg = Float64(data=rospy.get_time())
            self.time_pub.publish(time_msg)

            rospy.sleep(0.01)

        # for i in range(10 * int(env.max_episode_length)):
        #     actions = policy(obs.detach())
        #     obs, _, rews, dones, infos = env.step(actions.detach())
        #
        #     # cmd's linear velocity changing module
        #     env._change_cmds(self.joy_cmd_velx, self.joy_cmd_vely, self.joy_cmd_ang_vel)

    def observatioon_pub(self, obs_float):


        self.ang_vel_x_pub.publish(obs_float[0])
        self.ang_vel_y_pub.publish(obs_float[1])
        self.ang_vel_z_pub.publish(obs_float[2])

        self.gravity_x_pub.publish(obs_float[3])
        self.gravity_y_pub.publish(obs_float[4])
        self.gravity_z_pub.publish(obs_float[5])

        self.cmd_vel_x_pub.publish(obs_float[6])
        self.cmd_vel_y_pub.publish(obs_float[7])
        self.cmd_vel_z_pub.publish(obs_float[8])

        self.FL_hip_pos_pub.publish(obs_float[9])
        self.FL_thigh_pos_pub.publish(obs_float[10])
        self.FL_calf_pos_pub.publish(obs_float[11])
        self.FR_hip_pos_pub.publish(obs_float[12])
        self.FR_thigh_pos_pub.publish(obs_float[13])
        self.FR_calf_pos_pub.publish(obs_float[14])
        self.RL_hip_pos_pub.publish(obs_float[15])
        self.RL_thigh_pos_pub.publish(obs_float[16])
        self.RL_calf_pos_pub.publish(obs_float[17])
        self.RR_hip_pos_pub.publish(obs_float[18])
        self.RR_thigh_pos_pub.publish(obs_float[19])
        self.RR_calf_pos_pub.publish(obs_float[20])

        # self.FL_hip_vel_pub.publish(obs_float[21])
        # self.FL_thigh_vel_pub.publish(obs_float[22])
        # self.FL_calf_vel_pub.publish(obs_float[23])
        # self.FR_hip_vel_pub.publish(obs_float[24])
        # self.FR_thigh_vel_pub.publish(obs_float[25])
        # self.FR_calf_vel_pub.publish(obs_float[26])
        # self.RL_hip_vel_pub.publish(obs_float[27])
        # self.RL_thigh_vel_pub.publish(obs_float[28])
        # self.RL_calf_vel_pub.publish(obs_float[29])
        # self.RR_hip_vel_pub.publish(obs_float[30])
        # self.RR_thigh_vel_pub.publish(obs_float[31])
        # self.RR_calf_vel_pub.publish(obs_float[32])

        # self.FL_hip_act_pub.publish(obs_float[33])
        # self.FL_thigh_act_pub.publish(obs_float[34])
        # self.FL_calf_act_pub.publish(obs_float[35])
        # self.FR_hip_act_pub.publish(obs_float[36])
        # self.FR_thigh_act_pub.publish(obs_float[37])
        # self.FR_calf_act_pub.publish(obs_float[38])
        # self.RL_hip_act_pub.publish(obs_float[39])
        # self.RL_thigh_act_pub.publish(obs_float[40])
        # self.RL_calf_act_pub.publish(obs_float[41])
        # self.RR_hip_act_pub.publish(obs_float[42])
        # self.RR_thigh_act_pub.publish(obs_float[43])
        # self.RR_calf_act_pub.publish(obs_float[44])
        #
        # self.lin_vel_x_pub.publish(obs_float[45])
        # self.lin_vel_y_pub.publish(obs_float[46])
        # self.lin_vel_z_pub.publish(obs_float[47])

    def joy_callback(self, joy_msg):
        self.joy_cmd_velx = joy_msg.axes[4] * 1.0
        self.joy_cmd_vely = joy_msg.axes[3] * 1.0
        # self.joy_cmd_heading = joy_msg.axes[0] * 1.5
        self.joy_cmd_ang_vel = joy_msg.axes[0] * 1.0


    def run(self):
        rospy.spin()



if __name__ == '__main__':
    rospy.init_node('play_policy_with_joy')

    args = get_args()
    play_joy = PlayJoy(args)
    play_joy.play()
    # play_joy.run()