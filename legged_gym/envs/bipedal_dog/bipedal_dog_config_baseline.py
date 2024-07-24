from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BipedalBaseCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):  # comment if use visual
        num_envs = 4096  # was getting a seg fault
        num_actions = 12
        num_observations = 42
        # num_observations = 45 + 36
        num_privileged_obs = 225  # 187
        train_type = "Imi"  # standard, priv, lbc, standard, RMA, EST, Dream, Aumoral, Imi
        episode_length_s = 22  # episode length in seconds
        num_env_morph_priv_obs = 19  # ! everything related to morphology information
        num_histroy_obs = 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'stair'  # plane, stair, trimesh, stone

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        heading_command = True
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.34]  # x,y,z [m]
        # pos = [0.0, 0.0, 0.55]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.0,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.0,  # [rad]

        # 'FL_thigh_joint': 1.0,  # [rad]
        # 'RL_thigh_joint': 0.8,  # [rad]
        # 'FR_thigh_joint': 1.0,  # [rad]
        # 'RR_thigh_joint': 0.8,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5  # [rad]

        # "FL_hip_joint": 0.0,  # [rad]
        # "RL_hip_joint": 0.0,  # [rad]
        # "FR_hip_joint": -0.0,  # [rad]
        # "RR_hip_joint": -0.0,  # [rad]

        # "FL_thigh_joint": 0.4,  # [rad]
        # "RL_thigh_joint": -0.4,  # [rad]
        # "FR_thigh_joint": 0.4,  # [rad]
        # "RR_thigh_joint": -0.4,  # [rad]

        # "FL_calf_joint": -0.8,  # [rad]
        # "RL_calf_joint": 0.8,  # [rad]
        # "FR_calf_joint": -0.8,  # [rad]
        # "RR_calf_joint": 0.8,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "POSE"
        # stiffness = {"joint": 100.0}  # [N*m/rad]
        # damping = {"joint": 3.}  # [N*m*s/rad]
        stiffness = {"joint": 30.0}  # [N*m/rad]
        damping = {"joint": 0.8}  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4
        use_actuator_network = True
        # actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_go1_join_brick_stairs_it550.pt"
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_go1_2rd_f100_it2000_mlp_drop0.04_quad.pt"

        # actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_go1_bipedal_new_acnet_2.pt"
        # actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_go1_bipedal_upsidedown.pt"

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/all_dog'
        name = "random_dog"
        asset_name = ["go1"]
        # asset_name = ["a1", "go1", "aliengo", "arcdog", "b1", "laikago", "lite3",  "solo", "mini_cheetah_new", "spot", "anymal_b", "anymal_c"]
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "trunk", "hip"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        # flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-2.0, 2.0]

        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_center = True
        center_range = [-0.05, 0.05]

        randomize_motor_strength = True
        added_motor_strength = [0.9, 1.1]

        randomize_lag_timesteps = True  # actuator net: True
        added_lag_timesteps = 4

        randomize_Motor_Offset = True  # actuator net: True
        added_Motor_OffsetRange = [-0.02, 0.02]

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.55
        # base_height_target = 0.34
        max_contact_force = 500.0
        only_positive_rewards = True
        foot_height_target = 0.08

        class scales(LeggedRobotCfg.rewards.scales):
            """
            #### base
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            lin_vel_z = 0.
            ang_vel_xy = 0.
            orientation = -0.2
            torques = -0.00001
            dof_acc = -2.5e-7
            base_height = -0.1
            feet_air_time = 1.0
            collision = -0.1
            action_rate = -0.02

            # #### motion
            f_hip_motion = -0.0
            r_hip_motion = -0.0
            f_thigh_motion = -0.
            r_thigh_motion = -0.
            f_calf_motion = -0.
            r_calf_motion = -0.
      
            #### imitation
            tracking_ref_joint_pos = 0.2
            tracking_ref_joint_vel = 1.
            tracking_ref_body_pos = 0.1
            tracking_ref_body_quat = 0.1
            tracking_ref_body_vel = 1.0
            tracking_ref_both_pos = 0.1
            tracking_ref_both_vel = 0.1
            tracking_ref_feet_pos = 0.1
            # tracking_ref_shaped_force = 0.1
            # tracking_ref_shaped_vel = 0.1
            # tracking_com_pos = 0.05
            # tracking_body_roll_vel = 0.01
            # tracking_body_pitch_vel = 0.01
            # tracking_body_yaw_vel = 0.01
            # tracking_body_x_vel = 0.01
            # tracking_body_y_vel = 0.01
            # tracking_body_z_vel = 0.01

            # for pace ###
            # flrl_gait_diff = -0.5
            # frrr_gait_diff = -0.5
            # flrl_gait_diff_2 = -0.005
            # frrr_gait_diff_2 = -0.005

            # tracking_contacts_shaped_force = -1.
            # tracking_contacts_shaped_vel = -1.

            # for bound ###
            # flfr_gait_diff = -0.1
            # rlrr_gait_diff = -0.1
            # flfr_gait_diff_2 = -0.01
            # rlrr_gait_diff_2 = -0.01

            # #### smoothness
            dream_smoothness = -0.001
            # power_joint = -1e-4
            # foot_clearance = -0.01
            # foot_height = -0.01

            """
            #### base
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = 0.8
            orientation_3 = -0.03
            torques = -0.
            dof_acc = -0.
            # orientation = -0.2
            # torques = -0.00001
            # dof_acc = -2.5e-7
            # base_height = -0.1
            base_height = -0.5
            feet_air_time = 0.
            collision = -1.0
            action_rate = -0.02

            #### motion
            f_hip_motion = -0.15
            r_hip_motion = -0.15
            f_thigh_motion = -0.05
            r_thigh_motion = -0.05
            f_calf_motion = -0.05
            r_calf_motion = -0.05

            # inv_pendulum = -0.1
            # inv_pendulum_acc = -0.0001
            # inv_pendulum_both = -0.0001
            # cart_table_len_xy= -0.1

            fr_contact_force = -0.03
            # fl_contact_force = -0.03
            # f_contact_force = -0.03
            ri_contact_force = -0.03
            rr_contact_force = -0.03
            # rl_contact_force = -0.03
            # r_contact_force = -0.03

            # l_contact_force = -0.03

    class privInfo(LeggedRobotCfg.privInfo):
        enableMeasuredVel = True
        enableMeasuredHeight = True
        enableForce = True
        enableFrictionCenter = True
        # constant value
        enableBodysMass = True
        enableMotorStrength = True


class BipedalBaseCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        max_iterations = 6000  # number of policy updates
        save_interval = 2000  # check for potential saves every this many iterations
        experiment_name = 'random_dog'

    class Encoder(LeggedRobotCfgPPO.Encoder):
        encoder_mlp_units = [256, 128, 9]
        decoder_mlp_units = [256, 128, 3]
        priv_info_dim = 225
        HistoryLen = 4
