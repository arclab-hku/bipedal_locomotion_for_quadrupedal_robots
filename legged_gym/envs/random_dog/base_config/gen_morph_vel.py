from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class RandomBaseCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):  # comment if use visual
        # num_envs = 4096
        num_envs = 4096  # was getting a seg fault
        # num_envs = 100  # was getting a seg fault
        num_actions = 12
        num_observations = 45
        # num_proprio_obs = 48
        camera_res = [1280, 720]
        camera_type = "d"  # rgb
        num_privileged_obs = 224  # 187
        train_type = "GenMorphVel"  # standard, priv, lbc, standard, RMA, EST, Dream

        follow_cam = False
        float_cam = False

        measure_obs_heights = False
        num_env_priv_obs = 17  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_env_morph_priv_obs = 24 # ! everything related to morphology information
        num_histroy_obs = 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # trimesh
        curriculum = False
        selected = False # False # select a unique terrain type and pass all arguments
        terrain_kwargs = {'type': "terrain_utils.pyramid_sloped_terrain",
                          'slope': 0.5,
                          'platform_size': 3,
                          } # None # Dict of arguments for selected terrain
        terrain_length = 8.
        terrain_width = 8.


        difficulty = 0.5
        step_height = 0.05 + 0.18 * difficulty
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        discrete_obstacles_height = 0.05 + difficulty * 0.2

        terrain_kwargs1 = {'type': "terrain_utils.pyramid_stairs_terrain",
                           'step_width': 0.31,
                           'step_height': step_height,
                           'platform_size': 3,
                           }  # None # Dict of arguments for selected terrain

        # Dict of arguments for selected terrain
        terrain_kwargs4 = {'type': "terrain_utils.random_uniform_terrain",
                          'min_height': -0.04,
                          'max_height': 0.04,
                          'step': 0.01}  # None # Dict of arguments for selected terrain

        terrain_kwargs3 = {'type':"terrain_utils.wave_terrain",
                          'num_waves': 2,
                          'amplitude': 0.5} # None # Dict of arguments for selected terrain

        terrain_kwargs2 = {'type':"terrain_utils.discrete_obstacles_terrain",
                           'max_height': discrete_obstacles_height,
                           'min_size': 0.8,
                           'max_size': 1.3,
                           'num_rects': 20,
                           'platform_size': 3,
                           }  # None # Dict of arguments for selected terrain

    class init_state(LeggedRobotCfg.init_state):
        pos = [-0, -0, 0.75]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": -0.05,  # [rad]
            "RL_hip_joint": -0.05,  # [rad]
            "FR_hip_joint": 0.05,  # [rad]
            "RR_hip_joint": 0.05,  # [rad]

            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 1.0,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 1.0,  # [rad]

            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
        }

        default_joint_angles1 = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.0,  # [rad]
            "RL_hip_joint": 0.0,  # [rad]
            "FR_hip_joint": -0.0,  # [rad]
            "RR_hip_joint": -0.0,  # [rad]

            "FL_thigh_joint": 0.4,  # [rad]
            "RL_thigh_joint": -0.4,  # [rad]
            "FR_thigh_joint": 0.4,  # [rad]
            "RR_thigh_joint": -0.4,  # [rad]

            "FL_calf_joint": -0.8,  # [rad]
            "RL_calf_joint": 0.8,  # [rad]
            "FR_calf_joint": -0.8,  # [rad]
            "RR_calf_joint": 0.8,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "POSE"
        # set different stiffness for "small", "medium", "lagre" size robot

        # medium: lite3, mini_cheetah
        stiffness_mini = {'joint': 20.0}  # [N*m/rad]
        damping_mini = {'joint': 0.6}  # [N*m*s/rad]

        # medium: go1, a1,
        stiffness_s = {'joint': 30.0}  # [N*m/rad]
        damping_s = {'joint': 0.8}  # [N*m*s/rad]

        # medium: aliengo, laikago
        stiffness_m = {'joint': 40.0}  # [N*m/rad]
        damping_m = {'joint': 1.2}  # [N*m*s/rad]

        # large medium: spot, laikago
        stiffness_lm = {'joint': 60.0}  # [N*m/rad]
        damping_lm = {'joint': 1.8}  # [N*m*s/rad]

        # large: b1, anymal_c, anymal_b
        stiffness_l = {'joint': 80.0}  # [N*m/rad]
        damping_l = {'joint': 2.0}  # [N*m*s/rad]

        stiffness = {"joint": 30.0}  # [N*m/rad]
        damping = {"joint": 0.8}  # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_go1_join_brick_stairs_it550.pt"

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/all_dog'
        name = "random_dog"
        asset_name = ["go1"]
        # asset_name = ["a1", "go1", "aliengo", "b1", "laikago", "lite3", "arcdog1", "mini_cheetah_new", "spot", "anymal_b", "anymal_c"]

        # asset_name = ["a1", "go1", "aliengo", "b1", "laikago", "lite3", "mini_cheetah_new", "spot", "anymal_b", "anymal_c"]


        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        # terminate_after_contacts_on = ["base"]
        # terminate_after_contacts_on = ["base", "trunk", "hip"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        # flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = False
        added_mass_range = [-2.0, 2.0]

        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_center = True
        center_range = [-0.05, 0.05]

        randomize_motor_strength = True
        added_motor_strength = [0.9, 1.1]

        randomize_lag_timesteps = True   # actuator net: True
        added_lag_timesteps = 6

        randomize_Motor_Offset = True  # actuator net: True
        added_Motor_OffsetRange = [-0.02, 0.02]


    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.5
        max_contact_force = 500.0
        only_positive_rewards = True
        foot_height_target = 0.09

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            torques = -0.00001
            dof_acc = -2.5e-7
            base_height = -0.0
            feet_air_time = 1.0
            collision = -0.0
            action_rate = -0.01

            # #### motion
            f_hip_motion = -0.02
            r_hip_motion = -0.02
            f_thigh_motion = -0.02
            r_thigh_motion = -0.02
            f_calf_motion = -0.02
            r_calf_motion = -0.02

            #### smoothness
            # dream_smoothness = -0.001
            # power_joint = -1e-4
            # foot_clearance = -0.001
            # foot_height = -0.01

    class viewer:
        ref_env = 0
        pos = [10, -12, 5]  # [m]
        lookat = [11., 5, 3.]  # [m]
    class evals(LeggedRobotCfg.evals):
        feet_stumble = True
        feet_step = True
        crash_freq = True
        any_contacts = True

    class privInfo(LeggedRobotCfg.privInfo):

        enableMeasuredVel = True
        enableMeasuredHeight = True
        enableForce = True
        enableFrictionCenter = True

        # constant value
        enableBodysMass = True
        enableMotorStrength = True
        enableBodysSize = True

class RandomBaseCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        max_iterations = 3000  # number of policy updates
        resume = False
        save_interval = 250  # check for potential saves every this many iterations
        experiment_name = 'Random'
        export_policy = False

    class Encoder(LeggedRobotCfgPPO.Encoder):
        encoder_mlp_units = [258, 128, 12]
        priv_info_dim = 224
        proprio_adapt = False
        checkpoint_model = None
        proprio_adapt_out_dim = 11
        morph_priv_info = True

        HistoryLen = 4
        Hist_info_dim = 45 * HistoryLen