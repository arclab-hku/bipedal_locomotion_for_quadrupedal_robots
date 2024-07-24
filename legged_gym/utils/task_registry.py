
import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rl.RMA.env import VecEnv

from rl.Base.runners import BasePolicyRunner
from rl.RMA.runners import PPOPolicyRunner
from rl.RMA.runners import ProprioAdaptPolicyRunner
from rl.EST.runners import ESTPolicyRunner
from rl.EST_rough.runners import ESTRoughPolicyRunner

from rl.Dream.runners import DreamPolicyRunner

from rl.Gen.runners import GenPolicyRunner
from rl.Gen_base.runners import GenBasePolicyRunner
from rl.Gen_his.runners import GenHisPolicyRunner
from rl.Gen_his_est.runners import GenHisESTPolicyRunner
from rl.Gen_morph.runners import GenMorphPolicyRunner
from rl.Gen_morph_vel.runners import GenMorphVelPolicyRunner
from rl.Imi.runners import ImiPolicyRunner
from rl.Imi.runners import ImiDecoderPolicyRunner


from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

from shutil import copyfile


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(cfg=env_cfg,
                         sim_params=sim_params,
                         physics_engine=args.physics_engine,
                         sim_device=args.sim_device,
                         headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default"):
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root == "default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'outputs')
            log_dir = os.path.join(log_root, args.output_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, args.output_name)



        if not train_cfg.runner.resume:
            # check whether execute train by mistake:
            last_ckpt_path = os.path.join(
                log_dir,
                'stage1_nn' if args.algo == 'PPO' or "Base" or 'EST' or 'Dream' or 'GenHisEST' or 'GenHis'
                               or 'Gen' or 'GenBase' or 'GenMorph' or "GenMorphVel" or "Imi" else 'stage2_nn', 'last.pth'
            )
            if os.path.exists(last_ckpt_path):
                user_input = input(
                    f'are you intentionally going to overwrite files in {args.output_name}, type yes to continue \n')
                if user_input != 'yes':
                    exit()

            os.makedirs(log_dir, exist_ok=True)

            save_item = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs', name,
                                     name + '_config_baseline.py')
            copyfile(save_item, log_dir + '/train_cfg_robot.py')

            save_item1 = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs', name,
                                      name + '.py')
            copyfile(save_item1, log_dir +'/robot.py')

            save_item2 = os.path.join(LEGGED_GYM_ROOT_DIR, 'rl', args.algo, 'modules',
                                      'actor_critic' + '.py')
            copyfile(save_item2, log_dir + '/actor_critic.py')

            save_item3 = os.path.join(LEGGED_GYM_ROOT_DIR, 'rl', args.algo, 'algorithms',
                                      'ppo' + '.py')
            copyfile(save_item3, log_dir + '/ppo.py')

        train_cfg_dict = class_to_dict(train_cfg)

        resume = train_cfg.runner.resume

        log_dir1 = log_dir
        if args.resume_name is not None:
            log_dir1 = os.path.join(log_root, args.resume_name)

            os.makedirs(log_dir1, exist_ok=True)

            # save_item = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs', name,
            #                          name + '_config_baseline.py')
            # copyfile(save_item, log_dir1 + '/train_cfg_robot.py')
            #
            # save_item1 = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs', name,
            #                           name + '.py')
            # copyfile(save_item1, log_dir1 + '/robot.py')

        if resume:
            runner = eval(args.algo + "PolicyRunner")(env, train_cfg_dict, log_dir1, device=args.rl_device)
        else:
            runner = eval(args.algo + "PolicyRunner")(env, train_cfg_dict, log_dir, device=args.rl_device)

        print("**************** RUNNER ", runner)

        # save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            model_dir = None
            if args.s_flag == "1":
                model_dir = "stage1_nn"
            elif args.s_flag == "2":
                model_dir = "stage2_nn"
            elif args.s_flag == "3":
                model_dir = args.resume_name

            resume_path = get_load_path(os.path.join(log_dir, str(model_dir)), checkpoint=args.checkpoint_model)

            print(f"Loading model from: {log_dir, model_dir, resume_path}")
            runner.load(resume_path)
        return runner, train_cfg


# make global task registry
task_registry = TaskRegistry()