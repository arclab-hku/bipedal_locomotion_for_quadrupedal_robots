
from .base.legged_robot import LeggedRobot




from .random_dog.random_dog import Randomdog
from .bipedal_dog.bipedal_dog import Bipedaldog
from .random_dog.random_dog_config_baseline import RandomBaseCfg, RandomBaseCfgPPO
from .bipedal_dog.bipedal_dog_config_baseline import BipedalBaseCfg, BipedalBaseCfgPPO

from legged_gym.utils.task_registry import task_registry


task_registry.register( "random_dog", Randomdog, RandomBaseCfg(), RandomBaseCfgPPO())
task_registry.register( "bipedal_dog", Bipedaldog, BipedalBaseCfg(), BipedalBaseCfgPPO())