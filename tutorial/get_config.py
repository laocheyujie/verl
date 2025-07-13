import copy

import hydra
from omegaconf import DictConfig

from verl.utils.config import omega_conf_to_dataclass


def trainer_dict_to_dataclass(conf: DictConfig):
    """Convert specific nested sections of a DictConfig object into dataclass instances.

    Args:
        conf (DictConfig): An instance of DictConfig, typically from the omegaconf library,
                           representing a configuration dictionary.

    Returns:
        DictConfig: A deep copy of the input `conf` with specific sections converted to dataclasses.
    """
    # Create a deep copy of the input configuration to avoid modifying the original object
    config = copy.deepcopy(conf)
    config.algorithm = omega_conf_to_dataclass(config.algorithm)
    config.critic.profiler = omega_conf_to_dataclass(config.critic.profiler)
    config.reward_model.profiler = omega_conf_to_dataclass(config.reward_model.profiler)
    config.actor_rollout_ref.actor.profiler = omega_conf_to_dataclass(config.actor_rollout_ref.actor.profiler)
    config.actor_rollout_ref.ref.profiler = omega_conf_to_dataclass(config.actor_rollout_ref.ref.profiler)
    config.actor_rollout_ref.rollout.profiler = omega_conf_to_dataclass(config.actor_rollout_ref.rollout.profiler)
    return config


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config_dict):
    config = trainer_dict_to_dataclass(config_dict)
    return config


if __name__ == "__main__":
    conf = main()
    print(conf)
