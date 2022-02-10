import numpy as np
import tensorflow as tf
import ray
from ray import tune

from helpers import generate_path
from custom_gym_test import CustomEnv

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Base settings
    initial_lr = 5e-5
    final_lr = 1e-5
    num_iter = 400  # maximum number of iterations per trial (1 iter == 4000 time steps)

    _max_ts = num_iter*4000

    ray.init(address=None, ignore_reinit_error=False)

    sync_config = tune.SyncConfig()
    # bayes_opt = BayesOptSearch(metric="episode_reward_mean", mode="max")

    config = {
        "env": CustomEnv,
        "env_config": {"red_path_type": "random_circle",  # "random_circle" | "centered_circle" | "random_walk"
                       "reward_impl": tune.grid_search([4])},
        "num_gpus": 1,
        "framework": "tf",
        "eager_tracing": False,
        "model": {
            "use_attention": False,  # bad performance with True
            "use_lstm": False,  # bad performance with True
            "vf_share_layers": False,  # TODO: Use true for image processing to improve performance
            "fcnet_activation": "relu",
        },
        "lr": initial_lr,
        "lr_schedule": [(0, initial_lr), (_max_ts, final_lr)],
    }
    results = tune.run(
        'PPO',
        name="final_results_continued",
        local_dir="/checkpoints",
        sync_config=sync_config,
        stop={
            'timesteps_total': _max_ts  # 20 eps per iter == 4000 steps per iter (1 ep == 200 steps)
        },
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=5,
        config=config,
        search_alg=None,
        resume="AUTO",
    )

# Priority 1
# DONE: solve problem with velocity discount
# DONE: Change reward dependence on velocity?

# Priority 2
# TODO: Plot the overall reward function space?
# TODO: implement bayes_opt search without problems with 64/32-bit precision

