from matplotlib import pyplot as plt
from ray.rllib.agents.ppo import PPOTrainer
from custom_gym_test import CustomEnv

# TODO: reward agent for not moving too much!

if __name__ == '__main__':
    # Base settings
    initial_lr = 5e-5
    final_lr = 1e-5
    num_iter = 400  # 1 iter == 20 eps
    reward_impls = [4, 3, 2, 1, 0]
    _max_ts = num_iter*4000

    wall_handle = plt.subplots()
    for ri in reward_impls:
        print(ri)
        config = {
            "env": CustomEnv,
            "env_config": {"red_path_type": "random_circle",  # "random_circle" | "centered_circle" | "random_walk"
                           "reward_impl": ri,
                           "render": False},
            "num_gpus": 1,
            "model": {
                "vf_share_layers": False,
                "fcnet_activation": "relu",
            },
            "lr": initial_lr,
            "lr_schedule": [(0, initial_lr), (_max_ts, final_lr)],
            "evaluation_interval": 100,
            "evaluation_num_episodes": 2,
            "evaluation_num_workers": 1,
            "evaluation_config": {
                "env_config": {"red_path_type": "random_circle",
                               "render": True},
                "render_env": True,
            }
        }

        trainer = PPOTrainer(config=config)

        for it in range(num_iter):
            print(f"{it:03d}: {trainer.train()}")
            if (it+1) % 20 == 0:
                trainer.save(f"./checkpoints/final/{ri:02d}")

    # trainer.evaluate()