from matplotlib import pyplot as plt
from ray.rllib.agents.ppo import PPOTrainer
from custom_gym_test import CustomEnv

# TODO: reward agent for not moving too much!

if __name__ == '__main__':
    num_iter = 200  # 1 iter == 20 eps
    wall_handle = plt.subplots()
    config = {
        "env": CustomEnv,
        "env_config": {"red_path_type": "random_circle",  # "random_circle" | "centered_circle" | "random_walk"
                       "reward_impl": 3,
                       "render": False},
        "num_gpus": 1,
        "model": {
            "vf_share_layers": False,
            "fcnet_activation": "relu",
        },
        "lr": 1e-5,
        "evaluation_interval": 10,
        "evaluation_num_episodes": 50,
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "env_config": {"red_path_type": "random_circle",
                           "render": True},
            "render_env": True,
        }
    }

    trainer = PPOTrainer(config=config)

    mean_rewards = {}
    for i in range(5):
        rev_id = f"{i:02d}"
        trainer.restore(f"D:/!private/Lord/Git/laser-tracker-with-ray/checkpoints/final/{rev_id}/checkpoint_000400/checkpoint-400")
        res = trainer.evaluate()
        mean_rewards[rev_id] = res["evaluation"]["episode_reward_mean"]

    print(mean_rewards)
    # trainer.restore("D:/!private/Lord/Git/laser-tracker-with-ray/checkpoints/final/04/checkpoint_000400/checkpoint-400")

    # trainer.evaluate()
