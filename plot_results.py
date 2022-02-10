import os
import json
import pandas as pd
from matplotlib import pyplot as plt

CHECKPOINT_ROOT = "D:/checkpoints/final_results"

if __name__ == '__main__':
    _, folders, _ = next(os.walk(CHECKPOINT_ROOT))

    full_csv_paths = [f"{CHECKPOINT_ROOT}/{f}/progress.csv" for f in folders]

    fig, ax = plt.subplots()

    ax.set_xlabel("epizoda")
    ax.set_ylabel(r"odměna $r$")
    ax.set_title("Vývoj odměn agentů 2b)")

    for i, pth in enumerate(full_csv_paths):
        with open(pth, "rb") as f:
            res = pd.read_csv(f)
            ax.plot(res["episode_reward_mean"])

    plt.legend([r"$r_0 ... 1-d$", "$r_1 ... $sigmoid", "$r_2 ... 1-d^{0.4}$", "$r_3 ... r_2 - p$", "$r_4 ... \mu r_2-p$"])
    plt.savefig(f"{CHECKPOINT_ROOT}/collective_reward_plot.png", dpi=200)
    plt.show()
