import argparse
import json
import os
from ... import paths



def main(args):

    with open(paths.TASKS_FRUITS_COMB, "r") as f:
        task_list = json.load(f)

    folder_path = "data/fruits_DQN_models"

    if not os.path.isdir(folder_path):

        os.makedirs(folder_path)

    for idx, c in enumerate(task_list):

        model_name = "model_{:s}.pt".format(str(c))
        model_path = os.path.join(folder_path, model_name)

        print("{:s} goal".format(str(c)))

        subprocess.call([
            "python", "-m", "ap.scr.online.fruits.run_DQN", "--name", "fruits_DQN_models",
            "with", "device={:s}".format(args.device), "goal={:s}".format(str(c)),
            "max_steps=50000", "exploration_steps=1", "prioritized_replay=False", "prioritized_replay_max_steps=0",
            "save_model_path={:s}".format(model_path), "num_expert_steps=50000", "num_random_steps=0",
            "num_pretraining_steps=50000"
        ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device")

    parsed = parser.parse_args()
    main(parsed)
