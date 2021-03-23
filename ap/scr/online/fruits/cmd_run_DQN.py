
from action_priors.ap.run.online.fruits.RunDQN import RunDQN
from action_priors.ap.constants import Constants
from action_priors.ap import constants
from action_priors.ap.utils.logger import Logger
from action_priors.ap.hyperparameters import *
import numpy as np
import matplotlib.pyplot as plt
from action_priors.ap import paths
from datetime import datetime
import json
import os

# ex = Experiment("fruits_DQN")
# if constants.MONGO_URI is not None and constants.DB_NAME is not None:
#     ex.observers.append(MongoObserver(url=constants.MONGO_URI, db_name=constants.DB_NAME))
# else:
#     print("WARNING: results are not being saved. See 'Setup MongoDB' in README.")
# ex.add_config(paths.CFG_ONLINE_FRUITS_DQN)

def creat_path():
    save_path = '../../../results/'+'QV_'*qv_learning\
                +'DQN_'*(not qv_learning)\
                +'Dueling_'*dueling\
                +'Double_'*double_learning\
                +'Prioritized_replay_'*prioritized_replay\
                +str(goal)+'/'\
                +datetime.today().strftime('%m.%d.%H:%M:%S')+'/'
    if not os.path.exists(os.path.dirname(save_path)):
        try:
            os.makedirs(os.path.dirname(save_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    return save_path

def save_all(r_hist, t_hist, path):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    # with open(os.path.join(save_path, "parameters.json"), 'w') as f:
    #     json.dump(hyper_parameters, f, cls=NumpyEncoder)

    np.savez(os.path.join(path, 'learning_curve.npy'), rewards=r_hist, episodes=t_hist)
    plt.plot(t_hist, r_hist)
    # plt.title(str(alg))
    plt.xlabel('episodes')
    plt.ylabel('sucess rate')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'learning_curve.pdf'))
    plt.close()

def main():

    folder_path = creat_path()

    print("{:s} goal".format(str(goal)))

    # if not os.path.isdir(folder_path):
    #
    #     os.makedirs(folder_path)

    model_config = {
        Constants.QV_LEARNING: qv_learning,
        Constants.NUM_ACTIONS: num_actions_x * env_size ** 2,
        Constants.DUELING: dueling,
        Constants.PRIORITIZED_REPLAY: prioritized_replay,
        Constants.DISCOUNT: discount,
        Constants.EXPLORATION_STEPS: exploration_steps,
        Constants.PRIORITIZED_REPLAY_MAX_STEPS: prioritized_replay_max_steps,
        Constants.BUFFER_SIZE: buffer_size,
        Constants.INIT_TAU: init_tau,
        Constants.FINAL_TAU: final_tau
    }

    runner_config = {
        Constants.QV_LEARNING: qv_learning,
        Constants.DOUBLE_LEARNING: double_learning,
        Constants.LEARNING_RATE: learning_rate,
        Constants.WEIGHT_DECAY: weight_decay,
        Constants.BATCH_SIZE: batch_size,
        Constants.GOAL: goal,
        Constants.DEVICE: device,
        Constants.MAX_STEPS: max_steps,
        Constants.MAX_EPISODES: max_episodes,
        Constants.TARGET_NETWORK: target_network,
        Constants.TARGET_NETWORK_SYNC: target_network_sync,
        Constants.ENV_SIZE: env_size,
        Constants.NUM_FRUITS: num_fruits,
        Constants.POLICY: Constants(policy),
        Constants.SIDE_TRANSFER: side_transfer,
        Constants.FREEZE_ENCODER: freeze_encoder,
        Constants.SIDE_TRANSFER_LAST: side_transfer_last
    }

    dic_runner_config = {}
    for key in runner_config.keys():
        dic_runner_config[str(key)] = str(runner_config[key])
    dic_model_config = {}
    for key in model_config.keys():
        dic_model_config[str(key)] = str(model_config[key])
    json_data = {
        'runner_config': dic_runner_config,
        'model_config': dic_model_config
    }
    with open(os.path.join(folder_path, "parameters.json"), 'w') as f:
        json.dump(json_data, f, indent=2)

    logger = Logger(save_file=None, print_logs=True)

    runner = RunDQN(runner_config, model_config, logger)

    if num_expert_steps > 0 or num_random_steps > 0:
        logger.info("collecting {:d} expert and {:d} random transitions".format(num_expert_steps, num_random_steps))
        runner.generate_demonstrations(num_expert_steps, num_random_steps)

    if num_pretraining_steps > 0:
        assert num_expert_steps > 0 or num_random_steps > 0

        logger.info("pretraining DQN for {:d} steps".format(num_pretraining_steps))
        opt = runner.get_opt()
        for i in range(num_pretraining_steps):
            runner.learn_step_(opt, prioritized_replay_max_steps)
            if i % 1000 == 0:
                logger.info("pretraining step {:d}".format(i))
            if i % target_network_sync == 0:
                runner.target_dqn.sync_weights(runner.dqn)

    if encoder_load_path is not None:
        logger.info("loading encoder")
        runner.load_encoder(encoder_load_path)

    if side_encoder_load_path is not None:
        logger.info("loading side encoder")
        runner.load_side_encoder(side_encoder_load_path)

    if load_model_path is not None:
        runner.dqn.load(load_model_path)
    else:
        runner.train_model()

    if save_model_path is not None:
        runner.dqn.save(save_model_path)

    if demonstrate_dqn:
        runner.demonstrate_dqn()

    training_result = runner.training_result[Constants.TOTAL_REWARDS]
    save_all(training_result, np.arange(len(training_result)), folder_path)

    # sacred_utils.log_list("total_rewards", runner.training_result[Constants.TOTAL_REWARDS], ex)
    # sacred_utils.log_list("discounted_total_rewards", runner.training_result[Constants.DISCOUNTED_REWARDS], ex)
    #
    # sacred_utils.log_list("eval_total_rewards", runner.training_result[Constants.EVAL_TOTAL_REWARDS], ex)
    # sacred_utils.log_list(
    #     "eval_discounted_total_rewards", runner.training_result[Constants.EVAL_DISCOUNTED_TOTAL_REWARDS], ex
    # )
    # sacred_utils.log_list("eval_num_steps", runner.training_result[Constants.EVAL_NUM_STEPS], ex)

if __name__ == "__main__":

    main()
