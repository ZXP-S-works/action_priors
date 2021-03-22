
from action_priors.ap.run.online.fruits.RunDQN import RunDQN
from action_priors.ap.constants import Constants
from action_priors.ap import constants
from action_priors.ap.utils.logger import Logger
from action_priors.ap import paths
import os

# ex = Experiment("fruits_DQN")
# if constants.MONGO_URI is not None and constants.DB_NAME is not None:
#     ex.observers.append(MongoObserver(url=constants.MONGO_URI, db_name=constants.DB_NAME))
# else:
#     print("WARNING: results are not being saved. See 'Setup MongoDB' in README.")
# ex.add_config(paths.CFG_ONLINE_FRUITS_DQN)


def main(qv_learning=True, dueling=True, double_learning=True, prioritized_replay=True, prioritized_replay_max_steps=100000, discount=0.9,
         goal=[0, 1], learning_rate=0.0005, weight_decay=0.00001, batch_size=32, max_steps=100000, max_episodes=None,
         exploration_steps=80000, buffer_size=100000, target_network=True, target_network_sync=5000, num_fruits=5,
         policy='EPS', init_tau=1.0, final_tau=0.0, side_transfer=False, side_transfer_last=False,
         side_encoder_load_path=None, freeze_encoder=False, num_expert_steps=0, num_random_steps=0,
         num_pretraining_steps=0, load_model_path=None, save_model_path=None, demonstrate_dqn=False, device='cuda:0',
         encoder_load_path=None):

    folder_path = "data/fruits_DQN_models"

    print("{:s} goal".format(str(goal)))

    if not os.path.isdir(folder_path):

        os.makedirs(folder_path)

    model_config = {
        Constants.QV_LEARNING: qv_learning,
        Constants.NUM_ACTIONS: 25,
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
        Constants.NUM_FRUITS: num_fruits,
        Constants.POLICY: Constants(policy),
        Constants.SIDE_TRANSFER: side_transfer,
        Constants.FREEZE_ENCODER: freeze_encoder,
        Constants.SIDE_TRANSFER_LAST: side_transfer_last
    }

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
