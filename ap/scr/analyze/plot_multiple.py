import itertools
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from action_priors.ap.hyperparameters import *


def getRewardsSingle(rewards, window=1000):
    moving_avg = []
    i = window
    # while i - window < len(rewards):
    #     moving_avg.append(np.average(rewards[i - window:i]))
    #     i += window
    multi_window_len = (len(rewards) // window) * window
    rewards = np.array(rewards[:multi_window_len])
    moving_avg = rewards.reshape(-1, window).mean(1)

    return moving_avg


def plotLearningCurveAvg(rewards, window=1000, label='reward', color='b', shadow=True, ax=plt, legend=True,
                         linestyle='-'):
    lens = list(len(i) for i in rewards)
    min_len = min(lens)
    max_len = max(lens)
    if min_len != max_len:
        rewards = np.array(list(itertools.zip_longest(*rewards, fillvalue=0))).T
        rewards = rewards[:, :min_len]
    avg_rewards = np.mean(rewards, axis=0)
    # std_rewards = np.std(rewards, axis=0)
    std_rewards = stats.sem(rewards, axis=0)
    xs = np.arange(window, window * avg_rewards.shape[0] + 1, window)
    if shadow:
        ax.fill_between(xs, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, color=color)
    l = ax.plot(xs, avg_rewards, label=label, color=color, linestyle=linestyle, alpha=0.7)
    if legend:
        ax.legend(loc=4)
    return l


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def plotLearningCurve(base, ep=50000, use_default_cm=False, filer_pass_word='_', figname='plot.png'):
    plt.style.use('ggplot')
    plt.figure(dpi=300)
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    LEGEND_SIZE = 8

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    # colors = "bgrycmkwbgrycmkw"
    colors = ('blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan', 'gray')
    if use_default_cm:
        color_map = {}
    else:
        color_map = {
            'sdqfd': 'b',
            'adet': 'g',
            'dqfd': 'orange',
            'BC': 'purple',
            'dqn': 'r',
            'ddpg': 'r',

            'sdqfd_hier': 'b',
            'adet_hier': 'g',
            'dqfd_hier': 'orange',
            'BC_hier': 'purple',
            'dqn_hier': 'r',
            'dqn_hier_q1tq2o': 'b',

            'adet_hier_hc=1': 'g',
            'adet_hier_hc=10': 'limegreen',
            'dqfd_hier_hc=1': 'orange',
            'dqfd_hier_hc=10': 'y',
            'sdqfd_hier_hc=1': 'b',
            'sdqfd_hier_hc=10': 'cyan',
            'dqn_hier_hc=1': 'r',
            'dqn_hier_hc=10': 'pink',

            '1step': 'g',
            'nstep': 'b',
        }

    linestyle_map = {
        'ADET+Q* perfect': '--',
        'DQfD+Q* perfect': '--',
        'DQN+Q*+guided perfect': '--',
        'DAGGER perfect': '--'
    }
    name_map = {
        'DQN_Dueling_Double_Prioritized_replay': 'DQN',
        'QV_Dueling_Double_Prioritized_replay': 'QV_learning',
        'dqn_asr_resucat': 'DQN',
        'dqn_l2=1_asr_resucat': 'DQN l2=1',
        'dqn_l2=0.1_asr_resucat': 'DQN l2=0.1',
        'margin_asr_resucat': 'SDQfD',
        'v_asr_q_resucat_v_share_q': 'V+DQN',
        'v_l2=1_asr_q_resucat_v_share_q': 'V+DQN l2=1',
        'v_l2=0.1_asr_q_resucat_v_share_q': 'V+DQN l2=0.1',
        'v_margin_asr_q_resucat_v_share_q': 'V+SDQfD',
        'v_margin2_asr_q_resucat_v_share_q': 'V+Margin2',
        'v_eap_asr_q_resucat_v_e_share_q': 'V+NEP',
        'v_eap2_asr_q_resucat_v_share_q_e_resucat': 'V+NEP2',
        'v_eap2_1_asr_q_resucat_v_share_q_e_resucat': 'V+NEP3',

        'sdqfd': 'FCN SDQfD',
        'adet': 'FCN ADET',
        'dqfd': 'FCN DQfD',
        'BC': 'FCN BC',
        'dqn': 'FCN DQN',
        'ddpg': 'DDPG',

        'sdqfd_hier': 'ASRSE3 SDQfD',
        'adet_hier': 'ASRSE3 ADET',
        'dqfd_hier': 'ASRSE3 DQfD',
        'BC_hier': 'ASRSE3 BC',
        'dqn_hier': 'ASRSE3 DQN',
        'dqn_hier_q1tq2o': 'DQN hier no max',

        '1step': 'ASRSE3 DQN 1-step',
        'nstep': 'ASRSE3 DQN n-step',
    }

    sequence = {
        'sdqfd_hier': '0',
        'sdqfd': '0.5',
        'dqfd_hier': '1',
        'dqfd': '1.5',
        'adet_hier': '2',
        'adet': '2.5',
        'BC_hier': '3',
        'BC': '3.5',
        'dqn_hier': '4',
        'dqn': '4.5',
        'dqn_hier_q1tq2o': '5',
        'ddpg': '6'
    }

    i = 0
    methods = filter(lambda x: (x[0] != '.' and x.find(filer_pass_word) >= 0), get_immediate_subdirectories(base))
    for method in sorted(methods, key=lambda x: sequence[x] if x in sequence.keys() else x):
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                npz = np.load(os.path.join(base, method, run, 'learning_curve.npy.npz'))
                r = npz['rewards']
                epi = npz['episodes']
                if method.find('BC') >= 0:
                    rs.append(r)
                else:
                    # rs.append(r)
                    rs.append(getRewardsSingle(r, window=window_size))
            except Exception as e:
                print(e)
                continue

        if method.find('BC') >= 0:
            plt.plot([0, ep], [np.concatenate(rs).mean(), np.concatenate(rs).mean()],
                     color=color_map[method] if method in color_map else colors[i],
                     label=name_map[method] if method in name_map else method)
        else:
            # print(method[:method.find(filer_pass_word)])
            plotLearningCurveAvg(rs, window_size, label=name_map[method[:method.find(filer_pass_word)]]
                                 if method[:method.find(filer_pass_word)] in name_map
                                 else method[:method.find(filer_pass_word)],
                                 color=color_map[method] if method in color_map else colors[i],
                                 linestyle=linestyle_map[method] if method in linestyle_map else '-')
        i += 1

    # plt.legend(loc=0, facecolor='w', fontsize='x-large')
    plt.legend(loc=4, facecolor='w', fontsize=LEGEND_SIZE)
    plt.xlabel('number of episodes')
    plt.ylabel('task success rate')
    # plt.ylim((-0.01, 1.01))
    # plt.yticks(np.arange(0., 1.05, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(base, figname), bbox_inches='tight', pad_inches=0)


def showPerformance(base):
    methods = sorted(filter(lambda x: x[0] != '.', get_immediate_subdirectories(base)))
    for method in methods:
        rs = []
        for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
            try:
                r = np.load(os.path.join(base, method, run, 'info/rewards.npy'))
                rs.append(r[-1000:].mean())
            except Exception as e:
                print(e)
        print('{}: {:.3f}'.format(method, np.mean(rs)))


# def plotTDErrors():
#     plt.style.use('ggplot')
#     colors = "bgrycmkw"
#     method_map = {
#         'ADET': 'm',
#         'ADET+Q*': 'g',
#         'DAGGER': 'k',
#         'DQN': 'c',
#         'DQN+guided': 'y',
#         'DQN+Q*': 'b',
#         'DQN+Q*+guided': 'r',
#         "DQfD": 'chocolate',
#         "DQfD+Q*": 'grey'
#     }
#     i = 0
#
#     base = '/media/dian/hdd/unet/perlin'
#     for method in sorted(get_immediate_subdirectories(base)):
#         rs = []
#         if method[0] == '.' or method == 'DAGGER' or method == 'DQN':
#             continue
#         for j, run in enumerate(get_immediate_subdirectories(os.path.join(base, method))):
#             try:
#                 r = np.load(os.path.join(base, method, run, 'info/td_errors.npy'))
#                 rs.append(getRewardsSingle(r[:120000], window=1000))
#             except Exception as e:
#                 continue
#         if method in method_map:
#             plotLearningCurveAvg(rs, 1000, label=method, color=method_map[method])
#         else:
#             plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
#         # plotLearningCurveAvg(rs, 1000, label=method, color=colors[i])
#         i += 1
#
#     plt.legend(loc=0)
#     plt.xlabel('number of training steps')
#     plt.ylabel('TD error')
#     plt.yscale('log')
#     # plt.ylim((0.8, 0.93))
#     plt.show()

if __name__ == '__main__':
    base = '../../results'
    for goal in ['[0, 1]']:
        plotLearningCurve(base, 100000, filer_pass_word='_'+str(goal), figname='5x5x6 a50 [0, 1] exp20k')
    # showPerformance(base)
