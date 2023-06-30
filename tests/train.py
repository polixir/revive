'''
Copyright 2021-2023 Polixir Technologies Co., Ltd.

This file is the fast launch scipts for use revive.
'''
import os
import time
import argparse
from loguru import logger
from prettytable import PrettyTable

import revive
from revive.server import ReviveServer
from revive.data.dataset import DATADIR
from revive.utils.common_utils import load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default=None,
        help='the address of the ray cluster, by default the address is `None`, which means revive will start a cluster itself.')
    parser.add_argument('-df', '--data_file', type=str,
        help='name of the training `.npz` file.')
    parser.add_argument('-cf', '--config_file', type=str,
        help='name of the config `.yaml` file.')
    parser.add_argument('-umf', '--user_module_file_path', type=str, default=None,
        help='user module file path.')
    parser.add_argument('-mrf', '--matcher_reward_file', type=str, default=None,
        help='rule matcher reward file path.')
    parser.add_argument('-rf', '--reward_file', type=str, default=None,
        help='name of the reward function `.py` file, need to provide if you want to train policy.')
    parser.add_argument('-vf', '--val_file', type=str, default=None,
        help='name of the additional validation `.npz` file.')
    parser.add_argument('-rcf', '--revive_config_file', type=str, default=None,
        help='a json file that describes the parameters of revive')
    parser.add_argument('-tpn', '--target_policy_name', type=str, default=None, 
        help='control which policy to learn, if it is None, the first policy in the graph will be chosen.')
    parser.add_argument('--run_id', type=str, default=None,
        help='id of the running experiment, if it is not provide, revive will generate a random id.')
    parser.add_argument('--debug', action='store_true',
        help='whether to run in debug mode, in which training process will be short and log will be detailed.')
    parser.add_argument('-vm', '--venv_mode', type=str, default='tune', choices=['tune', 'once', 'None'],
        help='control the mode of venv training. `tune` means conducting hyper-parameter search; `once` means train with the default hyper-parameters; `None` means skip.')
    parser.add_argument('-pm', '--policy_mode', type=str, default='tune', choices=['tune', 'once', 'None'],
        help='control the mode of policy training. `tune` means conducting hyper-parameter search; `once` means train with the default hyper-parameters; `None` means skip.')
    parser.add_argument('-tm', '--tuning_mode', type=str, default='None', choices=['max', 'min', 'None'],
        help='control the mode of parameter tuning. `max` and `min` means enabling tuning and the direction; `None` means skip.')
    parser.add_argument('-tisf', '--tuning_initial_state_file', type=str, default=None,
        help='initial state of parameter tuning, needed when tuning mode is enabled.')
    args = parser.parse_known_args()[0].__dict__
    
    if os.path.exists(args['data_file']):
        DATADIR = "./"
            
    dataset_file_path = os.path.join(DATADIR, args['data_file'])
    dataset_desc_file_path = os.path.join(DATADIR, args['config_file'])
    dataset_val_file_path = os.path.join(DATADIR, args['val_file']) if not args['val_file'] is None else None
    user_module_file_path = os.path.join(DATADIR, args['user_module_file_path']) if not args['user_module_file_path'] is None else None
    matcher_reward_file_path = os.path.join(DATADIR, args['matcher_reward_file']) if not args['matcher_reward_file'] is None else None
    reward_file_path = os.path.join(DATADIR, args['reward_file']) if not args['reward_file'] is None else None
    revive_config_file_path = os.path.join(DATADIR, args['revive_config_file']) if not args['revive_config_file'] is None else None
    tuning_initial_state_file = os.path.join(DATADIR, args['tuning_initial_state_file']) if not args['tuning_initial_state_file'] is None else None

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(revive.__file__))), 'logs')

    if tuning_initial_state_file is not None:
        tune_initial_state = load_data(tuning_initial_state_file)
    else:
        tune_initial_state = None

    server = ReviveServer(dataset_file_path, dataset_desc_file_path, 
                          val_file_path=dataset_val_file_path, 
                          user_module_file_path=user_module_file_path,
                          matcher_reward_file_path=matcher_reward_file_path,
                          reward_file_path=reward_file_path, 
                          target_policy_name=args['target_policy_name'],
                          log_dir=log_dir,
                          run_id=args['run_id'],
                          address=args['address'], 
                          venv_mode=args['venv_mode'], 
                          policy_mode=args['policy_mode'], 
                          tuning_mode=args['tuning_mode'],
                          tune_initial_state=tune_initial_state,
                          debug=args['debug'],
                          revive_config_file_path=revive_config_file_path)
    server.train()

    while True:
        time.sleep(10)
        venv_status = server.get_virtualenv_env()
        policy_status = server.get_policy_model()
        tuning_status = server.get_parameter()

        # print(venv_status[:2], policy_status[:2], tuning_status)
        logger.info(f"Get the current model training status.")
        task_table = PrettyTable(["Task", "Task Status", "Model Accuracy", "Current Trail Number", "Total Number of Trails"])
        task_table.align["Task"] = "l"
        task_table.padding_width = 1 
        task_table.add_row(["Virtual Environment", venv_status[1]["task_state"], venv_status[1]["venv_acc"], venv_status[1]["current_num_of_trials"], venv_status[1]["total_num_of_trials"]])
        task_table.add_row(["Policy Model", policy_status[1]["task_state"], policy_status[1]["policy_acc"], policy_status[1]["current_num_of_trials"], policy_status[1]["total_num_of_trials"]])
        print(task_table)
        logger.info(task_table)

        try:
            if venv_status[1]['task_state'] == 'End' and policy_status[1]['task_state'] == 'End' and tuning_status[1]['task_state'] == 'End':
                break
        except:
            pass