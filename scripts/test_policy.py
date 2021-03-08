# This script is used to test policies on real environment

import ray
import argparse

from revive_core.utils import create_env
from revive_core.inference import load_policy, test_on_real_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='ib')
    parser.add_argument('-pf', '--policy_file', type=str)
    parser.add_argument('-pn', '--policy_name', type=str, default=None)
    parser.add_argument('-n', '--number_of_runs', type=int, default=10)
    args = parser.parse_args()

    ray.init()
    
    # Create the environment
    env = create_env(args.task)
    
    # Load trained policy
    policy = load_policy(args.policy_file, args.policy_name)

    # run test
    reward, length = test_on_real_env(env, policy, number_of_runs=args.number_of_runs)

    # Output result
    print('Test Result:')
    print('-' * 30)
    print(f'Reward: {reward}')
    print(f'Average Length: {length}')

    ray.shutdown()