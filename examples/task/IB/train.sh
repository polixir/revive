python train.py -df data/ib.npz -cf data/ib_env.yaml -rf data/ib_reward.py -rcf data/config.json -vm tune -pm None --run_id revive

python train.py -df data/ib.npz -cf data/ib_policy.yaml -rf data/ib_reward.py -rcf data/config.json -vm None -pm tune --run_id revive