###  Train the IB example use revive

    $ python train.py -df data/ib.npz -cf data/ib_env.yaml -rf data/ib_reward.py -rcf data/config.json -vm once -pm None --run_id revive

    $ python train.py -df data/ib.npz -cf data/ib_policy.yaml -rf data/ib_reward.py -rcf data/config.json -vm None -pm once --run_id revive
