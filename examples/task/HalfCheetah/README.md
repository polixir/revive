###  1. Download dataset

    $ cd data/   # [in data/ directory]
    $ python generate_data.py

###  2. Run the halfcheetah example use revive

    $ cd ..   # [in HalfCheetah/ directory]
    $ python train.py -df data/halfcheetah-medium-v2.npz -cf data/halfcheetah-medium-v2.yaml -rf data/halfcheetah_reward.py -rcf data/config.json --target_policy_name action -vm once -pm once --run_id halfcheetah-medium-v2-revive --revive_epoch 1500 --sac_epoch 1500