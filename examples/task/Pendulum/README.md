###  Run the pendulum example use revive

    $ python train.py -df data/expert_data.npz -cf data/Env-GAIL-pendulum.yaml -rf data/pendulum-reward.py -rcf data/config.json -vm once -pm once --run_id pendulum-v1 --revive_epoch 1000 --ppo_epoch 5000