###  Run the refrigerator example use bc

    $ python train.py -df data/refrigeration.npz -cf data/refrigeration.yaml -rf data/refrigeration_reward.py -vm once -pm once --venv_algo bc --run_id bc --ppo_epoch 500

###  Run the refrigerator example use revive

    $ python train.py -df data/refrigeration.npz -cf data/refrigeration.yaml -rf data/refrigeration_reward.py  -rcf data/config.json -vm once -pm once --run_id revive --revive_epoch 1000 --ppo_epoch 3000


###  Run the refrigerator example use revive and the refrigeration_rule

    $ python train.py -df data/refrigeration.npz -cf data/refrigeration.yaml -rf data/refrigeration_reward.py -mrf data/refrigeration_rule.py  -rcf data/config.json -vm once -pm once --run_id test --revive_epoch 1000 --ppo_epoch 3000


