###  Run the Lander Hover example use bc

    $ python train.py -df data/LanderHover.npz -cf data/LanderHover.yaml -rf data/LanderHover.py -vm once -pm once --venv_algo bc --run_id bc --ppo_epoch 1000

###  Run the Lander Hover example use revive

    $ python train.py -df data/LanderHover.npz -cf data/LanderHover.yaml -rf data/LanderHover.py  -rcf data/config.json -vm once -pm once --run_id revive --revive_epoch 1000 --ppo_epoch 3000
