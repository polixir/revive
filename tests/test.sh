python tests/test_dists.py
python tests/test_processor.py 
python train.py -df data/test.npz -cf data/test.yaml -rf data/test_reward.py -vm once -pm None --run_id test_1 --revive_epoch 5 --ppo_epoch 5
python train.py -df data/test.npz -cf data/test.yaml -rf data/test_reward.py -vm None -pm once --run_id test_1 --revive_epoch 5 --ppo_epoch 5
python train.py -df data/test.npz -cf data/test.yaml -rf data/test_reward.py -vm once -pm once --run_id test_2 --revive_epoch 5 --ppo_epoch 5 --venv_algo revive -vgpw 0.25 -pgpw 0.25  --train_venv_trials 10 --train_policy_trials 10
python train.py -df data/test.npz -cf tests/data/test_metric_nodes.yaml -rf data/test_reward.py -vm tune -pm tune --run_id test_3 --revive_epoch 5 --ppo_epoch 5 --train_venv_trials 10 --train_policy_trials 10 -vgpw 0.25 -pgpw 0.25
python train.py -df data/test.npz -cf tests/data/test_dim_fit.yaml -rf data/test_reward.py -vm once -pm None --run_id test_4 --revive_epoch 5 --ppo_epoch 5  --train_venv_trials 10 --train_policy_trials 10
python train.py -df data/test.npz -cf tests/data/test_metric_nodes.yaml -rf data/test_reward.py -rcf data/config.json -vm tune -pm None --run_id test_5 --revive_epoch 5 --ppo_epoch 5 --train_venv_trials 10 --train_policy_trials 10
python train.py -df data/test.npz -cf tests/data/test_metric_nodes.yaml -rf data/test_reward.py -rcf data/config.json -vm None -pm once --run_id test_5 --revive_epoch 5 --ppo_epoch 5 --train_venv_trials 10 --train_policy_trials 10
python train.py -df tests/data/test_mis_match.npz -cf data/test.yaml -rf data/test_reward.py -vm once -pm None --run_id test_6 --revive_epoch 5 --ppo_epoch 5 --ignore_check 1 
python train.py -df data/test.npz -cf tests/data/test_freeze_node.yaml -rf data/test_reward.py -vm once -pm None --run_id test_6 --revive_epoch 5 --ppo_epoch 5 
python train.py -df tests/data/test_nan.npz -cf examples/task/Pendulum/data/Env-GAIL-pendulum.yaml -vm once -pm None --run_id test_7 --revive_epoch 5 --ppo_epoch 5

python tests/test.py