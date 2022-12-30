import torch
import warnings
import pickle 
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation

try:
    from tianshou.data import Batch
except:
    pass

warnings.filterwarnings('ignore')
np.random.seed(17)

def save_frames_as_gif(frames, render_path):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(render_path, writer='imagemagick', fps=60)


class PolicyTest:
    def __init__(self, policy):
        if isinstance(policy, str):
            try:
                self.policy_mode = "revive"
                self.policy = pickle.load(open(policy, 'rb'), encoding='utf-8')
            except:
                self.policy_mode = "tianshou"
                self.policy = torch.load(policy)
        else:
            self.policy_mode = "function"
            self.policy = policy


    def __call__(self, obs):
        if len(obs.shape) == 1:
            obs = obs.reshape(1,-1)
        
        obs = {"obs":obs}
        if self.policy_mode == "revive":
            act = self.policy.infer(obs)[0]
        elif self.policy_mode == "tianshou":
            obs["info"] = {}
            obs['obs'] = np.concatenate([obs['obs'], np.array([0.]*4).reshape(-1, 4)], axis=1)
            obs = Batch(obs)
            act = self.policy(obs)["act"][0]
        else:
            act = self.policy(obs)

        return act

    def test_on_env(self, env, render_path=None):
        cur_state,_ = env.reset()
        cur_step = 0
        track_r = []
        frames = []
        while True:
            action = self(cur_state)
            if render_path:
                frames.append(env.render(mode="rgb_array"))

            next_state, reward, done, *_ = env.step(action)
            track_r.append(reward)
            cur_state = next_state
            cur_step += 1

            if done:
                ep_rs_sum = sum(track_r)
                break
        if frames:
            print(f"frames: {len(frames)}")
            save_frames_as_gif(frames, render_path)

        result =  {"reward" : np.sum(track_r), "steps" :cur_step}

        return result 

    def get_data(self, env, trj_num, dataset_save_path=None):
        episode_rewards = []
        episode_lengths = []

        dataset = {
            "obs": [],
            "action": [],
            "next_obs": [],
            "rew": [],
            "done": [],
            "index": [],
        }
        for trj_index in range(trj_num):
            state = env.reset()
            done = False
            rewards = 0
            lengths = 0
            step = 0
            old_x = 0
            delta_x = state[0]
            while not done:
                dataset['obs'].append(state)

                state = state[np.newaxis]

                state[0][0] = delta_x
                action = self(state)
                #print(action)
                #action = env.action_space.sample()
                #print(action)

                dataset['action'].append(action)
                new_state, reward, done, *_ = env.step(action)
                delta_x = new_state[0] - state[0][0]
                state = new_state
                dataset['next_obs'].append(state)
                dataset['rew'].append(reward)
                dataset['done'].append(done)

                rewards += reward
                lengths += 1
                step+= 1
                old_x = state[0]
                if step>= 1000:
                    done = True


            if not dataset['index']:
                dataset['index'].append(lengths)
            else:
                dataset['index'].append(dataset['index'][-1]+lengths)


            episode_rewards.append(rewards)
            episode_lengths.append(lengths)

        rew_mean = np.mean(episode_rewards)
        len_mean = np.mean(episode_lengths)

        for k, v in dataset.items():
            if k == "index":
                continue
            dataset[k] = np.array(v).reshape(dataset['index'][-1],-1)

        if dataset_save_path:
            np.savez_compressed(dataset_save_path, **dataset)


        return dataset, rew_mean
