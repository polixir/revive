import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from env import DoorOpen, Simulator
from policy import SamplingPolicy, VenPolicy


def sample_trajectory(cool_simulator, 
                      sample_agent,
                      door_open_agent, 
                      running_step,
                      init_temperature=10,
                      sampling_time=1,
                      open_interval=50):
    '''Sampling trajectory using sample agent on cool_simulator.'''
    init_temperature = init_temperature
    cool_simulator.reset(init_temperature=init_temperature)
    trajectory = {
        "door_open": [],
        "temperature": [],
        "action": [],
        "next_temperature": [],
    }
    temperature = cool_simulator.get_temperature()
    open_door = False
    door_open_after_step = open_interval + 1
    for step in range(running_step):
        if step % open_interval == 0:
            if random.random() < 0.5:
                open_door = True
            door_open_agent.reset()
            door_open_after_step = random.randint(0, open_interval - door_open_agent.door_open_time)
        action = sample_agent.act(temperature)
        # action = max(action, 0)
        # action = min(action, 1)
        if open_door and step % open_interval >= door_open_after_step:
            door_open = door_open_agent.act()
            cool_simulator.update(power=action, dt=sampling_time, door_open=door_open)
        else:
            door_open = False
            cool_simulator.update(power=action, dt=sampling_time, door_open=door_open)
        next_temperature = cool_simulator.get_temperature()
        trajectory["door_open"].append(float(door_open))
        trajectory["temperature"].append(temperature)
        trajectory["action"].append(action)
        trajectory["next_temperature"].append(next_temperature)
        temperature = next_temperature

    trajectory = {k:np.array(v) for k,v in trajectory.items()}
    
    return trajectory

def test_on_env(policy_path, 
                evaluate_epoch=1,
                target_temperature = -2,
                init_temperature = 10,
                running_step = 2000,):
    '''Run the policy on real environment.'''
    sample_policy = VenPolicy(policy_path)
    door_open_agent = DoorOpen(door_open_time=20)
    cool_simulator = Simulator(init_temperature)

    traj = sample_trajectory(cool_simulator, sample_policy, door_open_agent, 2000, open_interval=200)
    generate_data()
    print(f"Reward in Policy : {-np.mean(np.abs(traj['next_temperature'] - (target_temperature)))}")

    return traj
    """
    import matplotlib.pyplot as plt
    plt.plot(traj['action'])
    plt.plot(traj['next_temperature'])
    target_temp = target_temperature
    plt.plot([0, 2000], [target_temp, target_temp], 'r--')
    plt.show()
    plt.savefig("res.png")
    """


def generate_data(data_save_path=None):
    sim = Simulator()
    target_temp = -2
    ref_policy = SamplingPolicy(target_temperature=target_temp, p=0.2, i=0.00, d=0.)
    # ref_policy = SamplingPolicy(target_temperature=target_temp, p=1.5, i=0.00, d=0.)
    # ref_policy = SamplingPolicy(target_temperature=target_temp, p=0.1, i=0.005, d=0.4)
    door_open_policy = DoorOpen(door_open_time=20)
    traj = sample_trajectory(sim, ref_policy, door_open_policy, 2000, open_interval=200)
    print(f"Reward in Data : {-np.mean(np.abs(traj['next_temperature'] - (-2)))}")
    """
    import matplotlib.pyplot as plt
    plt.plot(traj['next_temperature'])
    plt.plot([0, 2000], [target_temp, target_temp], 'r--')
    plt.show()
    plt.savefig("res.png")
    """
    if data_save_path:
        traj = { k:v.reshape(-1,1) for k,v in traj.items()}
        traj["index"] = np.array([2000,])
        np.savez_compressed(data_save_path, **traj)

def plt_env(env_model_path):
    env_model = pickle.load(open(env_model_path, 'rb'))
    new_state = {}
    new_state['temperature'] = np.array([0])
    new_state['door_open']   = np.array([0])

    sim = Simulator()
    target_temp = -2
    ref_policy = SamplingPolicy(target_temperature=target_temp, p=0.2, i=0.00, d=0.)
    door_open_policy = DoorOpen(door_open_time=20)
    traj = sample_trajectory(sim, ref_policy, door_open_policy, 2000, open_interval=200)
    flag = 0
    for i,temperature in enumerate(traj["temperature"]):
        door_open = traj["door_open"][i]
        action = traj["action"][i]
        next_temperature = traj["next_temperature"][i]

        new_state = {}
        new_state['temperature'] = np.array([temperature,])
        new_state['door_open']   = np.array([door_open,])
        model_next_action = env_model.infer_one_step(new_state)["action"][0]
        new_state['action']   = np.array([action,])

        model_next_temperature = env_model.infer_one_step(new_state)["next_temperature"][0]
        #model_action = env_model.infer_one_step(new_state)["action"][0]
        if (next_temperature - temperature) * (model_next_temperature - temperature) < 0:
            flag += 1
        else:
            flag += 0

        print(temperature,door_open,"|",action,model_next_action,"|",next_temperature,model_next_temperature, "|", flag/(i+1))



if __name__ == "__main__":
    #generate_data()
    plt_env("./logs/revive/env.pkl")
    #test_on_env("../logs/refrigeration/policy.pkl")
    