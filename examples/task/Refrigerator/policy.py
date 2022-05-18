import pickle
import numpy as np


class SamplingPolicy():
    """ """
    def __init__(self, noise_prob=1.0, noise_ratio=0.1, target_temperature=-2, p=0.2, i=0.0, d=0.0):
        self.noise_prob = noise_prob
        self.noise_ratio = noise_ratio
        self.target_temperature = target_temperature -1.75
        self.p = p
        self.i = i
        self.d = d
        self.error = 0
        self.error_int = 0
        self.error_diff = 0


    def act(self, state):
        if np.random.random() < self.noise_prob:
            noise = np.random.normal(0, 0.1)
        else:
            noise = 0.0
        error = state - self.target_temperature
        self.error_diff = error - self.error
        self.error = state - self.target_temperature
        self.error_int = self.error_int + self.error
        action = self.p*self.error + self.i * self.error_int + self.d * self.error_diff

        return action + noise if action + noise >0 else 0

    def set_parameters(self, p):
        self.p = p
    
    def set_target_temperature(self, target_temperature):
        self.target_temperature = target_temperature

class VenPolicy():
    """Strategies for using environment initialization"""
    def __init__(self, policy_model_path):
        self.policy_model = pickle.load(open(policy_model_path, 'rb'))

    def act(self, state):
        new_state = {}
        new_state['temperature'] = np.array([state])
        new_state['door_open']   = np.array([0])

        try:
            next_state = self.policy_model.infer(new_state)
        except:
            next_state = self.policy_model.infer_one_step(new_state)["action"]

        return next_state[0]