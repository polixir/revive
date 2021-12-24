import os
import pickle
import numpy as np

# Get model path
policy_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models","policy.pkl")
# Load policy model
policy_model = pickle.load(open(policy_model_path, 'rb'), encoding='utf-8')
# Generate false state data
state = {"obs":np.random.rand(2,15)}
# Inference with policy model
action = policy_model.infer(state)
print("Model input state:", state)
print("Model output action:", action)