
config = {
    "LR": 2e-4,
    "NUM_ENVS": 1,
    "NUM_STEPS": 505, # MUST STAY 101 or 505
    "TOTAL_TIMESTEPS": 3_000_000,
    "UPDATE_EPOCHS": 1,
    "NUM_MINIBATCHES": 1, # must be less than num_envs since RNN shuffles environemnts
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ANNEAL_LR": True,
    "DEBUG": True,
    "PROFILE": False,
    "CONVOLUTIONS": True,
    "TRANSFER_LEARNING": True,
    "TRANSFER_LEARNING_BASE_MODEL": "models/model_20250307-001220.pkl"
}