
config = {
    "LR": 2.5e-4,
    "NUM_ENVS": 32,
    "NUM_STEPS": 101, # MUST STAY 101!!!
    "TOTAL_TIMESTEPS": 2_000_000,
    "UPDATE_EPOCHS": 8,
    "NUM_MINIBATCHES": 4, # must be less than num_envs since RNN shuffles environemnts
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
    "USE_SOLVER": False,
    "TRANSFER_LEARNING": False,
    "TRANSFER_LEARNING_BASE_MODEL": "models/model_20250309-142808.pkl"

}