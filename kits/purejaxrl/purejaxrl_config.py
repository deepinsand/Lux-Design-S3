
config = {
    "LR": 2.5e-4,
    "NUM_ENVS": 4,
    "NUM_STEPS": 505, # MUST STAY 101!!!
    "TOTAL_TIMESTEPS": 200_000,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 1, # must be less than num_envs since RNN shuffles environemnts
    "GAMMA": 0.995,
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
}