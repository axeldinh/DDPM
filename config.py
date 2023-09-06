model_params = {
    "in_channels": 1,
    "n_feat": 64,
    "nc_feat": 10,
    "height": 28,
    "beta_1": 1e-4,
    "beta_2": 0.02,
    "timesteps": 1000,
    "device": "cuda",
    "save_dir": "models"
}

hyperparams = {
    "batch_size": 128,
    "learning_rate": 0.001,
    "epochs": 20,
}

