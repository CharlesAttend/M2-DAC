import optuna
import torch.nn as nn


def objective(trial):
    from tp7 import Model, run, NUM_CLASSES, INPUT_DIM
    iterations = 200
    dims = [100, 100, 100]

    norm_type = trial.suggest_categorical('normalization', ["identity", "batchnorm", "layernorm"])
    normalization = norm_type

    dropouts = [trial.suggest_loguniform('dropout_p%d' % ix, 1e-2, 0.5) for ix in range(len(dims))]

    l2 = trial.suggest_uniform('l2', 0, 1)
    l1 = trial.suggest_uniform('l1', 0, 1)

    model = Model(INPUT_DIM, NUM_CLASSES, dims, dropouts, normalization)
    return run(iterations, model, l1, l2)

study = optuna.create_study()
study.optimize(objective, n_trials=20)
print(study.best_params)
