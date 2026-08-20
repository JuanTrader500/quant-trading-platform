from models.algorithms import GRADIENT_BOOSTING_HYPERPARAMS, build_candidate_models


def test_confirmed_hyperparameters_match_notebook_analysis():
    expected = {
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 5,
        "n_estimators": 200,
        "subsample": 0.9,
        "random_state": 42,
    }
    assert GRADIENT_BOOSTING_HYPERPARAMS == expected


def test_build_candidate_models_returns_only_gradient_boosting():
    models = build_candidate_models()
    assert list(models.keys()) == ["gradient_boosting_trees"]
    model = models["gradient_boosting_trees"]
    assert model.get_params()["learning_rate"] == 0.05
    assert model.get_params()["max_depth"] == 2
    assert model.get_params()["min_samples_leaf"] == 5
    assert model.get_params()["n_estimators"] == 200
    assert model.get_params()["subsample"] == 0.9
    assert model.get_params()["random_state"] == 42
