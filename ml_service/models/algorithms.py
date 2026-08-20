"""
models/algorithms.py
---------------------
Modelo de producción confirmado: Gradient Boosting Regressor con los
hiperparámetros exactos que arrojó el análisis en notebook (para que
el error de validación acá coincida con lo ya analizado):

    GradientBoostingRegressor(
        learning_rate=0.05, max_depth=2, min_samples_leaf=5,
        n_estimators=200, subsample=0.9, random_state=42,
    )

Ya no se comparan Ridge/Lasso/ElasticNet (RF09 original) — este es el
único algoritmo ML que se entrena y valida. El baseline GARCH (RF10)
se mantiene aparte, para la comparación de RF19.

RNF11: sigue existiendo `build_candidate_models()` devolviendo un
dict (aunque hoy tenga una sola entrada) para no tener que tocar
`training/walk_forward.py`, `training/retrain_manager.py` ni
`training/historical_backtest.py`, que iteran sobre ese dict. Agregar
otro algoritmo más adelante es agregar otra entrada ahí.
"""

from sklearn.ensemble import GradientBoostingRegressor

GRADIENT_BOOSTING_HYPERPARAMS = {
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_samples_leaf": 5,
    "n_estimators": 200,
    "subsample": 0.9,
    "random_state": 42,
}


def build_candidate_models() -> dict[str, GradientBoostingRegressor]:
    """Devuelve {nombre: modelo_sin_entrenar}. Se re-instancia en cada
    llamada para no arrastrar estado entre corridas de walk-forward."""
    return {
        "gradient_boosting_trees": GradientBoostingRegressor(**GRADIENT_BOOSTING_HYPERPARAMS),
    }
