from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from pun_detection.base_views import (
    validate_base_view_names,
)
from pun_detection.config import (
    BASE_MODELS,
    STACKING,
)
from pun_detection.models.base import (
    fit_logistic_classifier,
)


@dataclass(frozen=True)
class StackingMetaModel:
    model_name: str
    classifier: object
    view_names: tuple[str, ...]


def stacking_classifier_config(
    model_name: str,
    seed: int,
) -> dict:
    if model_name == "logistic_regression":
        return {
            "type": "logistic_regression",
            "C": BASE_MODELS.logistic_c,
            "solver": BASE_MODELS.logistic_solver,
            "max_iter": BASE_MODELS.logistic_max_iter,
            "seed": seed,
        }

    if model_name == "random_forest":
        return {
            "type": "random_forest",
            "n_estimators": STACKING.random_forest_estimators,
            "criterion": STACKING.random_forest_criterion,
            "max_depth": STACKING.random_forest_max_depth,
            "max_features": STACKING.random_forest_max_features,
            "seed": seed,
        }

    if model_name == "gradient_boosting":
        return {
            "type": "gradient_boosting",
            "n_estimators": STACKING.gradient_boosting_estimators,
            "learning_rate": STACKING.gradient_boosting_learning_rate,
            "max_depth": STACKING.gradient_boosting_max_depth,
            "subsample": STACKING.gradient_boosting_subsample,
            "seed": seed,
        }

    raise ValueError(
        f"Unknown stacking meta-classifier: {model_name}"
    )


def validate_stacking_input(
    matrix: np.ndarray,
    matrix_name: str,
    expected_features: int,
) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            f"{matrix_name} must be two-dimensional"
        )

    if matrix.shape[1] != expected_features:
        raise ValueError(
            f"{matrix_name} has an invalid "
            "number of features"
        )

    if not np.isfinite(
        matrix
    ).all():
        raise ValueError(
            f"{matrix_name} contains non-finite values"
        )

    if (
        np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
    ):
        raise ValueError(
            f"{matrix_name} contains invalid probabilities"
        )

    return matrix


def fit_stacking_meta_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    seed: int,
    view_names=None,
) -> StackingMetaModel:
    if view_names is None:
        view_names = STACKING.primary_views

    view_names = validate_base_view_names(
        view_names
    )

    X = validate_stacking_input(
        X,
        "stacking_train",
        expected_features=len(
            view_names
        ),
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    if y.ndim != 1:
        raise ValueError(
            "Stacking targets must be one-dimensional"
        )

    if len(y) != len(X):
        raise ValueError(
            "Stacking inputs and targets have different sizes"
        )

    if set(
        np.unique(y)
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Stacking targets must contain both binary classes"
        )

    if model_name == "logistic_regression":
        classifier = fit_logistic_classifier(
            X=X,
            y=y,
            seed=seed,
        )

    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=STACKING.random_forest_estimators,
            criterion=STACKING.random_forest_criterion,
            max_depth=STACKING.random_forest_max_depth,
            max_features=STACKING.random_forest_max_features,
            random_state=seed,
            n_jobs=1,
        )

        classifier.fit(
            X,
            y,
        )

    elif model_name == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=STACKING.gradient_boosting_estimators,
            learning_rate=STACKING.gradient_boosting_learning_rate,
            max_depth=STACKING.gradient_boosting_max_depth,
            subsample=STACKING.gradient_boosting_subsample,
            random_state=seed,
        )

        classifier.fit(
            X,
            y,
        )

    else:
        raise ValueError(
            f"Unknown stacking meta-classifier: {model_name}"
        )

    return StackingMetaModel(
        model_name=model_name,
        classifier=classifier,
        view_names=view_names,
    )


def predict_stacking_probabilities(
    model: StackingMetaModel,
    X: np.ndarray,
) -> np.ndarray:
    X = validate_stacking_input(
        X,
        "stacking_prediction",
        expected_features=len(
            model.view_names
        ),
    )

    probabilities = (
        model.classifier.predict_proba(
            X
        )[:, 1]
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Stacking produced non-finite probabilities"
        )

    if (
        np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError(
            "Stacking produced invalid probabilities"
        )

    return probabilities