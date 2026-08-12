import json

import numpy as np

from pun_detection.base_view_cache import (
    load_base_view_cache,
)
from pun_detection.base_views import (
    BASE_VIEW_NAMES,
)
from pun_detection.config import (
    DATA,
    EXPERIMENT,
    PATHS,
)
from pun_detection.data import (
    load_development_splits,
)
from pun_detection.models.stacking import (
    fit_stacking_meta_model,
)
from pun_detection.stacking_selection import (
    get_selected_stacking_model,
)


def summarize_values(
    values,
) -> dict:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(
            array.mean()
        ),
        "std": float(
            array.std(
                ddof=1
            )
            if len(array) > 1
            else 0.0
        ),
        "min": float(
            array.min()
        ),
        "max": float(
            array.max()
        ),
    }


def main():
    splits = load_development_splits()

    train = splits.train
    validation = splits.validation

    selected_model = (
        get_selected_stacking_model(
            train=train,
            validation=validation,
        )
    )

    if selected_model != "logistic_regression":
        raise ValueError(
            "Coefficient analysis requires "
            "logistic_regression to be selected"
        )

    y_train = train[
        DATA.label_column
    ].astype(int).to_numpy()

    per_seed = {}

    coefficient_values = {
        view_name: []
        for view_name in BASE_VIEW_NAMES
    }

    feature_mean_values = {
        view_name: []
        for view_name in BASE_VIEW_NAMES
    }

    feature_std_values = {
        view_name: []
        for view_name in BASE_VIEW_NAMES
    }

    effect_per_std_values = {
        view_name: []
        for view_name in BASE_VIEW_NAMES
    }

    intercept_values = []

    for seed in EXPERIMENT.seeds:
        matrices = load_base_view_cache(
            train=train,
            validation=validation,
            seed=seed,
        )

        model = fit_stacking_meta_model(
            X=matrices.train_oof,
            y=y_train,
            model_name=selected_model,
            seed=seed,
            view_names=BASE_VIEW_NAMES,
        )

        coefficients = np.asarray(
            model.classifier.coef_[0],
            dtype=np.float64,
        )

        feature_means = np.asarray(
            matrices.train_oof.mean(
                axis=0
            ),
            dtype=np.float64,
        )

        feature_stds = np.asarray(
            matrices.train_oof.std(
                axis=0,
                ddof=0,
            ),
            dtype=np.float64,
        )

        if coefficients.shape != (
            len(BASE_VIEW_NAMES),
        ):
            raise ValueError(
                "Unexpected stacking coefficient shape"
            )

        if np.any(
            feature_stds <= 0.0
        ):
            raise ValueError(
                "Stacking contains a constant base view"
            )

        effects_per_std = (
            coefficients
            * feature_stds
        )

        intercept = float(
            model.classifier.intercept_[0]
        )

        seed_coefficients = {}
        seed_feature_means = {}
        seed_feature_stds = {}
        seed_effects_per_std = {}

        for index, view_name in enumerate(
            BASE_VIEW_NAMES
        ):
            coefficient = float(
                coefficients[index]
            )

            feature_mean = float(
                feature_means[index]
            )

            feature_std = float(
                feature_stds[index]
            )

            effect_per_std = float(
                effects_per_std[index]
            )

            coefficient_values[
                view_name
            ].append(
                coefficient
            )

            feature_mean_values[
                view_name
            ].append(
                feature_mean
            )

            feature_std_values[
                view_name
            ].append(
                feature_std
            )

            effect_per_std_values[
                view_name
            ].append(
                effect_per_std
            )

            seed_coefficients[
                view_name
            ] = coefficient

            seed_feature_means[
                view_name
            ] = feature_mean

            seed_feature_stds[
                view_name
            ] = feature_std

            seed_effects_per_std[
                view_name
            ] = effect_per_std

        intercept_values.append(
            intercept
        )

        per_seed[
            str(seed)
        ] = {
            "coefficients": seed_coefficients,
            "feature_means": seed_feature_means,
            "feature_stds": seed_feature_stds,
            "effect_per_std": seed_effects_per_std,
            "intercept": intercept,
        }

        formatted = ", ".join(
            f"{view_name}="
            f"{seed_effects_per_std[view_name]:.6f}"
            for view_name in BASE_VIEW_NAMES
        )

        print(
            f"seed={seed}, "
            f"effect_per_std: "
            f"{formatted}"
        )

    summary = {}

    for view_name in BASE_VIEW_NAMES:
        summary[
            view_name
        ] = {
            "coefficient": summarize_values(
                coefficient_values[
                    view_name
                ]
            ),
            "feature_mean": summarize_values(
                feature_mean_values[
                    view_name
                ]
            ),
            "feature_std": summarize_values(
                feature_std_values[
                    view_name
                ]
            ),
            "effect_per_std": summarize_values(
                effect_per_std_values[
                    view_name
                ]
            ),
        }

    output = {
        "analysis_type": (
            "stacking_coefficient_analysis"
        ),
        "selection_role": "diagnostic_only",
        "selected_stacking_model": (
            selected_model
        ),
        "views": list(
            BASE_VIEW_NAMES
        ),
        "seeds": list(
            EXPERIMENT.seeds
        ),
        "per_seed": per_seed,
        "summary": summary,
        "intercept": summarize_values(
            intercept_values
        ),
    }

    output_path = (
        PATHS.validation_results_dir
        / "stacking_coefficients.json"
    )

    PATHS.validation_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    print()

    for view_name in BASE_VIEW_NAMES:
        values = summary[
            view_name
        ]

        print(
            f"view={view_name}, "
            f"coefficient_mean="
            f"{values['coefficient']['mean']:.6f}, "
            f"feature_std_mean="
            f"{values['feature_std']['mean']:.6f}, "
            f"effect_per_std_mean="
            f"{values['effect_per_std']['mean']:.6f}"
        )

    print(
        f"Saved coefficient analysis to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()