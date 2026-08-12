import json

import pandas as pd

from pun_detection.config import (
    DATA,
    EXPERIMENT,
    PATHS,
)
from pun_detection.data import (
    load_development_splits,
)
from pun_detection.early_fusion import (
    EARLY_FUSION_CONFIGURATIONS,
    combine_early_fusion_components,
    fit_early_fusion_feature_space,
    validate_early_fusion_configurations,
)
from pun_detection.evaluation import (
    compute_binary_metrics,
    probabilities_to_predictions,
    summarize_binary_metrics,
)
from pun_detection.fingerprints import (
    array_fingerprint,
)
from pun_detection.models.base import (
    fit_logistic_classifier,
)
from pun_detection.pairs import (
    twin_in_reference_mask,
)


def main():
    validate_early_fusion_configurations()

    splits = load_development_splits()

    train = splits.train
    validation = splits.validation

    y_train = train[
        DATA.label_column
    ].astype(int).to_numpy()

    y_validation = validation[
        DATA.label_column
    ].astype(int).to_numpy()

    twin_mask = twin_in_reference_mask(
        validation,
        train,
    )

    no_twin_mask = ~twin_mask

    prediction_columns = {
        "id": validation[
            DATA.id_column
        ].astype(str).to_numpy(),
        "pair_id": validation[
            "pair_id"
        ].astype(str).to_numpy(),
        "variant": validation[
            "variant"
        ].astype(str).to_numpy(),
        "y_true": y_validation,
        "twin_in_train": twin_mask,
    }

    configuration_results = {}
    selected_embedding_models = set()

    for configuration_name, components in (
        EARLY_FUSION_CONFIGURATIONS.items()
    ):
        configuration_results[
            configuration_name
        ] = {
            "components": list(
                components
            ),
            "per_seed": {},
        }

    metric_runs = {
        configuration_name: {
            "overall": [],
            "twin_in_train": [],
            "no_twin_in_train": [],
        }
        for configuration_name in (
            EARLY_FUSION_CONFIGURATIONS
        )
    }

    for seed in EXPERIMENT.seeds:
        feature_space = (
            fit_early_fusion_feature_space(
                train=train,
                target=validation,
                target_split_name="validation",
                seed=seed,
            )
        )

        selected_embedding_models.add(
            feature_space.selected_embedding_model
        )

        for configuration_name, components in (
            EARLY_FUSION_CONFIGURATIONS.items()
        ):
            X_train = (
                combine_early_fusion_components(
                    feature_space.train_components,
                    components,
                )
            )

            X_validation = (
                combine_early_fusion_components(
                    feature_space.target_components,
                    components,
                )
            )

            classifier = fit_logistic_classifier(
                X=X_train,
                y=y_train,
                seed=seed,
            )

            probabilities = (
                classifier.predict_proba(
                    X_validation
                )[:, 1]
            )

            overall = compute_binary_metrics(
                y_validation,
                probabilities,
            )

            twin = compute_binary_metrics(
                y_validation[twin_mask],
                probabilities[twin_mask],
            )

            no_twin = compute_binary_metrics(
                y_validation[no_twin_mask],
                probabilities[no_twin_mask],
            )

            metric_runs[
                configuration_name
            ][
                "overall"
            ].append(
                overall
            )

            metric_runs[
                configuration_name
            ][
                "twin_in_train"
            ].append(
                twin
            )

            metric_runs[
                configuration_name
            ][
                "no_twin_in_train"
            ].append(
                no_twin
            )

            configuration_results[
                configuration_name
            ][
                "per_seed"
            ][
                str(seed)
            ] = {
                "feature_dimension": int(
                    X_train.shape[1]
                ),
                "overall": overall.as_dict(),
                "twin_in_train": twin.as_dict(),
                "no_twin_in_train": (
                    no_twin.as_dict()
                ),
                "probability_fingerprint": (
                    array_fingerprint(
                        probabilities
                    )
                ),
            }

            prediction_columns[
                f"{configuration_name}_"
                f"seed_{seed}_probability"
            ] = probabilities

            prediction_columns[
                f"{configuration_name}_"
                f"seed_{seed}_prediction"
            ] = probabilities_to_predictions(
                probabilities
            )

            print(
                f"configuration="
                f"{configuration_name}, "
                f"seed={seed}, "
                f"features={X_train.shape[1]}, "
                f"accuracy={overall.accuracy:.6f}, "
                f"macro_f1={overall.macro_f1:.6f}, "
                f"twin_macro_f1="
                f"{twin.macro_f1:.6f}, "
                f"no_twin_macro_f1="
                f"{no_twin.macro_f1:.6f}"
            )

    if len(
        selected_embedding_models
    ) != 1:
        raise ValueError(
            "Early fusion used different selected "
            "embedding models"
        )

    selected_embedding_model = next(
        iter(
            selected_embedding_models
        )
    )

    for configuration_name in (
        EARLY_FUSION_CONFIGURATIONS
    ):
        summary = {
            "overall": summarize_binary_metrics(
                metric_runs[
                    configuration_name
                ][
                    "overall"
                ]
            ),
            "twin_in_train": summarize_binary_metrics(
                metric_runs[
                    configuration_name
                ][
                    "twin_in_train"
                ]
            ),
            "no_twin_in_train": summarize_binary_metrics(
                metric_runs[
                    configuration_name
                ][
                    "no_twin_in_train"
                ]
            ),
        }

        configuration_results[
            configuration_name
        ][
            "summary"
        ] = summary

    output = {
        "analysis_type": "early_fusion",
        "selection_role": (
            "predefined_comparison"
        ),
        "selected_embedding_model": (
            selected_embedding_model
        ),
        "seeds": list(
            EXPERIMENT.seeds
        ),
        "threshold": 0.5,
        "configurations": (
            configuration_results
        ),
    }

    PATHS.validation_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        PATHS.validation_results_dir
        / "early_fusion_metrics.json"
    )

    predictions_path = (
        PATHS.validation_results_dir
        / "early_fusion_predictions.csv"
    )

    with metrics_path.open(
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

    pd.DataFrame(
        prediction_columns
    ).to_csv(
        predictions_path,
        index=False,
    )

    print()

    print(
        f"selected_embedding_model="
        f"{selected_embedding_model}"
    )

    for configuration_name in (
        EARLY_FUSION_CONFIGURATIONS
    ):
        overall = configuration_results[
            configuration_name
        ][
            "summary"
        ][
            "overall"
        ]

        print(
            f"configuration="
            f"{configuration_name}, "
            f"macro_f1_mean="
            f"{overall['macro_f1']['mean']:.6f}, "
            f"macro_f1_std="
            f"{overall['macro_f1']['std']:.6f}, "
            f"accuracy_mean="
            f"{overall['accuracy']['mean']:.6f}, "
            f"accuracy_std="
            f"{overall['accuracy']['std']:.6f}"
        )

    print(
        f"Saved metrics to {metrics_path}"
    )

    print(
        f"Saved predictions to {predictions_path}"
    )


if __name__ == "__main__":
    main()