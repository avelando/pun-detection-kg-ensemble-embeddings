import json

import pandas as pd

from pun_detection.base_view_cache import (
    load_base_view_cache,
)
from pun_detection.base_views import (
    select_base_view_matrix,
)
from pun_detection.config import (
    DATA,
    EXPERIMENT,
    PATHS,
)
from pun_detection.data import (
    load_development_splits,
)
from pun_detection.evaluation import (
    compute_binary_metrics,
    probabilities_to_predictions,
    summarize_binary_metrics,
)
from pun_detection.fingerprints import (
    array_fingerprint,
)
from pun_detection.models.stacking import (
    fit_stacking_meta_model,
    predict_stacking_probabilities,
)
from pun_detection.pairs import (
    twin_in_reference_mask,
)
from pun_detection.stacking_ablation import (
    STACKING_ABLATIONS,
    validate_stacking_ablations,
)
from pun_detection.stacking_selection import (
    get_selected_stacking_model,
)


def main():
    validate_stacking_ablations()

    splits = load_development_splits()

    train = splits.train
    validation = splits.validation

    selected_stacking_model = (
        get_selected_stacking_model(
            train=train,
            validation=validation,
        )
    )

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

    matrices_by_seed = {
        seed: load_base_view_cache(
            train=train,
            validation=validation,
            seed=seed,
        )
        for seed in EXPERIMENT.seeds
    }

    selected_embedding_models = {
        matrices.selected_embedding_model
        for matrices in matrices_by_seed.values()
    }

    if len(
        selected_embedding_models
    ) != 1:
        raise ValueError(
            "Base view caches use different "
            "selected embedding models"
        )

    selected_embedding_model = next(
        iter(
            selected_embedding_models
        )
    )

    prediction_output = pd.DataFrame(
        {
            "id": validation[
                DATA.id_column
            ].astype(str),
            "pair_id": validation[
                "pair_id"
            ].astype(str),
            "variant": validation[
                "variant"
            ].astype(str),
            "y_true": y_validation,
            "twin_in_train": twin_mask,
        }
    )

    configuration_results = {}

    for configuration_name, view_names in (
        STACKING_ABLATIONS.items()
    ):
        overall_runs = []
        twin_runs = []
        no_twin_runs = []
        per_seed = {}

        for seed in EXPERIMENT.seeds:
            matrices = matrices_by_seed[
                seed
            ]

            X_train = select_base_view_matrix(
                matrices.train_oof,
                view_names,
            )

            X_validation = (
                select_base_view_matrix(
                    matrices.validation,
                    view_names,
                )
            )

            model = fit_stacking_meta_model(
                X=X_train,
                y=y_train,
                model_name=selected_stacking_model,
                seed=seed,
                view_names=view_names,
            )

            probabilities = (
                predict_stacking_probabilities(
                    model=model,
                    X=X_validation,
                )
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

            overall_runs.append(
                overall
            )

            twin_runs.append(
                twin
            )

            no_twin_runs.append(
                no_twin
            )

            per_seed[
                str(seed)
            ] = {
                "overall": overall.as_dict(),
                "twin_in_train": twin.as_dict(),
                "no_twin_in_train": no_twin.as_dict(),
                "train_matrix_fingerprint": (
                    array_fingerprint(
                        X_train
                    )
                ),
                "validation_matrix_fingerprint": (
                    array_fingerprint(
                        X_validation
                    )
                ),
                "probability_fingerprint": (
                    array_fingerprint(
                        probabilities
                    )
                ),
            }

            prediction_output[
                f"{configuration_name}_"
                f"seed_{seed}_probability"
            ] = probabilities

            prediction_output[
                f"{configuration_name}_"
                f"seed_{seed}_prediction"
            ] = probabilities_to_predictions(
                probabilities
            )

            print(
                f"configuration={configuration_name}, "
                f"seed={seed}, "
                f"accuracy={overall.accuracy:.6f}, "
                f"macro_f1={overall.macro_f1:.6f}, "
                f"twin_macro_f1="
                f"{twin.macro_f1:.6f}, "
                f"no_twin_macro_f1="
                f"{no_twin.macro_f1:.6f}"
            )

        summary = {
            "overall": summarize_binary_metrics(
                overall_runs
            ),
            "twin_in_train": summarize_binary_metrics(
                twin_runs
            ),
            "no_twin_in_train": summarize_binary_metrics(
                no_twin_runs
            ),
        }

        configuration_results[
            configuration_name
        ] = {
            "views": list(
                view_names
            ),
            "per_seed": per_seed,
            "summary": summary,
        }

        overall_summary = summary[
            "overall"
        ]

        print(
            f"configuration={configuration_name}, "
            f"macro_f1_mean="
            f"{overall_summary['macro_f1']['mean']:.6f}, "
            f"macro_f1_std="
            f"{overall_summary['macro_f1']['std']:.6f}, "
            f"accuracy_mean="
            f"{overall_summary['accuracy']['mean']:.6f}, "
            f"accuracy_std="
            f"{overall_summary['accuracy']['std']:.6f}"
        )

        print()

    output_metrics = {
        "analysis_type": "stacking_ablation",
        "selection_role": "diagnostic_only",
        "selected_stacking_model": (
            selected_stacking_model
        ),
        "selected_embedding_model": (
            selected_embedding_model
        ),
        "seeds": list(
            EXPERIMENT.seeds
        ),
        "threshold": 0.5,
        "configurations": configuration_results,
    }

    PATHS.validation_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        PATHS.validation_results_dir
        / "stacking_ablation_predictions.csv"
    )

    metrics_path = (
        PATHS.validation_results_dir
        / "stacking_ablation_metrics.json"
    )

    prediction_output.to_csv(
        predictions_path,
        index=False,
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_metrics,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    print(
        f"selected_stacking_model="
        f"{selected_stacking_model}"
    )

    print(
        f"selected_embedding_model="
        f"{selected_embedding_model}"
    )

    print(
        "selection_role=diagnostic_only"
    )

    print(
        f"Saved predictions to "
        f"{predictions_path}"
    )

    print(
        f"Saved metrics to "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()