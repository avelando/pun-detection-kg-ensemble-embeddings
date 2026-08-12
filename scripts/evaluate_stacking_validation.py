import json

import pandas as pd

from pun_detection.base_view_cache import (
    load_base_view_cache,
)
from pun_detection.base_views import (
    BASE_VIEW_NAMES,
    select_base_view_matrix,
)
from pun_detection.config import (
    DATA,
    EXPERIMENT,
    PATHS,
    STACKING,
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
    stacking_classifier_config,
)
from pun_detection.pairs import (
    twin_in_reference_mask,
)
from pun_detection.stacking_selection import (
    build_stacking_selection,
    save_stacking_selection,
)


def main():
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

    matrices_by_seed = {}

    for seed in EXPERIMENT.seeds:
        matrices_by_seed[
            seed
        ] = load_base_view_cache(
            train=train,
            validation=validation,
            seed=seed,
        )

    selected_embedding_models = {
        matrices.selected_embedding_model
        for matrices in matrices_by_seed.values()
    }

    if len(
        selected_embedding_models
    ) != 1:
        raise ValueError(
            "Base view caches use different selected "
            "embedding models"
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

    selection_scores = {}
    model_results = {}

    for model_name in STACKING.candidates:
        overall_runs = []
        twin_runs = []
        no_twin_runs = []
        per_seed = {}
        selection_runs = []

        for seed in EXPERIMENT.seeds:
            matrices = matrices_by_seed[
                seed
            ]

            if matrices.columns != BASE_VIEW_NAMES:
                raise ValueError(
                    "Unexpected base view column order"
                )

            X_train = select_base_view_matrix(
                matrices.train_oof,
                STACKING.primary_views,
            )

            X_validation = select_base_view_matrix(
                matrices.validation,
                STACKING.primary_views,
            )

            model = fit_stacking_meta_model(
                X=X_train,
                y=y_train,
                model_name=model_name,
                seed=seed,
                view_names=STACKING.primary_views,
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

            selection_runs.append(
                {
                    "seed": seed,
                    "macro_f1": overall.macro_f1,
                    "accuracy": overall.accuracy,
                }
            )

            per_seed[
                str(seed)
            ] = {
                "classifier": stacking_classifier_config(
                    model_name=model_name,
                    seed=seed,
                ),
                "overall": overall.as_dict(),
                "twin_in_train": twin.as_dict(),
                "no_twin_in_train": no_twin.as_dict(),
                "train_oof_fingerprint": array_fingerprint(
                    X_train
                ),
                "validation_matrix_fingerprint": array_fingerprint(
                    X_validation
                ),
                "probability_fingerprint": array_fingerprint(
                    probabilities
                ),
            }

            prediction_output[
                f"{model_name}_seed_{seed}_probability"
            ] = probabilities

            prediction_output[
                f"{model_name}_seed_{seed}_prediction"
            ] = probabilities_to_predictions(
                probabilities
            )

            print(
                f"model={model_name}, "
                f"seed={seed}, "
                f"accuracy={overall.accuracy:.6f}, "
                f"macro_f1={overall.macro_f1:.6f}, "
                f"twin_macro_f1={twin.macro_f1:.6f}, "
                f"no_twin_macro_f1={no_twin.macro_f1:.6f}"
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

        model_results[
            model_name
        ] = {
            "per_seed": per_seed,
            "summary": summary,
        }

        selection_scores[
            model_name
        ] = selection_runs

        overall_summary = summary[
            "overall"
        ]

        print(
            f"model={model_name}, "
            f"accuracy_mean="
            f"{overall_summary['accuracy']['mean']:.6f}, "
            f"accuracy_std="
            f"{overall_summary['accuracy']['std']:.6f}, "
            f"macro_f1_mean="
            f"{overall_summary['macro_f1']['mean']:.6f}, "
            f"macro_f1_std="
            f"{overall_summary['macro_f1']['std']:.6f}"
        )

        print()

    selection = build_stacking_selection(
        train=train,
        validation=validation,
        scores=selection_scores,
    )

    save_stacking_selection(
        selection
    )

    output_metrics = {
        "models": model_results,
        "views": list(
            STACKING.primary_views
        ),
        "selected_embedding_model": (
            selected_embedding_model
        ),
        "seeds": list(
            EXPERIMENT.seeds
        ),
        "threshold": 0.5,
    }

    PATHS.validation_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        PATHS.validation_results_dir
        / "stacking_predictions.csv"
    )

    metrics_path = (
        PATHS.validation_results_dir
        / "stacking_metrics.json"
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
        "Stacking ranking"
    )

    for item in selection[
        "ranking"
    ]:
        print(
            f"rank={item['rank']}, "
            f"model={item['model']}, "
            f"macro_f1_mean="
            f"{item['macro_f1']['mean']:.6f}, "
            f"macro_f1_std="
            f"{item['macro_f1']['std']:.6f}, "
            f"accuracy_mean="
            f"{item['accuracy']['mean']:.6f}, "
            f"accuracy_std="
            f"{item['accuracy']['std']:.6f}"
        )

    print(
        f"selected_model="
        f"{selection['selected_model']}"
    )

    print(
        f"selection_metric="
        f"{selection['primary_metric']}"
    )

    print(
        f"selection_aggregation="
        f"{selection['aggregation']}"
    )

    print(
        f"selection_seeds="
        f"{tuple(selection['selection_seeds'])}"
    )

    print(
        f"Saved predictions to "
        f"{predictions_path}"
    )

    print(
        f"Saved metrics to "
        f"{metrics_path}"
    )

    print(
        f"Saved selection to "
        f"{PATHS.stacking_selection_file}"
    )


if __name__ == "__main__":
    main()