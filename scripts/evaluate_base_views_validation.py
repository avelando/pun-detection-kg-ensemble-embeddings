import json

import pandas as pd

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
from pun_detection.evaluation import (
    compute_binary_metrics,
    probabilities_to_predictions,
    summarize_binary_metrics,
)
from pun_detection.fingerprints import (
    array_fingerprint,
)
from pun_detection.pairs import (
    twin_in_reference_mask,
)


def main():
    splits = load_development_splits()

    train = splits.train
    validation = splits.validation

    y_validation = validation[
        DATA.label_column
    ].astype(int).to_numpy()

    twin_mask = twin_in_reference_mask(
        validation,
        train,
    )

    no_twin_mask = ~twin_mask

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

    metric_runs = {
        view_name: {
            "overall": [],
            "twin_in_train": [],
            "no_twin_in_train": [],
        }
        for view_name in BASE_VIEW_NAMES
    }

    per_seed = {}
    selected_embedding_models = set()

    for seed in EXPERIMENT.seeds:
        matrices = load_base_view_cache(
            train=train,
            validation=validation,
            seed=seed,
        )

        selected_embedding_models.add(
            matrices.selected_embedding_model
        )

        if matrices.columns != BASE_VIEW_NAMES:
            raise ValueError(
                "Unexpected base view column order"
            )

        seed_results = {}

        for column_index, view_name in enumerate(
            BASE_VIEW_NAMES
        ):
            probabilities = matrices.validation[
                :,
                column_index,
            ]

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
                view_name
            ][
                "overall"
            ].append(
                overall
            )

            metric_runs[
                view_name
            ][
                "twin_in_train"
            ].append(
                twin
            )

            metric_runs[
                view_name
            ][
                "no_twin_in_train"
            ].append(
                no_twin
            )

            seed_results[
                view_name
            ] = {
                "overall": overall.as_dict(),
                "twin_in_train": twin.as_dict(),
                "no_twin_in_train": no_twin.as_dict(),
                "probability_fingerprint": array_fingerprint(
                    probabilities
                ),
            }

            prediction_output[
                f"{view_name}_seed_{seed}_probability"
            ] = probabilities

            prediction_output[
                f"{view_name}_seed_{seed}_prediction"
            ] = probabilities_to_predictions(
                probabilities
            )

            print(
                f"view={view_name}, "
                f"seed={seed}, "
                f"accuracy={overall.accuracy:.6f}, "
                f"macro_f1={overall.macro_f1:.6f}, "
                f"twin_macro_f1={twin.macro_f1:.6f}, "
                f"no_twin_macro_f1={no_twin.macro_f1:.6f}"
            )

        per_seed[
            str(seed)
        ] = {
            "validation_matrix_fingerprint": array_fingerprint(
                matrices.validation
            ),
            "views": seed_results,
        }

    if len(
        selected_embedding_models
    ) != 1:
        raise ValueError(
            "Base view caches use different selected embedding models"
        )

    selected_embedding_model = next(
        iter(
            selected_embedding_models
        )
    )

    summary = {}

    for view_name in BASE_VIEW_NAMES:
        summary[
            view_name
        ] = {
            "overall": summarize_binary_metrics(
                metric_runs[
                    view_name
                ][
                    "overall"
                ]
            ),
            "twin_in_train": summarize_binary_metrics(
                metric_runs[
                    view_name
                ][
                    "twin_in_train"
                ]
            ),
            "no_twin_in_train": summarize_binary_metrics(
                metric_runs[
                    view_name
                ][
                    "no_twin_in_train"
                ]
            ),
        }

    output_metrics = {
        "views": list(
            BASE_VIEW_NAMES
        ),
        "selected_embedding_model": selected_embedding_model,
        "seeds": list(
            EXPERIMENT.seeds
        ),
        "per_seed": per_seed,
        "summary": summary,
    }

    PATHS.validation_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        PATHS.validation_results_dir
        / "base_views_predictions.csv"
    )

    metrics_path = (
        PATHS.validation_results_dir
        / "base_views_metrics.json"
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

    print()

    print(
        f"selected_embedding_model="
        f"{selected_embedding_model}"
    )

    for view_name in BASE_VIEW_NAMES:
        overall_summary = summary[
            view_name
        ][
            "overall"
        ]

        print(
            f"view={view_name}, "
            f"accuracy_mean="
            f"{overall_summary['accuracy']['mean']:.6f}, "
            f"accuracy_std="
            f"{overall_summary['accuracy']['std']:.6f}, "
            f"macro_f1_mean="
            f"{overall_summary['macro_f1']['mean']:.6f}, "
            f"macro_f1_std="
            f"{overall_summary['macro_f1']['std']:.6f}"
        )

    print(
        f"Saved predictions to {predictions_path}"
    )

    print(
        f"Saved metrics to {metrics_path}"
    )


if __name__ == "__main__":
    main()