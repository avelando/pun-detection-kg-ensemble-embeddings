import json

import pandas as pd

from pun_detection.config import (
    BASE_MODELS,
    DATA,
    EMBEDDING_MODELS,
    EXPERIMENT,
    PATHS,
)
from pun_detection.data import (
    load_train_split,
    load_validation_split,
)
from pun_detection.evaluation import (
    compute_binary_metrics,
    probabilities_to_predictions,
    summarize_binary_metrics,
)
from pun_detection.models.embedding_model import (
    fit_embedding_view_model,
    predict_embedding_view_probabilities,
)
from pun_detection.pairs import (
    twin_in_reference_mask,
)
from pun_detection.selection import (
    build_embedding_selection,
    save_embedding_selection,
)


def main():
    train = load_train_split()
    validation = load_validation_split()

    y_validation = validation[
        DATA.label_column
    ].astype(int).to_numpy()

    twin_mask = twin_in_reference_mask(
        validation,
        train,
    )

    no_twin_mask = ~twin_mask

    predictions_output = pd.DataFrame(
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

    results = {}
    selection_scores = {}

    for model_name in EMBEDDING_MODELS:
        config = EMBEDDING_MODELS[
            model_name
        ]

        model_runs = []
        selection_runs = []

        overall_runs = []
        twin_runs = []
        no_twin_runs = []

        for seed in EXPERIMENT.seeds:
            model = fit_embedding_view_model(
                train=train,
                model_name=model_name,
                seed=seed,
            )

            probabilities = (
                predict_embedding_view_probabilities(
                    model=model,
                    dataframe=validation,
                    split_name="validation",
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

            model_runs.append(
                {
                    "seed": seed,
                    "iterations": model.iterations,
                    "overall": overall.as_dict(),
                    "twin_in_train": twin.as_dict(),
                    "no_twin_in_train": no_twin.as_dict(),
                }
            )

            selection_runs.append(
                {
                    "seed": seed,
                    "macro_f1": overall.macro_f1,
                    "accuracy": overall.accuracy,
                }
            )

            predictions_output[
                f"{model_name}_seed_"
                f"{seed}_probability"
            ] = probabilities

            predictions_output[
                f"{model_name}_seed_"
                f"{seed}_prediction"
            ] = probabilities_to_predictions(
                probabilities
            )

            print(
                f"{model_name}: "
                f"seed={seed}, "
                f"iterations={model.iterations}, "
                f"accuracy="
                f"{overall.accuracy:.6f}, "
                f"macro_f1="
                f"{overall.macro_f1:.6f}, "
                f"twin_macro_f1="
                f"{twin.macro_f1:.6f}, "
                f"no_twin_macro_f1="
                f"{no_twin.macro_f1:.6f}"
            )

        results[
            model_name
        ] = {
            "model_id": config.model_id,
            "model_revision": config.revision,
            "classifier": {
                "type": "logistic_regression",
                "C": BASE_MODELS.logistic_c,
                "solver": BASE_MODELS.logistic_solver,
                "max_iter": BASE_MODELS.logistic_max_iter,
            },
            "runs": model_runs,
            "summary": {
                "overall": summarize_binary_metrics(
                    overall_runs
                ),
                "twin_in_train": summarize_binary_metrics(
                    twin_runs
                ),
                "no_twin_in_train": summarize_binary_metrics(
                    no_twin_runs
                ),
            },
        }

        selection_scores[
            model_name
        ] = selection_runs

        overall_summary = results[
            model_name
        ][
            "summary"
        ][
            "overall"
        ]

        print(
            f"{model_name}: "
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

    selection = build_embedding_selection(
        train=train,
        validation=validation,
        scores=selection_scores,
    )

    ranking = selection[
        "ranking"
    ]

    selected_model = selection[
        "selected_model"
    ]

    PATHS.validation_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        PATHS.validation_results_dir
        / "embedding_views_predictions.csv"
    )

    metrics_path = (
        PATHS.validation_results_dir
        / "embedding_views_metrics.json"
    )

    predictions_output.to_csv(
        predictions_path,
        index=False,
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    save_embedding_selection(
        selection
    )

    print()

    for item in ranking:
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

    print()

    print(
        f"selected_model={selected_model}"
    )

    print(
        f"selection_metric="
        f"{EXPERIMENT.primary_metric}"
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
        f"{PATHS.embedding_selection_file}"
    )


if __name__ == "__main__":
    main()