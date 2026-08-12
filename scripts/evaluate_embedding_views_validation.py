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
)
from pun_detection.models.embedding_model import (
    fit_embedding_view_model,
    predict_embedding_view_probabilities,
)
from pun_detection.pairs import (
    twin_in_reference_mask,
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
    ranking_data = []

    for model_name in EMBEDDING_MODELS:
        model = fit_embedding_view_model(
            train=train,
            model_name=model_name,
            seed=EXPERIMENT.primary_seed,
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

        config = EMBEDDING_MODELS[
            model_name
        ]

        results[model_name] = {
            "model_id": config.model_id,
            "model_revision": (
                config.revision
            ),
            "classifier": {
                "type": (
                    "logistic_regression"
                ),
                "C": (
                    BASE_MODELS.logistic_c
                ),
                "solver": (
                    BASE_MODELS.logistic_solver
                ),
                "max_iter": (
                    BASE_MODELS.logistic_max_iter
                ),
                "iterations": (
                    model.iterations
                ),
            },
            "overall": (
                overall.as_dict()
            ),
            "twin_in_train": (
                twin.as_dict()
            ),
            "no_twin_in_train": (
                no_twin.as_dict()
            ),
        }

        ranking_data.append(
            (
                model_name,
                overall.macro_f1,
                overall.accuracy,
            )
        )

        predictions_output[
            f"{model_name}_probability"
        ] = probabilities

        predictions_output[
            f"{model_name}_prediction"
        ] = probabilities_to_predictions(
            probabilities
        )

        print(
            f"{model_name}: "
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

    ranking_data.sort(
        key=lambda item: (
            -item[1],
            -item[2],
            item[0],
        )
    )

    ranking = []

    for rank, (
        model_name,
        macro_f1,
        accuracy,
    ) in enumerate(
        ranking_data,
        start=1,
    ):
        ranking.append(
            {
                "rank": rank,
                "model": model_name,
                "macro_f1": macro_f1,
                "accuracy": accuracy,
            }
        )

    selected_model = ranking[
        0
    ]["model"]

    selection = {
        "selection_split": "validation",
        "primary_metric": (
            EXPERIMENT.primary_metric
        ),
        "selected_model": (
            selected_model
        ),
        "classifier": {
            "type": (
                "logistic_regression"
            ),
            "C": (
                BASE_MODELS.logistic_c
            ),
            "solver": (
                BASE_MODELS.logistic_solver
            ),
            "max_iter": (
                BASE_MODELS.logistic_max_iter
            ),
        },
        "ranking": ranking,
    }

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

    selection_path = (
        PATHS.validation_results_dir
        / "embedding_selection.json"
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

    with selection_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selection,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    print()

    for item in ranking:
        print(
            f"rank={item['rank']}, "
            f"model={item['model']}, "
            f"macro_f1="
            f"{item['macro_f1']:.6f}, "
            f"accuracy="
            f"{item['accuracy']:.6f}"
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
        f"Saved predictions to "
        f"{predictions_path}"
    )

    print(
        f"Saved metrics to "
        f"{metrics_path}"
    )

    print(
        f"Saved selection to "
        f"{selection_path}"
    )


if __name__ == "__main__":
    main()