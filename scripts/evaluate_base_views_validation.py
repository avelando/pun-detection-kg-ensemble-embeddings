import json

import numpy as np
import pandas as pd

from pun_detection.config import (
    DATA,
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
from pun_detection.models.graph_model import (
    fit_graph_view_model,
    predict_graph_view_probabilities,
)
from pun_detection.models.tfidf_model import (
    fit_tfidf_view_model,
    predict_tfidf_view_probabilities,
)
from pun_detection.pairs import (
    twin_in_reference_mask,
)


def evaluate_subset(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
):
    return compute_binary_metrics(
        y_true[mask],
        probabilities[mask],
    )


def print_metrics(
    model_name: str,
    subset_name: str,
    metrics,
) -> None:
    print(
        f"{model_name}/{subset_name}: "
        f"samples={metrics.samples}, "
        f"accuracy={metrics.accuracy:.6f}, "
        f"macro_precision="
        f"{metrics.macro_precision:.6f}, "
        f"macro_recall="
        f"{metrics.macro_recall:.6f}, "
        f"macro_f1="
        f"{metrics.macro_f1:.6f}"
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

    print(
        f"validation_rows={len(validation)}, "
        f"twin_in_train={int(twin_mask.sum())}, "
        f"no_twin_in_train="
        f"{int(no_twin_mask.sum())}"
    )

    tfidf_model = fit_tfidf_view_model(
        train=train,
        seed=EXPERIMENT.primary_seed,
    )

    probabilities = {
        "tfidf": (
            predict_tfidf_view_probabilities(
                tfidf_model,
                validation,
            )
        )
    }

    graph_model = fit_graph_view_model(
        train=train,
        seed=EXPERIMENT.primary_seed,
    )

    probabilities.update(
        predict_graph_view_probabilities(
            graph_model,
            validation,
        )
    )

    metrics = {}

    for model_name, model_probabilities in (
        probabilities.items()
    ):
        overall = compute_binary_metrics(
            y_validation,
            model_probabilities,
        )

        twin = evaluate_subset(
            y_validation,
            model_probabilities,
            twin_mask,
        )

        no_twin = evaluate_subset(
            y_validation,
            model_probabilities,
            no_twin_mask,
        )

        metrics[model_name] = {
            "overall": overall.as_dict(),
            "twin_in_train": twin.as_dict(),
            "no_twin_in_train": (
                no_twin.as_dict()
            ),
        }

        print_metrics(
            model_name,
            "overall",
            overall,
        )

        print_metrics(
            model_name,
            "twin_in_train",
            twin,
        )

        print_metrics(
            model_name,
            "no_twin_in_train",
            no_twin,
        )

    output = pd.DataFrame(
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

    for model_name, model_probabilities in (
        probabilities.items()
    ):
        output[
            f"{model_name}_probability"
        ] = model_probabilities

        output[
            f"{model_name}_prediction"
        ] = probabilities_to_predictions(
            model_probabilities
        )

    PATHS.validation_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        PATHS.validation_results_dir
        / (
            "base_views_"
            f"seed_{EXPERIMENT.primary_seed}_"
            "predictions.csv"
        )
    )

    metrics_path = (
        PATHS.validation_results_dir
        / (
            "base_views_"
            f"seed_{EXPERIMENT.primary_seed}_"
            "metrics.json"
        )
    )

    output.to_csv(
        predictions_path,
        index=False,
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            sort_keys=True,
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