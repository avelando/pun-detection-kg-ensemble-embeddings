import json

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
    summarize_binary_metrics,
)
from pun_detection.models.reference_tfidf import (
    fit_reference_tfidf_model,
    predict_reference_tfidf_probabilities,
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

    overall_runs = []
    twin_runs = []
    no_twin_runs = []

    per_seed_metrics = {}

    for seed in EXPERIMENT.seeds:
        model = fit_reference_tfidf_model(
            train=train,
            seed=seed,
        )

        probabilities = (
            predict_reference_tfidf_probabilities(
                model,
                validation,
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

        overall_runs.append(overall)
        twin_runs.append(twin)
        no_twin_runs.append(no_twin)

        per_seed_metrics[str(seed)] = {
            "overall": overall.as_dict(),
            "twin_in_train": twin.as_dict(),
            "no_twin_in_train": (
                no_twin.as_dict()
            ),
        }

        prediction_output[
            f"probability_seed_{seed}"
        ] = probabilities

        prediction_output[
            f"prediction_seed_{seed}"
        ] = probabilities_to_predictions(
            probabilities
        )

        print(
            f"seed={seed}, "
            f"accuracy={overall.accuracy:.6f}, "
            f"macro_f1={overall.macro_f1:.6f}, "
            f"twin_macro_f1={twin.macro_f1:.6f}, "
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

    output_metrics = {
        "model": "reference_tfidf_ensemble",
        "seeds": list(
            EXPERIMENT.seeds
        ),
        "per_seed": per_seed_metrics,
        "summary": summary,
    }

    PATHS.validation_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        PATHS.validation_results_dir
        / "reference_tfidf_ensemble_predictions.csv"
    )

    metrics_path = (
        PATHS.validation_results_dir
        / "reference_tfidf_ensemble_metrics.json"
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
            indent=2,
            sort_keys=True,
        )

    overall_summary = summary[
        "overall"
    ]

    print(
        "reference_tfidf_ensemble: "
        f"accuracy="
        f"{overall_summary['accuracy']['mean']:.6f}"
        f"±"
        f"{overall_summary['accuracy']['std']:.6f}, "
        f"macro_f1="
        f"{overall_summary['macro_f1']['mean']:.6f}"
        f"±"
        f"{overall_summary['macro_f1']['std']:.6f}"
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