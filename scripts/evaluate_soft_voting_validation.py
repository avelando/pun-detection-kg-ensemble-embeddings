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
from pun_detection.models.soft_voting import (
    soft_voting_probabilities,
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

    overall_runs = []
    twin_runs = []
    no_twin_runs = []

    per_seed_metrics = {}
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

        probabilities = soft_voting_probabilities(
            matrices.validation
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

        probability_fingerprint = (
            array_fingerprint(
                probabilities
            )
        )

        validation_matrix_fingerprint = (
            array_fingerprint(
                matrices.validation
            )
        )

        per_seed_metrics[
            str(seed)
        ] = {
            "overall": overall.as_dict(),
            "twin_in_train": twin.as_dict(),
            "no_twin_in_train": (
                no_twin.as_dict()
            ),
            "validation_matrix_fingerprint": (
                validation_matrix_fingerprint
            ),
            "probability_fingerprint": (
                probability_fingerprint
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
            f"twin_macro_f1="
            f"{twin.macro_f1:.6f}, "
            f"no_twin_macro_f1="
            f"{no_twin.macro_f1:.6f}"
        )

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
        "model": "soft_voting",
        "aggregation": "arithmetic_mean",
        "threshold": 0.5,
        "views": list(
            BASE_VIEW_NAMES
        ),
        "selected_embedding_model": (
            selected_embedding_model
        ),
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
        / "soft_voting_predictions.csv"
    )

    metrics_path = (
        PATHS.validation_results_dir
        / "soft_voting_metrics.json"
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

    overall_summary = summary[
        "overall"
    ]

    print()

    print(
        f"selected_embedding_model="
        f"{selected_embedding_model}"
    )

    print(
        f"views="
        f"{','.join(BASE_VIEW_NAMES)}"
    )

    print(
        "soft_voting: "
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