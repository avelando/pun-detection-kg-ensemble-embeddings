import json

from pun_detection.config import (
    EXPERIMENT,
    PATHS,
)
from pun_detection.data import (
    load_development_splits,
)
from pun_detection.stacking_ablation import (
    STACKING_ABLATIONS,
    validate_stacking_ablations,
)
from pun_detection.stacking_selection import (
    load_stacking_selection,
)


def main():
    validate_stacking_ablations()

    splits = load_development_splits()

    selection = load_stacking_selection(
        train=splits.train,
        validation=splits.validation,
    )

    metrics_path = (
        PATHS.validation_results_dir
        / "stacking_ablation_metrics.json"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing stacking ablation metrics: "
            f"{metrics_path}"
        )

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(
            file
        )

    if metrics.get(
        "analysis_type"
    ) != "stacking_ablation":
        raise ValueError(
            "Invalid stacking ablation analysis type"
        )

    if metrics.get(
        "selection_role"
    ) != "diagnostic_only":
        raise ValueError(
            "Stacking ablation must remain diagnostic"
        )

    if metrics.get(
        "selected_stacking_model"
    ) != selection[
        "selected_model"
    ]:
        raise ValueError(
            "Stacking ablation uses an unexpected "
            "meta-classifier"
        )

    if metrics.get(
        "selected_embedding_model"
    ) != selection[
        "base_views"
    ][
        "selected_embedding_model"
    ]:
        raise ValueError(
            "Stacking ablation uses an unexpected "
            "embedding model"
        )

    if metrics.get(
        "seeds"
    ) != list(
        EXPERIMENT.seeds
    ):
        raise ValueError(
            "Stacking ablation seeds mismatch"
        )

    configurations = metrics.get(
        "configurations"
    )

    if set(
        configurations
    ) != set(
        STACKING_ABLATIONS
    ):
        raise ValueError(
            "Stacking ablation configurations mismatch"
        )

    for configuration_name, view_names in (
        STACKING_ABLATIONS.items()
    ):
        if configurations[
            configuration_name
        ][
            "views"
        ] != list(
            view_names
        ):
            raise ValueError(
                f"Invalid views for "
                f"{configuration_name}"
            )

    selected_ranking = next(
        item
        for item in selection[
            "ranking"
        ]
        if item[
            "model"
        ] == selection[
            "selected_model"
        ]
    )

    selected_runs = {
        int(
            run["seed"]
        ): run
        for run in selected_ranking[
            "runs"
        ]
    }

    full_runs = configurations[
        "all_views"
    ][
        "per_seed"
    ]

    for seed in EXPERIMENT.seeds:
        expected = selected_runs[
            seed
        ]

        actual = full_runs[
            str(seed)
        ][
            "overall"
        ]

        if (
            actual["macro_f1"]
            != expected["macro_f1"]
        ):
            raise ValueError(
                f"All-views Macro-F1 mismatch "
                f"for seed {seed}"
            )

        if (
            actual["accuracy"]
            != expected["accuracy"]
        ):
            raise ValueError(
                f"All-views accuracy mismatch "
                f"for seed {seed}"
            )

    print(
        f"selected_stacking_model="
        f"{selection['selected_model']}"
    )

    print(
        f"configurations="
        f"{len(STACKING_ABLATIONS)}"
    )

    print(
        "All-views results reproduce "
        "the selected stacking experiment"
    )

    print(
        "Stacking ablation is valid"
    )


if __name__ == "__main__":
    main()