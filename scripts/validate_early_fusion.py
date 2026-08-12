import json

from pun_detection.config import (
    EXPERIMENT,
    PATHS,
)
from pun_detection.data import (
    load_development_splits,
)
from pun_detection.early_fusion import (
    EARLY_FUSION_CONFIGURATIONS,
    validate_early_fusion_configurations,
)
from pun_detection.selection import (
    get_selected_embedding_model,
)


def main():
    validate_early_fusion_configurations()

    splits = load_development_splits()

    selected_embedding_model = (
        get_selected_embedding_model(
            train=splits.train,
            validation=splits.validation,
        )
    )

    metrics_path = (
        PATHS.validation_results_dir
        / "early_fusion_metrics.json"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing early fusion metrics: "
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
    ) != "early_fusion":
        raise ValueError(
            "Invalid early fusion analysis type"
        )

    if metrics.get(
        "selection_role"
    ) != "predefined_comparison":
        raise ValueError(
            "Early fusion must remain a "
            "predefined comparison"
        )

    if metrics.get(
        "selected_embedding_model"
    ) != selected_embedding_model:
        raise ValueError(
            "Early fusion uses an unexpected "
            "embedding model"
        )

    if metrics.get(
        "seeds"
    ) != list(
        EXPERIMENT.seeds
    ):
        raise ValueError(
            "Early fusion seeds mismatch"
        )

    configurations = metrics.get(
        "configurations"
    )

    if not isinstance(
        configurations,
        dict,
    ):
        raise ValueError(
            "Invalid early fusion configurations"
        )

    if set(
        configurations
    ) != set(
        EARLY_FUSION_CONFIGURATIONS
    ):
        raise ValueError(
            "Early fusion configurations mismatch"
        )

    for configuration_name, components in (
        EARLY_FUSION_CONFIGURATIONS.items()
    ):
        actual_components = configurations[
            configuration_name
        ][
            "components"
        ]

        if actual_components != list(
            components
        ):
            raise ValueError(
                f"Invalid early fusion components "
                f"for {configuration_name}"
            )

        per_seed = configurations[
            configuration_name
        ][
            "per_seed"
        ]

        if set(
            per_seed
        ) != {
            str(seed)
            for seed in EXPERIMENT.seeds
        }:
            raise ValueError(
                f"Early fusion seeds missing for "
                f"{configuration_name}"
            )

    print(
        f"selected_embedding_model="
        f"{selected_embedding_model}"
    )

    print(
        f"configurations="
        f"{len(EARLY_FUSION_CONFIGURATIONS)}"
    )

    print(
        "selection_role=predefined_comparison"
    )

    print(
        "Early fusion evaluation is valid"
    )


if __name__ == "__main__":
    main()