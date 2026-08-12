import json
import math
from dataclasses import asdict

import pandas as pd

from pun_detection.base_view_cache import (
    load_base_view_cache,
)
from pun_detection.base_views import (
    BASE_VIEW_NAMES,
)
from pun_detection.config import (
    BASE_MODELS,
    EXPERIMENT,
    PATHS,
    STACKING,
)
from pun_detection.evaluation import (
    summarize_metric_values,
)
from pun_detection.fingerprints import (
    array_fingerprint,
    supervised_dataset_fingerprint,
)


def stacking_configuration() -> dict:
    stacking_config = asdict(
        STACKING
    )

    stacking_config[
        "candidates"
    ] = list(
        STACKING.candidates
    )

    return {
        "base_models": asdict(
            BASE_MODELS
        ),
        "stacking": stacking_config,
    }


def rank_stacking_models(
    scores: dict[str, list[dict]],
) -> list[dict]:
    if set(
        scores
    ) != set(
        STACKING.candidates
    ):
        raise ValueError(
            "Stacking selection candidates mismatch"
        )

    expected_seeds = tuple(
        sorted(
            EXPERIMENT.seeds
        )
    )

    ranking_data = []

    for model_name, runs in scores.items():
        if not isinstance(
            runs,
            list,
        ):
            raise ValueError(
                f"Stacking runs must be a list for {model_name}"
            )

        normalized_runs = []
        seen_seeds = set()

        for run in runs:
            try:
                seed = int(
                    run["seed"]
                )

                macro_f1 = float(
                    run["macro_f1"]
                )

                accuracy = float(
                    run["accuracy"]
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    f"Invalid stacking run for {model_name}"
                ) from error

            if seed in seen_seeds:
                raise ValueError(
                    f"Duplicated seed {seed} for {model_name}"
                )

            seen_seeds.add(
                seed
            )

            for metric_name, metric_value in (
                ("macro_f1", macro_f1),
                ("accuracy", accuracy),
            ):
                if not math.isfinite(
                    metric_value
                ):
                    raise ValueError(
                        f"Non-finite {metric_name} for "
                        f"{model_name}, seed={seed}"
                    )

                if not 0.0 <= metric_value <= 1.0:
                    raise ValueError(
                        f"Invalid {metric_name} for "
                        f"{model_name}, seed={seed}"
                    )

            normalized_runs.append(
                {
                    "seed": seed,
                    "macro_f1": macro_f1,
                    "accuracy": accuracy,
                }
            )

        normalized_runs.sort(
            key=lambda item: item["seed"]
        )

        actual_seeds = tuple(
            item["seed"]
            for item in normalized_runs
        )

        if actual_seeds != expected_seeds:
            raise ValueError(
                f"Stacking selection seeds mismatch "
                f"for {model_name}"
            )

        macro_f1_summary = summarize_metric_values(
            [
                item["macro_f1"]
                for item in normalized_runs
            ]
        )

        accuracy_summary = summarize_metric_values(
            [
                item["accuracy"]
                for item in normalized_runs
            ]
        )

        ranking_data.append(
            (
                model_name,
                macro_f1_summary,
                accuracy_summary,
                normalized_runs,
            )
        )

    ranking_data.sort(
        key=lambda item: (
            -item[1]["mean"],
            item[0],
        )
    )

    return [
        {
            "rank": rank,
            "model": model_name,
            "macro_f1": macro_f1_summary,
            "accuracy": accuracy_summary,
            "runs": normalized_runs,
        }
        for rank, (
            model_name,
            macro_f1_summary,
            accuracy_summary,
            normalized_runs,
        ) in enumerate(
            ranking_data,
            start=1,
        )
    ]


def base_view_selection_context(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict:
    fingerprints = {}
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

        fingerprints[
            str(seed)
        ] = {
            "train_oof": array_fingerprint(
                matrices.train_oof
            ),
            "validation": array_fingerprint(
                matrices.validation
            ),
        }

    if len(
        selected_embedding_models
    ) != 1:
        raise ValueError(
            "Base view caches use different selected "
            "embedding models"
        )

    return {
        "columns": list(
            BASE_VIEW_NAMES
        ),
        "selected_embedding_model": next(
            iter(
                selected_embedding_models
            )
        ),
        "fingerprints": fingerprints,
    }


def build_stacking_selection(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    scores: dict[str, list[dict]],
) -> dict:
    if EXPERIMENT.primary_metric != "macro_f1":
        raise ValueError(
            "Stacking selection currently requires "
            "primary_metric='macro_f1'"
        )

    ranking = rank_stacking_models(
        scores
    )

    return {
        "selection_type": "stacking_meta_classifier",
        "selection_split": "validation",
        "primary_metric": EXPERIMENT.primary_metric,
        "aggregation": "mean",
        "selection_seeds": list(
            EXPERIMENT.seeds
        ),
        "tie_breaker": "model_name",
        "threshold": 0.5,
        "selected_model": ranking[0]["model"],
        "candidate_models": list(
            STACKING.candidates
        ),
        "configuration": stacking_configuration(),
        "base_views": base_view_selection_context(
            train=train,
            validation=validation,
        ),
        "supervised_dataset_fingerprints": {
            "train": supervised_dataset_fingerprint(
                train
            ),
            "validation": supervised_dataset_fingerprint(
                validation
            ),
        },
        "ranking": ranking,
    }


def save_stacking_selection(
    selection: dict,
) -> None:
    PATHS.stacking_selection_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PATHS.stacking_selection_file.open(
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


def validate_stacking_selection(
    selection: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    expected_fields = {
        "selection_type": "stacking_meta_classifier",
        "selection_split": "validation",
        "primary_metric": EXPERIMENT.primary_metric,
        "aggregation": "mean",
        "selection_seeds": list(
            EXPERIMENT.seeds
        ),
        "tie_breaker": "model_name",
        "threshold": 0.5,
        "candidate_models": list(
            STACKING.candidates
        ),
        "configuration": stacking_configuration(),
        "base_views": base_view_selection_context(
            train=train,
            validation=validation,
        ),
        "supervised_dataset_fingerprints": {
            "train": supervised_dataset_fingerprint(
                train
            ),
            "validation": supervised_dataset_fingerprint(
                validation
            ),
        },
    }

    for key, expected_value in expected_fields.items():
        if selection.get(
            key
        ) != expected_value:
            raise ValueError(
                f"Stacking selection mismatch for {key}"
            )

    ranking = selection.get(
        "ranking"
    )

    if not isinstance(
        ranking,
        list,
    ):
        raise ValueError(
            "Stacking ranking must be a list"
        )

    scores = {}

    for expected_rank, item in enumerate(
        ranking,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "Invalid stacking ranking item"
            )

        if item.get(
            "rank"
        ) != expected_rank:
            raise ValueError(
                "Stacking ranking positions are invalid"
            )

        model_name = item.get(
            "model"
        )

        if model_name in scores:
            raise ValueError(
                f"Duplicated stacking ranking model: "
                f"{model_name}"
            )

        if model_name not in STACKING.candidates:
            raise ValueError(
                f"Unknown stacking ranking model: "
                f"{model_name}"
            )

        runs = item.get(
            "runs"
        )

        if not isinstance(
            runs,
            list,
        ):
            raise ValueError(
                f"Missing stacking runs for {model_name}"
            )

        scores[
            model_name
        ] = runs

    expected_ranking = rank_stacking_models(
        scores
    )

    if ranking != expected_ranking:
        raise ValueError(
            "Stacking ranking is inconsistent "
            "with the selection rule"
        )

    selected_model = selection.get(
        "selected_model"
    )

    if selected_model != ranking[0]["model"]:
        raise ValueError(
            "Selected stacking model does not match "
            "the top-ranked model"
        )

    return selected_model


def load_stacking_selection(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict:
    if not PATHS.stacking_selection_file.is_file():
        raise FileNotFoundError(
            "Missing stacking selection result: "
            f"{PATHS.stacking_selection_file}"
        )

    try:
        with PATHS.stacking_selection_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            selection = json.load(
                file
            )
    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid stacking selection JSON"
        ) from error

    validate_stacking_selection(
        selection=selection,
        train=train,
        validation=validation,
    )

    return selection


def get_selected_stacking_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    selection = load_stacking_selection(
        train=train,
        validation=validation,
    )

    return selection[
        "selected_model"
    ]