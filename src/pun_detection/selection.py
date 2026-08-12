import json
import math

import pandas as pd

from pun_detection.config import (
    BASE_MODELS,
    EMBEDDING_MODELS,
    EXPERIMENT,
    PATHS,
)
from pun_detection.fingerprints import (
    supervised_dataset_fingerprint,
)
from pun_detection.evaluation import (
    summarize_metric_values,
)


def embedding_classifier_config() -> dict:
    return {
        "type": "logistic_regression",
        "C": BASE_MODELS.logistic_c,
        "solver": BASE_MODELS.logistic_solver,
        "max_iter": BASE_MODELS.logistic_max_iter,
    }


def rank_embedding_models(
    scores: dict[str, list[dict]],
) -> list[dict]:
    expected_models = set(
        EMBEDDING_MODELS
    )

    actual_models = set(
        scores
    )

    if actual_models != expected_models:
        missing_models = sorted(
            expected_models.difference(
                actual_models
            )
        )

        unexpected_models = sorted(
            actual_models.difference(
                expected_models
            )
        )

        raise ValueError(
            "Embedding selection candidates mismatch: "
            f"missing={missing_models}, "
            f"unexpected={unexpected_models}"
        )

    expected_seeds = tuple(
        EXPERIMENT.seeds
    )

    ranking_data = []

    for model_name, model_runs in scores.items():
        if not isinstance(
            model_runs,
            list,
        ):
            raise ValueError(
                f"Embedding runs must be a list for "
                f"{model_name}"
            )

        normalized_runs = []

        seen_seeds = set()

        for run in model_runs:
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
                    f"Invalid embedding run for "
                    f"{model_name}"
                ) from error

            if seed in seen_seeds:
                raise ValueError(
                    f"Duplicated seed {seed} for "
                    f"{model_name}"
                )

            seen_seeds.add(
                seed
            )

            if not math.isfinite(
                macro_f1
            ):
                raise ValueError(
                    f"Non-finite Macro-F1 for "
                    f"{model_name}, seed={seed}"
                )

            if not math.isfinite(
                accuracy
            ):
                raise ValueError(
                    f"Non-finite accuracy for "
                    f"{model_name}, seed={seed}"
                )

            if not 0.0 <= macro_f1 <= 1.0:
                raise ValueError(
                    f"Invalid Macro-F1 for "
                    f"{model_name}, seed={seed}: "
                    f"{macro_f1}"
                )

            if not 0.0 <= accuracy <= 1.0:
                raise ValueError(
                    f"Invalid accuracy for "
                    f"{model_name}, seed={seed}: "
                    f"{accuracy}"
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

        if actual_seeds != tuple(
            sorted(
                expected_seeds
            )
        ):
            raise ValueError(
                f"Embedding selection seeds mismatch "
                f"for {model_name}: "
                f"{actual_seeds}"
            )

        macro_f1_summary = (
            summarize_metric_values(
                [
                    item["macro_f1"]
                    for item in normalized_runs
                ]
            )
        )

        accuracy_summary = (
            summarize_metric_values(
                [
                    item["accuracy"]
                    for item in normalized_runs
                ]
            )
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


def build_embedding_selection(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    scores: dict[str, list[dict]],
) -> dict:
    if EXPERIMENT.primary_metric != "macro_f1":
        raise ValueError(
            "Embedding selection currently requires "
            "primary_metric='macro_f1'"
        )

    ranking = rank_embedding_models(
        scores
    )

    return {
        "selection_type": "embedding_model",
        "selection_split": "validation",
        "primary_metric": EXPERIMENT.primary_metric,
        "aggregation": "mean",
        "selection_seeds": list(
            EXPERIMENT.seeds
        ),
        "tie_breaker": "model_name",
        "selected_model": ranking[0]["model"],
        "candidate_models": sorted(
            EMBEDDING_MODELS
        ),
        "supervised_dataset_fingerprints": {
            "train": supervised_dataset_fingerprint(
                train
            ),
            "validation": supervised_dataset_fingerprint(
                validation
            ),
        },
        "classifier": embedding_classifier_config(),
        "ranking": ranking,
    }


def save_embedding_selection(
    selection: dict,
) -> None:
    PATHS.embedding_selection_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PATHS.embedding_selection_file.open(
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


def validate_embedding_selection(
    selection: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    if selection.get(
        "selection_type"
    ) != "embedding_model":
        raise ValueError(
            "Invalid embedding selection type"
        )

    if selection.get(
        "selection_split"
    ) != "validation":
        raise ValueError(
            "Embedding selection must use validation"
        )

    if selection.get(
        "primary_metric"
    ) != EXPERIMENT.primary_metric:
        raise ValueError(
            "Embedding selection metric mismatch"
        )

    if EXPERIMENT.primary_metric != "macro_f1":
        raise ValueError(
            "Embedding selection currently requires "
            "primary_metric='macro_f1'"
        )

    if selection.get(
        "aggregation"
    ) != "mean":
        raise ValueError(
            "Embedding selection aggregation mismatch"
        )

    if selection.get(
        "selection_seeds"
    ) != list(
        EXPERIMENT.seeds
    ):
        raise ValueError(
            "Embedding selection seeds mismatch"
        )

    if selection.get(
        "tie_breaker"
    ) != "model_name":
        raise ValueError(
            "Embedding selection tie breaker mismatch"
        )

    expected_candidates = sorted(
        EMBEDDING_MODELS
    )

    if selection.get(
        "candidate_models"
    ) != expected_candidates:
        raise ValueError(
            "Embedding candidate models mismatch"
        )

    expected_classifier = (
        embedding_classifier_config()
    )

    if selection.get(
        "classifier"
    ) != expected_classifier:
        raise ValueError(
            "Embedding classifier configuration mismatch"
        )

    expected_fingerprints = {
        "train": supervised_dataset_fingerprint(
            train
        ),
        "validation": supervised_dataset_fingerprint(
            validation
        ),
    }

    if selection.get(
        "supervised_dataset_fingerprints"
    ) != expected_fingerprints:
        raise ValueError(
            "Embedding selection dataset fingerprint mismatch"
        )

    ranking = selection.get(
        "ranking"
    )

    if not isinstance(
        ranking,
        list,
    ):
        raise ValueError(
            "Embedding ranking must be a list"
        )

    if len(
        ranking
    ) != len(
        expected_candidates
    ):
        raise ValueError(
            "Embedding ranking size mismatch"
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
                "Invalid embedding ranking item"
            )

        if item.get(
            "rank"
        ) != expected_rank:
            raise ValueError(
                "Embedding ranking positions are invalid"
            )

        model_name = item.get(
            "model"
        )

        if model_name in scores:
            raise ValueError(
                f"Duplicated embedding ranking model: "
                f"{model_name}"
            )

        if model_name not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown embedding ranking model: "
                f"{model_name}"
            )

        try:
            runs = item.get(
                "runs"
            )

            if not isinstance(
                runs,
                list,
            ):
                raise ValueError(
                    f"Missing embedding runs for "
                    f"{model_name}"
                )

            scores[
                model_name
            ] = runs
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"Invalid ranking metrics for {model_name}"
            ) from error

    expected_ranking = (
        rank_embedding_models(
            scores
        )
    )

    if ranking != expected_ranking:
        raise ValueError(
            "Embedding ranking is inconsistent "
            "with the selection rule"
        )

    selected_model = selection.get(
        "selected_model"
    )

    if selected_model != ranking[0]["model"]:
        raise ValueError(
            "Selected embedding does not match "
            "the top-ranked model"
        )

    return selected_model


def load_embedding_selection(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict:
    if not PATHS.embedding_selection_file.is_file():
        raise FileNotFoundError(
            "Missing embedding selection result: "
            f"{PATHS.embedding_selection_file}"
        )

    try:
        with PATHS.embedding_selection_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            selection = json.load(
                file
            )
    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid embedding selection JSON"
        ) from error

    validate_embedding_selection(
        selection=selection,
        train=train,
        validation=validation,
    )

    return selection


def get_selected_embedding_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    selection = load_embedding_selection(
        train=train,
        validation=validation,
    )

    return selection[
        "selected_model"
    ]