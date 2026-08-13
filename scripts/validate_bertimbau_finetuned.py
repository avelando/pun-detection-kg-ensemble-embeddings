import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from pun_detection.config import (
    DATA,
    EXPERIMENT,
    FINE_TUNING,
    PATHS,
)
from pun_detection.data import load_development_splits
from pun_detection.evaluation import (
    compute_binary_metrics,
    probabilities_to_predictions,
    summarize_binary_metrics,
)
from pun_detection.fine_tuning import checkpoint_sha256
from pun_detection.fingerprints import (
    array_fingerprint,
    supervised_dataset_fingerprint,
)
from pun_detection.pairs import twin_in_reference_mask


def assert_metric_equal(
    actual: float,
    expected: float,
    name: str,
) -> None:
    if not np.isclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{name} mismatch: {actual} != {expected}"
        )


def validate_metric_block(
    actual,
    stored: dict,
    prefix: str,
) -> None:
    expected = actual.as_dict()

    if set(stored) != set(expected):
        raise ValueError(
            f"{prefix} metric fields mismatch"
        )

    for field, expected_value in expected.items():
        stored_value = stored[field]

        if field == "samples":
            if int(stored_value) != int(expected_value):
                raise ValueError(
                    f"{prefix} samples mismatch"
                )
            continue

        assert_metric_equal(
            float(stored_value),
            float(expected_value),
            f"{prefix} {field}",
        )


def validate_summary_block(
    actual: dict,
    stored: dict,
    prefix: str,
) -> None:
    if set(
        actual
    ) != set(
        stored
    ):
        raise ValueError(
            f"{prefix} summary metrics mismatch"
        )

    for metric_name in actual:
        actual_metric = actual[
            metric_name
        ]

        stored_metric = stored[
            metric_name
        ]

        if set(
            actual_metric
        ) != set(
            stored_metric
        ):
            raise ValueError(
                f"{prefix} {metric_name} "
                "summary fields mismatch"
            )

        for field in actual_metric:
            assert_metric_equal(
                float(
                    stored_metric[
                        field
                    ]
                ),
                float(
                    actual_metric[
                        field
                    ]
                ),
                (
                    f"{prefix} "
                    f"{metric_name} "
                    f"{field}"
                ),
            )


def main():
    splits = load_development_splits()
    train = splits.train
    validation = splits.validation

    metrics_path = (
        PATHS.validation_results_dir
        / "bertimbau_finetuned_metrics.json"
    )
    predictions_path = (
        PATHS.validation_results_dir
        / "bertimbau_finetuned_predictions.csv"
    )
    predictions_npz_path = (
        PATHS.validation_results_dir
        / "bertimbau_finetuned_predictions.npz"
    )

    for path in (
        metrics_path,
        predictions_path,
        predictions_npz_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing BERTimbau artifact: {path}"
            )

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    if metrics.get("analysis_type") != "bertimbau_finetuned":
        raise ValueError(
            "Invalid BERTimbau analysis type"
        )

    if metrics.get("selection_role") != "predefined_baseline":
        raise ValueError(
            "BERTimbau must remain a predefined baseline"
        )

    if metrics.get("model_id") != FINE_TUNING.model_id:
        raise ValueError(
            "BERTimbau model ID mismatch"
        )

    if metrics.get("model_revision") != FINE_TUNING.revision:
        raise ValueError(
            "BERTimbau revision mismatch"
        )

    if metrics.get("configuration") != asdict(FINE_TUNING):
        raise ValueError(
            "BERTimbau configuration mismatch"
        )

    if metrics.get("seeds") != list(EXPERIMENT.seeds):
        raise ValueError(
            "BERTimbau seed configuration mismatch"
        )

    if metrics.get("threshold") != 0.5:
        raise ValueError(
            "BERTimbau threshold mismatch"
        )

    expected_policy = {
        "validation_during_training": False,
        "early_stopping": False,
        "hyperparameter_search": False,
        "checkpoint_policy": "final_epoch_only",
    }

    if metrics.get("training_policy") != expected_policy:
        raise ValueError(
            "BERTimbau training policy mismatch"
        )

    expected_fingerprints = {
        "train": supervised_dataset_fingerprint(
            train
        ),
        "validation": supervised_dataset_fingerprint(
            validation
        ),
    }

    if metrics.get(
        "supervised_dataset_fingerprints"
    ) != expected_fingerprints:
        raise ValueError(
            "BERTimbau dataset fingerprints mismatch"
        )

    per_seed = metrics.get("per_seed")

    if not isinstance(per_seed, dict):
        raise ValueError(
            "Invalid BERTimbau per-seed results"
        )

    if set(per_seed) != {
        str(seed)
        for seed in EXPERIMENT.seeds
    }:
        raise ValueError(
            "BERTimbau per-seed results mismatch"
        )

    predictions = pd.read_csv(
        predictions_path
    )

    expected_prediction_columns = {
        "id",
        "pair_id",
        "variant",
        "y_true",
        "twin_in_train",
    }

    for seed in EXPERIMENT.seeds:
        expected_prediction_columns.add(
            f"seed_{seed}_probability"
        )

        expected_prediction_columns.add(
            f"seed_{seed}_prediction"
        )

    if set(
        predictions.columns
    ) != expected_prediction_columns:
        raise ValueError(
            "BERTimbau prediction columns mismatch"
        )

    if len(predictions) != len(validation):
        raise ValueError(
            "BERTimbau prediction row count mismatch"
        )

    expected_ids = validation[
        DATA.id_column
    ].astype(str).to_numpy()
    actual_ids = predictions[
        "id"
    ].astype(str).to_numpy()

    if not np.array_equal(
        actual_ids,
        expected_ids,
    ):
        raise ValueError(
            "BERTimbau prediction IDs mismatch"
        )

    expected_labels = validation[
        DATA.label_column
    ].astype(int).to_numpy(copy=True)
    actual_labels = predictions[
        "y_true"
    ].astype(int).to_numpy(copy=True)

    if not np.array_equal(
        actual_labels,
        expected_labels,
    ):
        raise ValueError(
            "BERTimbau prediction labels mismatch"
        )

    expected_twin_mask = twin_in_reference_mask(
        validation,
        train,
    )
    actual_twin_mask = predictions[
        "twin_in_train"
    ].astype(bool).to_numpy(copy=True)

    if not np.array_equal(
        actual_twin_mask,
        expected_twin_mask,
    ):
        raise ValueError(
            "BERTimbau twin mask mismatch"
        )

    overall_runs = []
    twin_runs = []
    no_twin_runs = []

    with np.load(
        predictions_npz_path,
        allow_pickle=False,
    ) as archive:
        expected_archive_keys = {
            "y_true",
            "twin_in_train",
        }

        expected_archive_keys.update(
            {
                f"seed_{seed}_probability"
                for seed in EXPERIMENT.seeds
            }
        )

        if set(
            archive.files
        ) != expected_archive_keys:
            raise ValueError(
                "BERTimbau prediction archive "
                "fields mismatch"
            )

        if not np.array_equal(
            archive["y_true"],
            expected_labels,
        ):
            raise ValueError(
                "BERTimbau archive labels mismatch"
            )

        if not np.array_equal(
            archive["twin_in_train"],
            expected_twin_mask,
        ):
            raise ValueError(
                "BERTimbau archive twin mask mismatch"
            )

        for seed in EXPERIMENT.seeds:
            result = per_seed[str(seed)]

            checkpoint_dir = (
                PATHS.project_root
                / result["checkpoint_dir"]
            )

            if not checkpoint_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing checkpoint for seed {seed}: "
                    f"{checkpoint_dir}"
                )

            metadata_path = (
                checkpoint_dir
                / "training_metadata.json"
            )

            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Missing training metadata for seed {seed}"
                )

            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)

            if metadata.get("seed") != seed:
                raise ValueError(
                    f"Checkpoint seed mismatch for seed {seed}"
                )

            if metadata.get("model_id") != FINE_TUNING.model_id:
                raise ValueError(
                    f"Checkpoint model ID mismatch for seed {seed}"
                )

            if metadata.get(
                "model_revision"
            ) != FINE_TUNING.revision:
                raise ValueError(
                    f"Checkpoint revision mismatch for seed {seed}"
                )

            if metadata.get("configuration") != asdict(FINE_TUNING):
                raise ValueError(
                    f"Checkpoint configuration mismatch for seed {seed}"
                )

            if metadata.get(
                "checkpoint_policy"
            ) != "final_epoch_only":
                raise ValueError(
                    f"Checkpoint policy mismatch for seed {seed}"
                )

            if metadata.get(
                "train_dataset_fingerprint"
            ) != expected_fingerprints["train"]:
                raise ValueError(
                    f"Checkpoint train fingerprint mismatch "
                    f"for seed {seed}"
                )

            if metadata.get(
                "validation_dataset_fingerprint"
            ) != expected_fingerprints["validation"]:
                raise ValueError(
                    f"Checkpoint validation fingerprint mismatch "
                    f"for seed {seed}"
                )

            actual_hashes = checkpoint_sha256(
                checkpoint_dir
            )

            if actual_hashes != result[
                "checkpoint_sha256"
            ]:
                raise ValueError(
                    f"Checkpoint SHA mismatch for seed {seed}"
                )

            if metadata.get(
                "checkpoint_sha256"
            ) != actual_hashes:
                raise ValueError(
                    f"Checkpoint metadata SHA mismatch "
                    f"for seed {seed}"
                )

            training = result.get("training")

            if training != metadata.get("training"):
                raise ValueError(
                    f"Training history mismatch for seed {seed}"
                )

            if len(
                training.get(
                    "epoch_train_loss",
                    [],
                )
            ) != FINE_TUNING.epochs:
                raise ValueError(
                    f"Training epoch count mismatch for seed {seed}"
                )

            probability_key = (
                f"seed_{seed}_probability"
            )

            if probability_key not in archive:
                raise ValueError(
                    f"Missing archived probabilities for seed {seed}"
                )

            probabilities = np.asarray(
                archive[probability_key],
                dtype=np.float64,
            )

            if probabilities.shape != (
                len(validation),
            ):
                raise ValueError(
                    f"Probability shape mismatch for seed {seed}"
                )

            if array_fingerprint(
                probabilities
            ) != result[
                "probability_fingerprint"
            ]:
                raise ValueError(
                    f"Probability fingerprint mismatch for seed {seed}"
                )

            csv_probabilities = predictions[
                probability_key
            ].to_numpy(dtype=np.float64)

            if not np.allclose(
                csv_probabilities,
                probabilities,
                rtol=0.0,
                atol=1e-15,
            ):
                raise ValueError(
                    f"CSV probability mismatch for seed {seed}"
                )

            prediction_key = (
                f"seed_{seed}_prediction"
            )

            stored_predictions = predictions[
                prediction_key
            ].to_numpy(
                dtype=int
            )

            expected_predictions = (
                probabilities_to_predictions(
                    probabilities,
                    threshold=0.5,
                )
            )

            if not np.array_equal(
                stored_predictions,
                expected_predictions,
            ):
                raise ValueError(
                    f"Binary prediction mismatch "
                    f"for seed {seed}"
                )

            overall = compute_binary_metrics(
                expected_labels,
                probabilities,
            )
            twin = compute_binary_metrics(
                expected_labels[
                    expected_twin_mask
                ],
                probabilities[
                    expected_twin_mask
                ],
            )
            no_twin = compute_binary_metrics(
                expected_labels[
                    ~expected_twin_mask
                ],
                probabilities[
                    ~expected_twin_mask
                ],
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

            validate_metric_block(
                overall,
                result["overall"],
                f"overall seed {seed}",
            )
            validate_metric_block(
                twin,
                result["twin_in_train"],
                f"twin seed {seed}",
            )
            validate_metric_block(
                no_twin,
                result["no_twin_in_train"],
                f"no-twin seed {seed}",
            )

    expected_summary = {
        "overall": summarize_binary_metrics(
            overall_runs
        ),
        "twin_in_train": (
            summarize_binary_metrics(
                twin_runs
            )
        ),
        "no_twin_in_train": (
            summarize_binary_metrics(
                no_twin_runs
            )
        ),
    }

    stored_summary = metrics.get(
        "summary"
    )

    if not isinstance(
        stored_summary,
        dict,
    ):
        raise ValueError(
            "Invalid BERTimbau summary"
        )

    for summary_name in expected_summary:
        if summary_name not in stored_summary:
            raise ValueError(
                "Missing BERTimbau summary: "
                f"{summary_name}"
            )

        validate_summary_block(
            expected_summary[
                summary_name
            ],
            stored_summary[
                summary_name
            ],
            summary_name,
        )

    print(
        f"model_id={FINE_TUNING.model_id}"
    )
    print(
        f"revision={FINE_TUNING.revision}"
    )
    print(
        f"max_length={FINE_TUNING.max_length}"
    )
    print(
        f"epochs={FINE_TUNING.epochs}"
    )
    print(
        f"seeds={EXPERIMENT.seeds}"
    )
    print(
        "validation_during_training=False"
    )
    print(
        "early_stopping=False"
    )
    print(
        "hyperparameter_search=False"
    )
    print(
        "checkpoint_policy=final_epoch_only"
    )
    print(
        "BERTimbau fine-tuning evaluation is valid"
    )


if __name__ == "__main__":
    main()
