import gc
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
from pun_detection.fine_tuning import (
    clear_fine_tuning_device_cache,
    get_fine_tuning_device,
    load_fine_tuning_model,
    load_fine_tuning_tokenizer,
    make_evaluation_loader,
    make_fine_tuning_dataset,
    make_train_loader,
    predict_fine_tuning_probabilities,
    save_fine_tuning_checkpoint,
    train_fine_tuning_model,
)
from pun_detection.fingerprints import (
    array_fingerprint,
    supervised_dataset_fingerprint,
)
from pun_detection.pairs import twin_in_reference_mask


def validate_checkpoint_targets() -> None:
    existing = []

    for seed in EXPERIMENT.seeds:
        checkpoint_dir = (
            PATHS.fine_tuned_models_dir
            / f"seed_{seed}"
        )

        if checkpoint_dir.exists() and any(
            checkpoint_dir.iterdir()
        ):
            existing.append(
                str(checkpoint_dir)
            )

    if existing:
        raise FileExistsError(
            "Fine-tuning checkpoints already exist: "
            f"{existing}"
        )


def main():
    validate_checkpoint_targets()

    splits = load_development_splits()
    train = splits.train
    validation = splits.validation

    y_validation = validation[
        DATA.label_column
    ].astype(int).to_numpy(copy=True)

    twin_mask = twin_in_reference_mask(
        validation,
        train,
    )
    no_twin_mask = ~twin_mask

    tokenizer = load_fine_tuning_tokenizer()

    train_dataset = make_fine_tuning_dataset(
        dataframe=train,
        tokenizer=tokenizer,
    )
    validation_dataset = make_fine_tuning_dataset(
        dataframe=validation,
        tokenizer=tokenizer,
    )
    evaluation_loader = make_evaluation_loader(
        dataset=validation_dataset,
    )

    device = get_fine_tuning_device()

    prediction_columns = {
        "id": validation[
            DATA.id_column
        ].astype(str).to_numpy(copy=True),
        "pair_id": validation[
            "pair_id"
        ].astype(str).to_numpy(copy=True),
        "variant": validation[
            "variant"
        ].astype(str).to_numpy(copy=True),
        "y_true": y_validation,
        "twin_in_train": twin_mask,
    }

    probability_arrays = {}

    overall_runs = []
    twin_runs = []
    no_twin_runs = []
    per_seed = {}

    for seed in EXPERIMENT.seeds:
        print()
        print(
            f"seed={seed}, stage=training"
        )

        train_loader = make_train_loader(
            dataset=train_dataset,
            seed=seed,
        )

        model = load_fine_tuning_model(
            seed=seed,
            device=device,
        )

        training_history = train_fine_tuning_model(
            model=model,
            train_loader=train_loader,
            device=device,
        )

        print(
            f"seed={seed}, stage=validation"
        )

        probabilities = predict_fine_tuning_probabilities(
            model=model,
            evaluation_loader=evaluation_loader,
            device=device,
        )

        if probabilities.shape != (
            len(validation),
        ):
            raise ValueError(
                "Fine-tuned validation "
                "prediction count mismatch"
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

        checkpoint_dir = (
            PATHS.fine_tuned_models_dir
            / f"seed_{seed}"
        )

        checkpoint_metadata = save_fine_tuning_checkpoint(
            model=model,
            tokenizer=tokenizer,
            output_dir=checkpoint_dir,
            seed=seed,
            train_dataframe=train,
            validation_dataframe=validation,
            training_history=training_history,
        )

        probability_fingerprint = array_fingerprint(
            probabilities
        )

        probability_arrays[seed] = probabilities.copy()

        per_seed[str(seed)] = {
            "overall": overall.as_dict(),
            "twin_in_train": twin.as_dict(),
            "no_twin_in_train": no_twin.as_dict(),
            "probability_fingerprint": (
                probability_fingerprint
            ),
            "checkpoint_dir": str(
                checkpoint_dir.relative_to(
                    PATHS.project_root
                )
            ),
            "checkpoint_sha256": (
                checkpoint_metadata[
                    "checkpoint_sha256"
                ]
            ),
            "training": training_history,
        }

        prediction_columns[
            f"seed_{seed}_probability"
        ] = probabilities
        prediction_columns[
            f"seed_{seed}_prediction"
        ] = probabilities_to_predictions(
            probabilities
        )

        print(
            f"seed={seed}, "
            f"accuracy={overall.accuracy:.6f}, "
            f"macro_f1={overall.macro_f1:.6f}, "
            f"twin_macro_f1={twin.macro_f1:.6f}, "
            f"no_twin_macro_f1={no_twin.macro_f1:.6f}"
        )

        del model
        del train_loader
        gc.collect()
        clear_fine_tuning_device_cache()

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

    output = {
        "analysis_type": "bertimbau_finetuned",
        "selection_role": "predefined_baseline",
        "model_id": FINE_TUNING.model_id,
        "model_revision": FINE_TUNING.revision,
        "configuration": asdict(FINE_TUNING),
        "seeds": list(EXPERIMENT.seeds),
        "threshold": 0.5,
        "training_policy": {
            "validation_during_training": False,
            "early_stopping": False,
            "hyperparameter_search": False,
            "checkpoint_policy": "final_epoch_only",
        },
        "supervised_dataset_fingerprints": {
            "train": supervised_dataset_fingerprint(
                train
            ),
            "validation": supervised_dataset_fingerprint(
                validation
            ),
        },
        "per_seed": per_seed,
        "summary": summary,
    }

    PATHS.validation_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    pd.DataFrame(
        prediction_columns
    ).to_csv(
        predictions_path,
        index=False,
        float_format="%.17g",
    )

    np.savez_compressed(
        predictions_npz_path,
        y_true=y_validation,
        twin_in_train=twin_mask,
        **{
            f"seed_{seed}_probability": probability_arrays[seed]
            for seed in EXPERIMENT.seeds
        },
    )

    print()
    print(
        "BERTimbau fine-tuning summary"
    )

    overall_summary = summary[
        "overall"
    ]
    twin_summary = summary[
        "twin_in_train"
    ]
    no_twin_summary = summary[
        "no_twin_in_train"
    ]

    print(
        f"accuracy_mean="
        f"{overall_summary['accuracy']['mean']:.6f}, "
        f"accuracy_std="
        f"{overall_summary['accuracy']['std']:.6f}"
    )
    print(
        f"macro_f1_mean="
        f"{overall_summary['macro_f1']['mean']:.6f}, "
        f"macro_f1_std="
        f"{overall_summary['macro_f1']['std']:.6f}"
    )
    print(
        f"twin_macro_f1_mean="
        f"{twin_summary['macro_f1']['mean']:.6f}, "
        f"twin_macro_f1_std="
        f"{twin_summary['macro_f1']['std']:.6f}"
    )
    print(
        f"no_twin_macro_f1_mean="
        f"{no_twin_summary['macro_f1']['mean']:.6f}, "
        f"no_twin_macro_f1_std="
        f"{no_twin_summary['macro_f1']['std']:.6f}"
    )
    print(
        f"Saved metrics to {metrics_path}"
    )
    print(
        f"Saved predictions to {predictions_path}"
    )
    print(
        f"Saved prediction archive to {predictions_npz_path}"
    )


if __name__ == "__main__":
    main()
