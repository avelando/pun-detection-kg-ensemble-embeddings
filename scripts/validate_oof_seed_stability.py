import numpy as np

from pun_detection.config import EXPERIMENT
from pun_detection.data import load_train_split
from pun_detection.oof import create_oof_splits


def main():
    train = load_train_split()

    reference = create_oof_splits(
        train
    )

    reference_holdouts = [
        split.holdout_indices
        for split in reference
    ]

    for model_seed in EXPERIMENT.seeds:
        current = create_oof_splits(
            train
        )

        for reference_indices, current_split in zip(
            reference_holdouts,
            current,
        ):
            if not np.array_equal(
                reference_indices,
                current_split.holdout_indices,
            ):
                raise ValueError(
                    f"OOF folds changed for model seed "
                    f"{model_seed}"
                )

        print(
            f"model_seed={model_seed}, "
            f"oof_split_seed={EXPERIMENT.oof_split_seed}, "
            f"folds_stable=True"
        )

    print(
        "OOF fold assignment is stable"
    )


if __name__ == "__main__":
    main()