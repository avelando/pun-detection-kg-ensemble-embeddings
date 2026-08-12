import numpy as np

from pun_detection.config import EXPERIMENT
from pun_detection.data import load_train_split
from pun_detection.oof import create_oof_splits


def main():
    train = load_train_split()

    first = create_oof_splits(train)
    second = create_oof_splits(train)

    if len(first) != EXPERIMENT.oof_folds:
        raise ValueError(
            "Invalid number of OOF folds"
        )

    coverage = np.zeros(
        len(train),
        dtype=np.int32,
    )

    for left, right in zip(first, second):
        if left.fold != right.fold:
            raise ValueError(
                "OOF fold identifiers differ"
            )

        if not np.array_equal(
            left.train_indices,
            right.train_indices,
        ):
            raise ValueError(
                f"Fold {left.fold} train indices differ"
            )

        if not np.array_equal(
            left.holdout_indices,
            right.holdout_indices,
        ):
            raise ValueError(
                f"Fold {left.fold} holdout indices differ"
            )

        coverage[
            left.holdout_indices
        ] += 1

    if not np.all(coverage == 1):
        raise ValueError(
            "OOF coverage is invalid"
        )

    print(
        f"folds={EXPERIMENT.oof_folds}, "
        f"split_seed={EXPERIMENT.oof_split_seed}, "
        f"instances={len(train)}, "
        f"coverage_valid=True"
    )

    print(
        "OOF integrity is valid"
    )


if __name__ == "__main__":
    main()