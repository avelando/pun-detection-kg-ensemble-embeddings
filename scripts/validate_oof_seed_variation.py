import numpy as np

from pun_detection.config import EXPERIMENT
from pun_detection.data import load_train_split
from pun_detection.oof import create_oof_splits


def main():
    train = load_train_split()

    holdouts_by_seed = {}

    for seed in EXPERIMENT.seeds:
        first = create_oof_splits(
            train,
            seed=seed,
        )

        second = create_oof_splits(
            train,
            seed=seed,
        )

        first_holdouts = tuple(
            split.holdout_indices
            for split in first
        )

        second_holdouts = tuple(
            split.holdout_indices
            for split in second
        )

        for first_indices, second_indices in zip(
            first_holdouts,
            second_holdouts,
        ):
            if not np.array_equal(
                first_indices,
                second_indices,
            ):
                raise ValueError(
                    f"OOF folds are not deterministic "
                    f"for seed {seed}"
                )

        holdouts_by_seed[seed] = first_holdouts

        print(
            f"seed={seed}, "
            f"folds_deterministic=True"
        )

    seeds = list(
        holdouts_by_seed
    )

    reference = holdouts_by_seed[
        seeds[0]
    ]

    variation_found = False

    for seed in seeds[1:]:
        current = holdouts_by_seed[
            seed
        ]

        if any(
            not np.array_equal(
                reference_indices,
                current_indices,
            )
            for reference_indices, current_indices in zip(
                reference,
                current,
            )
        ):
            variation_found = True
            break

    if not variation_found:
        raise ValueError(
            "OOF fold assignments do not vary across seeds"
        )

    print(
        "OOF fold assignments vary across seeds"
    )


if __name__ == "__main__":
    main()