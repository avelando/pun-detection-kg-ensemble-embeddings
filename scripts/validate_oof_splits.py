import numpy as np

from pun_detection.config import DATA, EXPERIMENT
from pun_detection.data import load_dataset_splits
from pun_detection.oof import create_oof_splits


def main():
    splits = load_dataset_splits()
    train = splits.train

    oof_splits = create_oof_splits(train)

    if len(oof_splits) != EXPERIMENT.oof_folds:
        raise ValueError(
            f"Expected {EXPERIMENT.oof_folds} folds, "
            f"found {len(oof_splits)}"
        )

    holdout_counts = np.zeros(
        len(train),
        dtype=np.int32,
    )

    for split in oof_splits:
        fold_train = train.iloc[
            split.train_indices
        ]

        fold_holdout = train.iloc[
            split.holdout_indices
        ]

        train_ids = set(
            fold_train[DATA.id_column]
        )

        holdout_ids = set(
            fold_holdout[DATA.id_column]
        )

        if train_ids.intersection(holdout_ids):
            raise ValueError(
                f"Fold {split.fold} has instance overlap"
            )

        train_pairs = set(
            fold_train["pair_id"]
        )

        holdout_pairs = set(
            fold_holdout["pair_id"]
        )

        overlapping_pairs = (
            train_pairs.intersection(
                holdout_pairs
            )
        )

        if overlapping_pairs:
            raise ValueError(
                f"Fold {split.fold} has pair overlap"
            )

        holdout_counts[
            split.holdout_indices
        ] += 1

        train_class_counts = (
            fold_train[
                DATA.label_column
            ]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        )

        holdout_class_counts = (
            fold_holdout[
                DATA.label_column
            ]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            f"fold={split.fold}, "
            f"train={len(fold_train)}, "
            f"holdout={len(fold_holdout)}, "
            f"train_pairs={fold_train['pair_id'].nunique()}, "
            f"holdout_pairs={fold_holdout['pair_id'].nunique()}, "
            f"train_classes={train_class_counts}, "
            f"holdout_classes={holdout_class_counts}"
        )

    if not np.all(holdout_counts == 1):
        invalid = np.where(
            holdout_counts != 1
        )[0]

        raise ValueError(
            f"OOF coverage is invalid for "
            f"{len(invalid)} instances"
        )

    print(
        "OOF split validation is valid"
    )


if __name__ == "__main__":
    main()