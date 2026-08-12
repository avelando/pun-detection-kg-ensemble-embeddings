from collections import Counter

from pun_detection.config import DATA, EXPERIMENT, GRAPHS
from pun_detection.data import (
    load_development_splits,
    test_access_is_unlocked,
)


def print_split_summary(name, dataframe):
    label_counts = Counter(dataframe[DATA.label_column].astype(int))
    pair_count = dataframe["pair_id"].nunique()

    print(
        f"{name}: "
        f"rows={len(dataframe)}, "
        f"pairs={pair_count}, "
        f"class_0={label_counts[0]}, "
        f"class_1={label_counts[1]}"
    )


def main():
    splits = load_development_splits()

    if not EXPERIMENT.seeds:
        raise ValueError(
            "Experiment seeds cannot be empty"
        )

    if len(
        EXPERIMENT.seeds
    ) != len(
        set(EXPERIMENT.seeds)
    ):
        raise ValueError(
            "Experiment seeds must be unique"
        )

    if (
        EXPERIMENT.primary_seed
        not in EXPERIMENT.seeds
    ):
        raise ValueError(
            "Primary seed must be included "
            "in experiment seeds"
        )

    print_split_summary("train", splits.train)
    print_split_summary("validation", splits.validation)

    print(f"oof_folds={EXPERIMENT.oof_folds}")
    print(f"primary_seed={EXPERIMENT.primary_seed}")
    print(f"seeds={EXPERIMENT.seeds}")
    print(f"primary_metric={EXPERIMENT.primary_metric}")
    print(f"graph_svd_dimensions={GRAPHS.svd_dimensions}")

    test_access = (
        "unlocked"
        if test_access_is_unlocked()
        else "locked"
    )

    print(f"test_access={test_access}")
    print("Dataset setup is valid")


if __name__ == "__main__":
    main()