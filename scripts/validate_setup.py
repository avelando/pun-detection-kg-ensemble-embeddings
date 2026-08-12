from collections import Counter

from pun_detection.config import DATA, EXPERIMENT, GRAPHS
from pun_detection.data import load_dataset_splits


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
    splits = load_dataset_splits()

    print_split_summary("train", splits.train)
    print_split_summary("validation", splits.validation)
    print_split_summary("test", splits.test)

    print(f"oof_folds={EXPERIMENT.oof_folds}")
    print(f"primary_seed={EXPERIMENT.primary_seed}")
    print(f"seeds={EXPERIMENT.seeds}")
    print(f"primary_metric={EXPERIMENT.primary_metric}")
    print(f"graph_svd_dimensions={GRAPHS.svd_dimensions}")
    print("Dataset setup is valid")


if __name__ == "__main__":
    main()