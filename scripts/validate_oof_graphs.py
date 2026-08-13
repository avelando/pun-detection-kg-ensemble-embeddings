from pun_detection.config import EXPERIMENT
from pun_detection.data import load_train_split
from pun_detection.graphs.builders import build_graph_set
from pun_detection.graphs.features import (
    fit_graph_encoder_set,
    transform_graph_encoder_set,
)
from pun_detection.graphs.io import graph_statistics
from pun_detection.oof import create_oof_splits


def main():
    train = load_train_split()

    oof_splits = create_oof_splits(
        train,
        seed=EXPERIMENT.primary_seed,
    )

    for split in oof_splits:
        fold_train = train.iloc[
            split.train_indices
        ]

        fold_holdout = train.iloc[
            split.holdout_indices
        ]

        graph_set = build_graph_set(
            fold_train
        )

        encoders = fit_graph_encoder_set(
            graph_set
        )

        holdout_features = (
            transform_graph_encoder_set(
                fold_holdout["text"].tolist(),
                encoders,
            )
        )

        print(
            f"fold={split.fold}, "
            f"train={len(fold_train)}, "
            f"holdout={len(fold_holdout)}"
        )

        for graph_name, graph in (
            graph_set.as_dict().items()
        ):
            statistics = graph_statistics(
                graph
            )

            matrix = (
                holdout_features
                .as_dict()[graph_name]
            )

            print(
                f"  {graph_name}: "
                f"nodes={statistics['nodes']}, "
                f"edges={statistics['edges']}, "
                f"features={matrix.shape}"
            )

    print(
        "OOF graph construction is valid"
    )


if __name__ == "__main__":
    main()