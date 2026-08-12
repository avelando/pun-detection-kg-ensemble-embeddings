import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from pun_detection.config import (
    DATA,
    EXPERIMENT,
)
from pun_detection.data import load_train_split
from pun_detection.models.graph_oof import (
    generate_graph_oof_predictions,
)


def main():
    train = load_train_split()

    y_true = train[
        DATA.label_column
    ].astype(int).to_numpy()

    first_run = generate_graph_oof_predictions(
        train=train,
        seed=EXPERIMENT.primary_seed,
    )

    second_run = generate_graph_oof_predictions(
        train=train,
        seed=EXPERIMENT.primary_seed,
    )

    for graph_name in first_run.as_dict():
        first = first_run.as_dict()[
            graph_name
        ]

        second = second_run.as_dict()[
            graph_name
        ]

        if first.shape != (len(train),):
            raise ValueError(
                f"{graph_name} has invalid shape "
                f"{first.shape}"
            )

        if not np.isfinite(first).all():
            raise ValueError(
                f"{graph_name} contains "
                f"non-finite predictions"
            )

        if np.any(first < 0.0) or np.any(
            first > 1.0
        ):
            raise ValueError(
                f"{graph_name} contains invalid "
                f"probabilities"
            )

        if not np.allclose(
            first,
            second,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                f"{graph_name} OOF predictions "
                f"are not deterministic"
            )

        y_pred = (
            first >= 0.5
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
        )

        print(
            f"{graph_name}: "
            f"shape={first.shape}, "
            f"accuracy={accuracy:.6f}, "
            f"macro_f1={macro_f1:.6f}, "
            f"min_probability={first.min():.6f}, "
            f"max_probability={first.max():.6f}"
        )

    print(
        "Graph OOF prediction pipeline is valid"
    )


if __name__ == "__main__":
    main()