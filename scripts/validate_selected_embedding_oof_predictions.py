import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from pun_detection.config import (
    DATA,
    EXPERIMENT,
)
from pun_detection.data import (
    load_development_splits,
)
from pun_detection.models.embedding_oof import (
    generate_embedding_oof_predictions,
)
from pun_detection.selection import (
    get_selected_embedding_model,
)


def main():
    splits = load_development_splits()

    selected_model = get_selected_embedding_model(
        train=splits.train,
        validation=splits.validation,
    )

    y_true = splits.train[
        DATA.label_column
    ].astype(int).to_numpy()

    first_run = generate_embedding_oof_predictions(
        train=splits.train,
        model_name=selected_model,
        seed=EXPERIMENT.primary_seed,
    )

    second_run = generate_embedding_oof_predictions(
        train=splits.train,
        model_name=selected_model,
        seed=EXPERIMENT.primary_seed,
    )

    if first_run.model_name != selected_model:
        raise ValueError(
            "First OOF run used an unexpected "
            "embedding model"
        )

    if second_run.model_name != selected_model:
        raise ValueError(
            "Second OOF run used an unexpected "
            "embedding model"
        )

    first = first_run.probabilities
    second = second_run.probabilities

    if first.shape != (
        len(splits.train),
    ):
        raise ValueError(
            f"{selected_model} has invalid "
            f"OOF shape {first.shape}"
        )

    if second.shape != first.shape:
        raise ValueError(
            f"{selected_model} OOF runs "
            "have different shapes"
        )

    if not np.isfinite(
        first
    ).all():
        raise ValueError(
            f"{selected_model} contains "
            "non-finite OOF predictions"
        )

    if (
        np.any(first < 0.0)
        or np.any(first > 1.0)
    ):
        raise ValueError(
            f"{selected_model} contains "
            "invalid OOF probabilities"
        )

    if not np.allclose(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            f"{selected_model} OOF predictions "
            "are not deterministic"
        )

    if (
        first_run.fold_iterations
        != second_run.fold_iterations
    ):
        raise ValueError(
            f"{selected_model} OOF fold iterations "
            "are not deterministic"
        )

    if len(
        first_run.fold_iterations
    ) != EXPERIMENT.oof_folds:
        raise ValueError(
            f"{selected_model} has an invalid "
            "number of OOF folds"
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
        f"selected_model={selected_model}"
    )

    print(
        f"fold_iterations="
        f"{first_run.fold_iterations}"
    )

    print(
        f"{selected_model}: "
        f"shape={first.shape}, "
        f"accuracy={accuracy:.6f}, "
        f"macro_f1={macro_f1:.6f}, "
        f"min_probability={first.min():.6f}, "
        f"max_probability={first.max():.6f}"
    )

    print(
        "Selected embedding OOF prediction "
        "pipeline is valid"
    )


if __name__ == "__main__":
    main()