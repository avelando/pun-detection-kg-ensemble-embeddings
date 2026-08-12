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
from pun_detection.models.tfidf_oof import (
    generate_tfidf_oof_predictions,
)


def main():
    train = load_train_split()

    y_true = train[
        DATA.label_column
    ].astype(int).to_numpy()

    first_run = generate_tfidf_oof_predictions(
        train=train,
        seed=EXPERIMENT.primary_seed,
    )

    second_run = generate_tfidf_oof_predictions(
        train=train,
        seed=EXPERIMENT.primary_seed,
    )

    if first_run.shape != (len(train),):
        raise ValueError(
            f"TF-IDF has invalid shape "
            f"{first_run.shape}"
        )

    if not np.isfinite(first_run).all():
        raise ValueError(
            "TF-IDF contains non-finite predictions"
        )

    if np.any(first_run < 0.0) or np.any(
        first_run > 1.0
    ):
        raise ValueError(
            "TF-IDF contains invalid probabilities"
        )

    if not np.allclose(
        first_run,
        second_run,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "TF-IDF OOF predictions "
            "are not deterministic"
        )

    y_pred = (
        first_run >= 0.5
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
        f"tfidf: "
        f"shape={first_run.shape}, "
        f"accuracy={accuracy:.6f}, "
        f"macro_f1={macro_f1:.6f}, "
        f"min_probability={first_run.min():.6f}, "
        f"max_probability={first_run.max():.6f}"
    )

    print(
        "TF-IDF OOF prediction pipeline is valid"
    )


if __name__ == "__main__":
    main()