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
    load_train_split,
    load_validation_split,
)
from pun_detection.models.graph_model import (
    fit_graph_view_model,
    predict_graph_view_probabilities,
)
from pun_detection.models.tfidf_model import (
    fit_tfidf_view_model,
    predict_tfidf_view_probabilities,
)


def evaluate(
    name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
    )

    print(
        f"{name}: "
        f"accuracy={accuracy:.6f}, "
        f"macro_f1={macro_f1:.6f}, "
        f"min_probability={probabilities.min():.6f}, "
        f"max_probability={probabilities.max():.6f}"
    )


def main():
    train = load_train_split()
    validation = load_validation_split()

    y_validation = validation[
        DATA.label_column
    ].astype(int).to_numpy()

    tfidf_model = fit_tfidf_view_model(
        train=train,
        seed=EXPERIMENT.primary_seed,
    )

    tfidf_probabilities = (
        predict_tfidf_view_probabilities(
            tfidf_model,
            validation,
        )
    )

    evaluate(
        "tfidf",
        y_validation,
        tfidf_probabilities,
    )

    graph_model = fit_graph_view_model(
        train=train,
        seed=EXPERIMENT.primary_seed,
    )

    graph_probabilities = (
        predict_graph_view_probabilities(
            graph_model,
            validation,
        )
    )

    for graph_name, probabilities in (
        graph_probabilities.items()
    ):
        evaluate(
            graph_name,
            y_validation,
            probabilities,
        )


if __name__ == "__main__":
    main()