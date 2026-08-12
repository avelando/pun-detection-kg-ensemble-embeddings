import numpy as np

from pun_detection.config import DATA, GRAPHS
from pun_detection.data import load_development_splits
from pun_detection.graphs.builders import build_graph_set
from pun_detection.graphs.features import (
    fit_graph_encoder_set,
    transform_graph_encoder_set,
)


def validate_feature_matrix(
    name,
    matrix,
    expected_rows,
):
    expected_columns = (
        GRAPHS.svd_dimensions
        + GRAPHS.coverage_features
    )

    if matrix.shape != (
        expected_rows,
        expected_columns,
    ):
        raise ValueError(
            f"{name} has shape {matrix.shape}, "
            f"expected "
            f"({expected_rows}, {expected_columns})"
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            f"{name} contains non-finite values"
        )

    coverage = matrix[
        :,
        GRAPHS.svd_dimensions,
    ]

    log_hits = matrix[
        :,
        GRAPHS.svd_dimensions + 1,
    ]

    if np.any(coverage < 0.0):
        raise ValueError(
            f"{name} contains negative coverage"
        )

    if np.any(coverage > 1.0):
        raise ValueError(
            f"{name} contains coverage above one"
        )

    if np.any(log_hits < 0.0):
        raise ValueError(
            f"{name} contains negative log hits"
        )


def main():
    splits = load_development_splits()

    graph_set = build_graph_set(
        splits.train
    )

    encoders_a = fit_graph_encoder_set(
        graph_set
    )

    encoders_b = fit_graph_encoder_set(
        graph_set
    )

    for graph_name in encoders_a.as_dict():
        first = encoders_a.as_dict()[
            graph_name
        ].node_embeddings

        second = encoders_b.as_dict()[
            graph_name
        ].node_embeddings

        if not np.allclose(
            first,
            second,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                f"{graph_name} SVD is not deterministic"
            )

    train_features = transform_graph_encoder_set(
        splits.train[DATA.text_column].tolist(),
        encoders_a,
    )

    validation_features = (
        transform_graph_encoder_set(
            splits.validation[
                DATA.text_column
            ].tolist(),
            encoders_a,
        )
    )

    split_features = {
        "train": (
            train_features,
            len(splits.train),
        ),
        "validation": (
            validation_features,
            len(splits.validation),
        ),
    }

    for split_name, (
        feature_set,
        expected_rows,
    ) in split_features.items():
        for graph_name, matrix in (
            feature_set.as_dict().items()
        ):
            name = (
                f"{split_name}/{graph_name}"
            )

            validate_feature_matrix(
                name,
                matrix,
                expected_rows,
            )

            coverage = matrix[
                :,
                GRAPHS.svd_dimensions,
            ]

            zero_coverage = int(
                np.sum(coverage == 0.0)
            )

            mean_coverage = float(
                coverage.mean()
            )

            print(
                f"{name}: "
                f"shape={matrix.shape}, "
                f"mean_coverage="
                f"{mean_coverage:.6f}, "
                f"zero_coverage={zero_coverage}"
            )

    print(
        "Graph feature extraction is valid"
    )


if __name__ == "__main__":
    main()