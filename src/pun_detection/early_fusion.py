from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import (
    csr_matrix,
    hstack,
    issparse,
)
from sklearn.preprocessing import StandardScaler

from pun_detection.config import (
    DATA,
)
from pun_detection.embedding_cache import (
    load_embedding_cache,
)
from pun_detection.graphs.builders import (
    build_graph_set,
)
from pun_detection.graphs.features import (
    fit_graph_encoder_set,
    transform_graph_encoder_set,
)
from pun_detection.selection import (
    get_selected_embedding_model,
)
from pun_detection.text.features import (
    make_tfidf_vectorizer,
)


EARLY_FUSION_COMPONENTS = (
    "tfidf",
    "selected_embedding",
    "cooccurrence",
    "ppmi",
    "pun_context",
)


EARLY_FUSION_CONFIGURATIONS = {
    "tfidf_only": (
        "tfidf",
    ),
    "selected_embedding_only": (
        "selected_embedding",
    ),
    "all_graphs": (
        "cooccurrence",
        "ppmi",
        "pun_context",
    ),
    "tfidf_selected_embedding": (
        "tfidf",
        "selected_embedding",
    ),
    "selected_embedding_cooccurrence": (
        "selected_embedding",
        "cooccurrence",
    ),
    "selected_embedding_ppmi": (
        "selected_embedding",
        "ppmi",
    ),
    "selected_embedding_pun_context": (
        "selected_embedding",
        "pun_context",
    ),
    "selected_embedding_all_graphs": (
        "selected_embedding",
        "cooccurrence",
        "ppmi",
        "pun_context",
    ),
    "tfidf_all_graphs": (
        "tfidf",
        "cooccurrence",
        "ppmi",
        "pun_context",
    ),
    "tfidf_selected_embedding_cooccurrence": (
        "tfidf",
        "selected_embedding",
        "cooccurrence",
    ),
    "tfidf_selected_embedding_ppmi": (
        "tfidf",
        "selected_embedding",
        "ppmi",
    ),
    "tfidf_selected_embedding_pun_context": (
        "tfidf",
        "selected_embedding",
        "pun_context",
    ),
    "all_components": EARLY_FUSION_COMPONENTS,
}


@dataclass(frozen=True)
class EarlyFusionFeatureSpace:
    selected_embedding_model: str
    train_components: dict
    target_components: dict
    component_dimensions: dict[str, int]


def validate_early_fusion_components(
    components,
) -> tuple[str, ...]:
    components = tuple(
        components
    )

    if not components:
        raise ValueError(
            "Early fusion components cannot be empty"
        )

    if len(
        components
    ) != len(
        set(components)
    ):
        raise ValueError(
            "Early fusion components contain duplicates"
        )

    unknown = [
        component
        for component in components
        if component not in EARLY_FUSION_COMPONENTS
    ]

    if unknown:
        raise ValueError(
            f"Unknown early fusion components: {unknown}"
        )

    return components


def validate_early_fusion_configurations() -> None:
    if not EARLY_FUSION_CONFIGURATIONS:
        raise ValueError(
            "Early fusion configurations cannot be empty"
        )

    for name, components in (
        EARLY_FUSION_CONFIGURATIONS.items()
    ):
        if not name:
            raise ValueError(
                "Early fusion configuration name "
                "cannot be empty"
            )

        validate_early_fusion_components(
            components
        )

    if EARLY_FUSION_CONFIGURATIONS[
        "all_components"
    ] != EARLY_FUSION_COMPONENTS:
        raise ValueError(
            "All-components configuration is invalid"
        )


def fit_early_fusion_feature_space(
    train: pd.DataFrame,
    target: pd.DataFrame,
    target_split_name: str,
    seed: int,
) -> EarlyFusionFeatureSpace:
    selected_embedding_model = (
        get_selected_embedding_model(
            train=train,
            validation=target,
        )
        if target_split_name == "validation"
        else None
    )

    if selected_embedding_model is None:
        raise ValueError(
            "Early fusion development currently "
            "requires validation as target"
        )

    vectorizer = make_tfidf_vectorizer()

    train_tfidf = vectorizer.fit_transform(
        train[
            DATA.text_column
        ].astype(str)
    )

    target_tfidf = vectorizer.transform(
        target[
            DATA.text_column
        ].astype(str)
    )

    train_embedding = load_embedding_cache(
        model_name=selected_embedding_model,
        split_name="train",
        dataframe=train,
    )

    target_embedding = load_embedding_cache(
        model_name=selected_embedding_model,
        split_name=target_split_name,
        dataframe=target,
    )

    graph_set = build_graph_set(
        train
    )

    graph_encoders = fit_graph_encoder_set(
        graph_set,
        seed=seed,
    )

    train_graph_features = (
        transform_graph_encoder_set(
            train[
                DATA.text_column
            ].tolist(),
            graph_encoders,
        )
    )

    target_graph_features = (
        transform_graph_encoder_set(
            target[
                DATA.text_column
            ].tolist(),
            graph_encoders,
        )
    )

    train_components = {
        "tfidf": train_tfidf,
        "selected_embedding": np.asarray(
            train_embedding,
            dtype=np.float64,
        ),
    }

    target_components = {
        "tfidf": target_tfidf,
        "selected_embedding": np.asarray(
            target_embedding,
            dtype=np.float64,
        ),
    }

    for graph_name in (
        "cooccurrence",
        "ppmi",
        "pun_context",
    ):
        train_graph = (
            train_graph_features.as_dict()[
                graph_name
            ]
        )

        target_graph = (
            target_graph_features.as_dict()[
                graph_name
            ]
        )

        scaler = StandardScaler()

        train_graph_scaled = (
            scaler.fit_transform(
                train_graph
            )
        )

        target_graph_scaled = (
            scaler.transform(
                target_graph
            )
        )

        train_components[
            graph_name
        ] = np.asarray(
            train_graph_scaled,
            dtype=np.float64,
        )

        target_components[
            graph_name
        ] = np.asarray(
            target_graph_scaled,
            dtype=np.float64,
        )

    component_dimensions = {
        component_name: int(
            component_matrix.shape[1]
        )
        for component_name, component_matrix in (
            train_components.items()
        )
    }

    return EarlyFusionFeatureSpace(
        selected_embedding_model=(
            selected_embedding_model
        ),
        train_components=train_components,
        target_components=target_components,
        component_dimensions=component_dimensions,
    )


def combine_early_fusion_components(
    component_matrices: dict,
    components,
):
    components = validate_early_fusion_components(
        components
    )

    matrices = [
        component_matrices[
            component
        ]
        for component in components
    ]

    row_counts = {
        matrix.shape[0]
        for matrix in matrices
    }

    if len(
        row_counts
    ) != 1:
        raise ValueError(
            "Early fusion components have "
            "different row counts"
        )

    if "tfidf" in components:
        sparse_matrices = [
            matrix
            if issparse(matrix)
            else csr_matrix(
                np.asarray(
                    matrix,
                    dtype=np.float64,
                )
            )
            for matrix in matrices
        ]

        combined = hstack(
            sparse_matrices,
            format="csr",
        )

        if combined.data.size and not np.isfinite(
            combined.data
        ).all():
            raise ValueError(
                "Early fusion matrix contains "
                "non-finite values"
            )

        return combined

    dense_matrices = [
        np.asarray(
            matrix,
            dtype=np.float64,
        )
        for matrix in matrices
    ]

    combined = np.column_stack(
        dense_matrices
    )

    if not np.isfinite(
        combined
    ).all():
        raise ValueError(
            "Early fusion matrix contains "
            "non-finite values"
        )

    return combined