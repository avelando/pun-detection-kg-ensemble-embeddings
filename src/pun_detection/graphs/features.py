from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from pun_detection.config import EXPERIMENT, GRAPHS
from pun_detection.preprocessing import tokenize_graph_document


@dataclass(frozen=True)
class GraphFeatureEncoder:
    node_list: tuple[str, ...]
    node_to_index: dict[str, int]
    node_embeddings: np.ndarray
    dimensions: int
    seed: int


def graph_to_sparse_adjacency(
    graph: nx.Graph,
    node_list: Sequence[str],
) -> csr_matrix:
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=list(node_list),
        weight="weight",
        dtype=np.float64,
        format="csr",
    )

    return csr_matrix(adjacency)


def fit_graph_feature_encoder(
    graph: nx.Graph,
    dimensions: int = GRAPHS.svd_dimensions,
    seed: int = EXPERIMENT.primary_seed,
) -> GraphFeatureEncoder:
    if graph.number_of_nodes() <= dimensions:
        raise ValueError(
            f"Graph has {graph.number_of_nodes()} nodes "
            f"but requires more than {dimensions}"
        )

    node_list = tuple(sorted(graph.nodes()))
    node_to_index = {
        node: index
        for index, node in enumerate(node_list)
    }

    adjacency = graph_to_sparse_adjacency(
        graph,
        node_list,
    )

    svd = TruncatedSVD(
        n_components=dimensions,
        random_state=seed,
    )

    node_embeddings = svd.fit_transform(
        adjacency
    ).astype(np.float32)

    return GraphFeatureEncoder(
        node_list=node_list,
        node_to_index=node_to_index,
        node_embeddings=node_embeddings,
        dimensions=dimensions,
        seed=seed,
    )


def transform_documents_with_graph(
    texts: Sequence[str],
    encoder: GraphFeatureEncoder,
) -> np.ndarray:
    feature_dimension = (
        encoder.dimensions
        + GRAPHS.coverage_features
    )

    features = np.zeros(
        (len(texts), feature_dimension),
        dtype=np.float32,
    )

    for row_index, text in enumerate(texts):
        tokens = tokenize_graph_document(text)

        if not tokens:
            continue

        node_indices = [
            encoder.node_to_index[token]
            for token in tokens
            if token in encoder.node_to_index
        ]

        hits = len(node_indices)
        total_tokens = len(tokens)

        if hits:
            features[
                row_index,
                :encoder.dimensions,
            ] = encoder.node_embeddings[
                node_indices
            ].mean(axis=0)

        features[
            row_index,
            encoder.dimensions,
        ] = hits / total_tokens

        features[
            row_index,
            encoder.dimensions + 1,
        ] = np.log1p(hits)

    return features


@dataclass(frozen=True)
class GraphFeatureSet:
    cooccurrence: np.ndarray
    ppmi: np.ndarray
    pun_context: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "cooccurrence": self.cooccurrence,
            "ppmi": self.ppmi,
            "pun_context": self.pun_context,
        }


@dataclass(frozen=True)
class GraphEncoderSet:
    cooccurrence: GraphFeatureEncoder
    ppmi: GraphFeatureEncoder
    pun_context: GraphFeatureEncoder

    def as_dict(self) -> dict[str, GraphFeatureEncoder]:
        return {
            "cooccurrence": self.cooccurrence,
            "ppmi": self.ppmi,
            "pun_context": self.pun_context,
        }


def fit_graph_encoder_set(
    graph_set,
    seed: int = EXPERIMENT.primary_seed,
) -> GraphEncoderSet:
    return GraphEncoderSet(
        cooccurrence=fit_graph_feature_encoder(
            graph_set.cooccurrence,
            seed=seed,
        ),
        ppmi=fit_graph_feature_encoder(
            graph_set.ppmi,
            seed=seed,
        ),
        pun_context=fit_graph_feature_encoder(
            graph_set.pun_context,
            seed=seed,
        ),
    )


def transform_graph_encoder_set(
    texts: Sequence[str],
    encoders: GraphEncoderSet,
) -> GraphFeatureSet:
    return GraphFeatureSet(
        cooccurrence=transform_documents_with_graph(
            texts,
            encoders.cooccurrence,
        ),
        ppmi=transform_documents_with_graph(
            texts,
            encoders.ppmi,
        ),
        pun_context=transform_documents_with_graph(
            texts,
            encoders.pun_context,
        ),
    )