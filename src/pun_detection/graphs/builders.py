import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import networkx as nx
import pandas as pd

from pun_detection.config import DATA, GRAPHS
from pun_detection.preprocessing import extract_pun_tokens, normalize_graph_tokens


EdgeKey = tuple[str, str]


@dataclass(frozen=True)
class GraphSet:
    cooccurrence: nx.Graph
    ppmi: nx.Graph
    pun_context: nx.Graph

    def as_dict(self) -> dict[str, nx.Graph]:
        return {
            "cooccurrence": self.cooccurrence,
            "ppmi": self.ppmi,
            "pun_context": self.pun_context,
        }


def _canonical_edge(left: str, right: str) -> EdgeKey:
    return (left, right) if left < right else (right, left)


def _build_vocabulary(
    documents: list[list[str]],
    min_frequency: int,
) -> set[str]:
    frequencies = Counter()

    for document in documents:
        frequencies.update(document)

    return {
        token
        for token, frequency in frequencies.items()
        if frequency >= min_frequency
    }


def _prune_edges(
    edges,
    top_k: int | None,
) -> dict[EdgeKey, float]:
    if top_k is None:
        return {
            edge: float(weight)
            for edge, weight in edges.items()
        }

    neighborhoods = defaultdict(list)

    for (left, right), weight in edges.items():
        neighborhoods[left].append((right, weight))
        neighborhoods[right].append((left, weight))

    kept_edges = set()

    for node, neighbors in neighborhoods.items():
        neighbors.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        for neighbor, _ in neighbors[:top_k]:
            kept_edges.add(
                _canonical_edge(node, neighbor)
            )

    return {
        edge: float(edges[edge])
        for edge in kept_edges
    }


def _edges_to_graph(
    edges: dict[EdgeKey, float],
) -> nx.Graph:
    graph = nx.Graph()

    for (left, right), weight in sorted(edges.items()):
        graph.add_edge(
            left,
            right,
            weight=float(weight),
        )

    return graph


def _normalized_documents(
    dataframe: pd.DataFrame,
) -> list[list[str]]:
    return [
        normalize_graph_tokens(tokens)
        for tokens in dataframe[DATA.tokens_column]
    ]


def build_cooccurrence_graph(
    dataframe: pd.DataFrame,
) -> nx.Graph:
    documents = _normalized_documents(dataframe)

    vocabulary = _build_vocabulary(
        documents,
        GRAPHS.cooccurrence_min_frequency,
    )

    edges = defaultdict(int)
    window = GRAPHS.cooccurrence_window

    for document in documents:
        filtered = [
            token
            for token in document
            if token in vocabulary
        ]

        for index, token in enumerate(filtered):
            left = max(0, index - window)
            right = min(
                len(filtered),
                index + window + 1,
            )

            for neighbor_index in range(left, right):
                if neighbor_index == index:
                    continue

                neighbor = filtered[neighbor_index]

                if neighbor == token:
                    continue

                edge = _canonical_edge(
                    token,
                    neighbor,
                )

                edges[edge] += 1

    pruned = _prune_edges(
        edges,
        GRAPHS.cooccurrence_top_k,
    )

    return _edges_to_graph(pruned)


def build_ppmi_graph(
    dataframe: pd.DataFrame,
) -> nx.Graph:
    documents = _normalized_documents(dataframe)

    vocabulary = _build_vocabulary(
        documents,
        GRAPHS.ppmi_min_frequency,
    )

    cooccurrences = defaultdict(int)
    word_counts = Counter()
    total_pairs = 0
    window = GRAPHS.ppmi_window

    for document in documents:
        filtered = [
            token
            for token in document
            if token in vocabulary
        ]

        for index, token in enumerate(filtered):
            word_counts[token] += 1

            left = max(0, index - window)
            right = min(
                len(filtered),
                index + window + 1,
            )

            for neighbor_index in range(left, right):
                if neighbor_index == index:
                    continue

                neighbor = filtered[neighbor_index]

                if neighbor == token:
                    continue

                edge = _canonical_edge(
                    token,
                    neighbor,
                )

                cooccurrences[edge] += 1
                total_pairs += 1

    total_words = sum(word_counts.values())
    edges = {}

    for (left, right), pair_count in cooccurrences.items():
        pair_probability = pair_count / max(
            total_pairs,
            1,
        )

        left_probability = word_counts[left] / max(
            total_words,
            1,
        )

        right_probability = word_counts[right] / max(
            total_words,
            1,
        )

        pmi = math.log(
            (
                pair_probability
                / (
                    left_probability
                    * right_probability
                    + 1e-12
                )
            )
            + 1e-12
        )

        ppmi = max(0.0, pmi)

        if ppmi > 0.0:
            edges[(left, right)] = ppmi

    pruned = _prune_edges(
        edges,
        GRAPHS.ppmi_top_k,
    )

    return _edges_to_graph(pruned)


def build_pun_context_graph(
    dataframe: pd.DataFrame,
) -> nx.Graph:
    positive = dataframe[
        dataframe[DATA.label_column].astype(int) == 1
    ]

    normalized_documents = {
        index: normalize_graph_tokens(
            row[DATA.tokens_column]
        )
        for index, row in positive.iterrows()
    }

    pun_tokens = {
        index: extract_pun_tokens(
            row[DATA.tokens_column],
            row[DATA.token_labels_column],
        )
        for index, row in positive.iterrows()
    }

    frequencies = Counter()

    for document in normalized_documents.values():
        frequencies.update(document)

    vocabulary = {
        token
        for token, frequency in frequencies.items()
        if frequency
        >= GRAPHS.pun_context_min_frequency
    }

    edges = defaultdict(int)

    for index in positive.index:
        pun = [
            token
            for token in pun_tokens[index]
            if token in vocabulary
        ]

        context = [
            token
            for token in normalized_documents[index]
            if token in vocabulary
        ]

        for pun_token in pun:
            for context_token in context:
                if pun_token == context_token:
                    continue

                edge = _canonical_edge(
                    pun_token,
                    context_token,
                )

                edges[edge] += 1

    pruned = _prune_edges(
        edges,
        GRAPHS.pun_context_top_k,
    )

    return _edges_to_graph(pruned)


def build_graph_set(
    dataframe: pd.DataFrame,
) -> GraphSet:
    return GraphSet(
        cooccurrence=build_cooccurrence_graph(
            dataframe
        ),
        ppmi=build_ppmi_graph(
            dataframe
        ),
        pun_context=build_pun_context_graph(
            dataframe
        ),
    )