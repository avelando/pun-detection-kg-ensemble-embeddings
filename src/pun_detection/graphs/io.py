import csv
import hashlib
import json
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

from pun_detection.config import DATA, GRAPHS
from pun_detection.graphs.builders import GraphSet


GRAPH_FILENAMES = {
    "cooccurrence": "cooccurrence.gpickle",
    "ppmi": "ppmi.gpickle",
    "pun_context": "pun_context.gpickle",
}


def graph_signature(graph: nx.Graph) -> str:
    rows = []

    for left, right, attributes in graph.edges(
        data=True
    ):
        first, second = sorted(
            (str(left), str(right))
        )

        weight = format(
            float(
                attributes.get(
                    "weight",
                    1.0,
                )
            ),
            ".17g",
        )

        rows.append(
            f"{first}\t{second}\t{weight}"
        )

    payload = "\n".join(
        sorted(rows)
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def instance_id_signature(
    dataframe: pd.DataFrame,
) -> str:
    payload = "\n".join(
        dataframe[
            DATA.id_column
        ].astype(str).tolist()
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def graph_statistics(
    graph: nx.Graph,
) -> dict[str, int | float | str]:
    components = (
        nx.number_connected_components(graph)
        if graph.number_of_nodes()
        else 0
    )

    largest_component = (
        len(
            max(
                nx.connected_components(graph),
                key=len,
            )
        )
        if graph.number_of_nodes()
        else 0
    )

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "components": components,
        "largest_component_nodes": largest_component,
        "signature": graph_signature(graph),
    }


def save_graph(
    graph: nx.Graph,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("wb") as file:
        pickle.dump(
            graph,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def save_edges(
    graph: nx.Graph,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            ["src", "dst", "weight"]
        )

        edges = []

        for left, right, attributes in graph.edges(
            data=True
        ):
            first, second = sorted(
                (str(left), str(right))
            )

            edges.append(
                (
                    first,
                    second,
                    float(
                        attributes.get(
                            "weight",
                            1.0,
                        )
                    ),
                )
            )

        for row in sorted(edges):
            writer.writerow(row)


def save_graph_set(
    graph_set: GraphSet,
    output_dir: Path,
    source_name: str,
    source_dataframe: pd.DataFrame,
) -> dict:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "source": source_name,
        "rows": len(source_dataframe),
        "instance_id_signature": (
            instance_id_signature(
                source_dataframe
            )
        ),
        "config": {
            "cooccurrence_window": (
                GRAPHS.cooccurrence_window
            ),
            "cooccurrence_min_frequency": (
                GRAPHS.cooccurrence_min_frequency
            ),
            "cooccurrence_top_k": (
                GRAPHS.cooccurrence_top_k
            ),
            "ppmi_window": (
                GRAPHS.ppmi_window
            ),
            "ppmi_min_frequency": (
                GRAPHS.ppmi_min_frequency
            ),
            "ppmi_top_k": (
                GRAPHS.ppmi_top_k
            ),
            "pun_context_min_frequency": (
                GRAPHS.pun_context_min_frequency
            ),
            "pun_context_top_k": (
                GRAPHS.pun_context_top_k
            ),
            "svd_dimensions": (
                GRAPHS.svd_dimensions
            ),
        },
        "graphs": {},
    }

    for name, graph in graph_set.as_dict().items():
        graph_path = (
            output_dir
            / GRAPH_FILENAMES[name]
        )

        edge_path = (
            output_dir
            / f"{name}_edges.csv"
        )

        save_graph(
            graph,
            graph_path,
        )

        save_edges(
            graph,
            edge_path,
        )

        metadata["graphs"][name] = (
            graph_statistics(graph)
        )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    return metadata