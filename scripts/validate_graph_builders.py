from pun_detection.data import load_dataset_splits
from pun_detection.graphs.builders import build_graph_set
from pun_detection.graphs.io import graph_statistics


EXPECTED = {
    "cooccurrence": {
        "nodes": 1153,
        "edges": 22730,
        "signature": (
            "06eb23ea501b6901a53e44a0d10e5858"
            "33892452f43295dd4941f70651ea0851"
        ),
    },
    "ppmi": {
        "nodes": 1153,
        "edges": 18363,
        "signature": (
            "89fe470aa4f09ee37e2aec7b66801ef7"
            "ef75e74cc451c7d9785d7b1eb504dcf7"
        ),
    },
    "pun_context": {
        "nodes": 921,
        "edges": 5071,
        "signature": (
            "9c5cd37c97278aa4c39cb42d1c915a4"
            "78dd240a5da9ff8f4d279841a611836b4"
        ),
    },
}


def main():
    splits = load_dataset_splits()

    graph_set = build_graph_set(
        splits.train
    )

    for name, graph in graph_set.as_dict().items():
        actual = graph_statistics(graph)
        expected = EXPECTED[name]

        for key, expected_value in expected.items():
            if actual[key] != expected_value:
                raise ValueError(
                    f"{name} {key} mismatch: "
                    f"actual={actual[key]}, "
                    f"expected={expected_value}"
                )

        print(
            f"{name}: "
            f"nodes={actual['nodes']}, "
            f"edges={actual['edges']}, "
            f"signature={actual['signature']}"
        )

    print(
        "Graph builders reproduce "
        "the reference graphs"
    )


if __name__ == "__main__":
    main()