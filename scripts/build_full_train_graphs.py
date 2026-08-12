from pun_detection.config import PATHS
from pun_detection.data import load_train_split
from pun_detection.graphs.builders import build_graph_set
from pun_detection.graphs.io import save_graph_set


def main():
    train = load_train_split()

    graph_set = build_graph_set(
        train
    )

    metadata = save_graph_set(
        graph_set=graph_set,
        output_dir=PATHS.full_train_graphs_dir,
        source_name="train",
        source_dataframe=train,
    )

    for name, statistics in metadata[
        "graphs"
    ].items():
        print(
            f"{name}: "
            f"nodes={statistics['nodes']}, "
            f"edges={statistics['edges']}, "
            f"signature={statistics['signature']}"
        )

    print(
        f"Saved graphs to "
        f"{PATHS.full_train_graphs_dir}"
    )


if __name__ == "__main__":
    main()