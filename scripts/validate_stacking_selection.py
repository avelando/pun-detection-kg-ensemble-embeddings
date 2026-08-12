from pun_detection.data import (
    load_development_splits,
)
from pun_detection.stacking_selection import (
    load_stacking_selection,
)


def main():
    splits = load_development_splits()

    selection = load_stacking_selection(
        train=splits.train,
        validation=splits.validation,
    )

    print(
        f"selection_split="
        f"{selection['selection_split']}"
    )

    print(
        f"primary_metric="
        f"{selection['primary_metric']}"
    )

    print(
        f"aggregation="
        f"{selection['aggregation']}"
    )

    print(
        f"selection_seeds="
        f"{tuple(selection['selection_seeds'])}"
    )

    print(
        f"selected_embedding_model="
        f"{selection['base_views']['selected_embedding_model']}"
    )

    print(
        f"views="
        f"{','.join(selection['base_views']['columns'])}"
    )

    for item in selection[
        "ranking"
    ]:
        print(
            f"rank={item['rank']}, "
            f"model={item['model']}, "
            f"macro_f1_mean="
            f"{item['macro_f1']['mean']:.6f}, "
            f"macro_f1_std="
            f"{item['macro_f1']['std']:.6f}, "
            f"macro_f1_min="
            f"{item['macro_f1']['min']:.6f}, "
            f"macro_f1_max="
            f"{item['macro_f1']['max']:.6f}"
        )

    print(
        f"selected_model="
        f"{selection['selected_model']}"
    )

    print(
        "Stacking selection is valid"
    )


if __name__ == "__main__":
    main()