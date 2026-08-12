from pun_detection.data import (
    load_development_splits,
)
from pun_detection.selection import (
    load_embedding_selection,
)


def main():
    splits = load_development_splits()

    selection = load_embedding_selection(
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

    for item in selection[
        "ranking"
    ]:
        print(
            f"rank={item['rank']}, "
            f"model={item['model']}, "
            f"macro_f1="
            f"{item['macro_f1']:.6f}, "
            f"accuracy="
            f"{item['accuracy']:.6f}"
        )

    print(
        f"selected_model="
        f"{selection['selected_model']}"
    )

    print(
        "Embedding selection is valid"
    )


if __name__ == "__main__":
    main()