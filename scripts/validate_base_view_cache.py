from pun_detection.base_view_cache import (
    get_base_view_cache_paths,
    load_base_view_cache,
)
from pun_detection.config import (
    EXPERIMENT,
)
from pun_detection.data import (
    load_development_splits,
)


def main():
    splits = load_development_splits()

    matrices = load_base_view_cache(
        train=splits.train,
        validation=splits.validation,
        seed=EXPERIMENT.primary_seed,
    )

    matrices_path, metadata_path = (
        get_base_view_cache_paths(
            seed=EXPERIMENT.primary_seed,
        )
    )

    print(
        f"selected_embedding_model="
        f"{matrices.selected_embedding_model}"
    )

    print(
        f"columns="
        f"{','.join(matrices.columns)}"
    )

    print(
        f"train_oof_shape="
        f"{matrices.train_oof.shape}"
    )

    print(
        f"validation_shape="
        f"{matrices.validation.shape}"
    )

    print(
        f"matrices_path="
        f"{matrices_path}"
    )

    print(
        f"metadata_path="
        f"{metadata_path}"
    )

    print(
        "Base view cache is valid"
    )


if __name__ == "__main__":
    main()