import argparse

from pun_detection.base_view_cache import (
    get_base_view_cache_paths,
    save_base_view_cache,
)
from pun_detection.base_views import (
    generate_base_view_matrices,
)
from pun_detection.config import (
    EXPERIMENT,
)
from pun_detection.data import (
    load_development_splits,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    splits = load_development_splits()

    matrices = generate_base_view_matrices(
        train=splits.train,
        validation=splits.validation,
        seed=EXPERIMENT.primary_seed,
    )

    metadata = save_base_view_cache(
        matrices=matrices,
        train=splits.train,
        validation=splits.validation,
        seed=EXPERIMENT.primary_seed,
        overwrite=args.force,
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
        f"train_oof_fingerprint="
        f"{metadata['matrices']['train_oof']['fingerprint']}"
    )

    print(
        f"validation_fingerprint="
        f"{metadata['matrices']['validation']['fingerprint']}"
    )

    print(
        f"Saved matrices to "
        f"{matrices_path}"
    )

    print(
        f"Saved metadata to "
        f"{metadata_path}"
    )


if __name__ == "__main__":
    main()