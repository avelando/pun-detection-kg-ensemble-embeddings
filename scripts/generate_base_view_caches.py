import argparse

from pun_detection.base_view_cache import (
    get_base_view_cache_paths,
    get_base_view_cache_state,
    load_base_view_cache,
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
from pun_detection.fingerprints import (
    array_fingerprint,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def print_cache_summary(
    seed: int,
    status: str,
    matrices,
) -> None:
    print(
        f"seed={seed}, "
        f"status={status}, "
        f"selected_embedding_model="
        f"{matrices.selected_embedding_model}"
    )

    print(
        f"seed={seed}, "
        f"columns="
        f"{','.join(matrices.columns)}"
    )

    print(
        f"seed={seed}, "
        f"train_oof_shape="
        f"{matrices.train_oof.shape}, "
        f"train_oof_fingerprint="
        f"{array_fingerprint(matrices.train_oof)}"
    )

    print(
        f"seed={seed}, "
        f"validation_shape="
        f"{matrices.validation.shape}, "
        f"validation_fingerprint="
        f"{array_fingerprint(matrices.validation)}"
    )


def process_seed(
    train,
    validation,
    seed: int,
    force: bool,
) -> None:
    state = get_base_view_cache_state(
        seed=seed,
    )

    if (
        state == "complete"
        and not force
    ):
        matrices = load_base_view_cache(
            train=train,
            validation=validation,
            seed=seed,
        )

        print_cache_summary(
            seed=seed,
            status="reused",
            matrices=matrices,
        )

        return

    if (
        state == "partial"
        and not force
    ):
        matrices_path, metadata_path = (
            get_base_view_cache_paths(
                seed=seed,
            )
        )

        raise RuntimeError(
            "Base view cache is incomplete: "
            f"{matrices_path}, "
            f"{metadata_path}"
        )

    matrices = generate_base_view_matrices(
        train=train,
        validation=validation,
        seed=seed,
    )

    save_base_view_cache(
        matrices=matrices,
        train=train,
        validation=validation,
        seed=seed,
        overwrite=(
            force
            or state == "partial"
        ),
    )

    status = (
        "generated"
        if state == "missing"
        else "regenerated"
    )

    print_cache_summary(
        seed=seed,
        status=status,
        matrices=matrices,
    )


def main():
    args = parse_args()

    splits = load_development_splits()

    for seed in EXPERIMENT.seeds:
        process_seed(
            train=splits.train,
            validation=splits.validation,
            seed=seed,
            force=args.force,
        )

    print(
        "Base view caches are ready"
    )


if __name__ == "__main__":
    main()