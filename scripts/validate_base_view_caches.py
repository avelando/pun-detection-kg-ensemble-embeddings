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
from pun_detection.fingerprints import (
    array_fingerprint,
)


def main():
    splits = load_development_splits()

    for seed in EXPERIMENT.seeds:
        matrices = load_base_view_cache(
            train=splits.train,
            validation=splits.validation,
            seed=seed,
        )

        matrices_path, metadata_path = (
            get_base_view_cache_paths(
                seed=seed,
            )
        )

        print(
            f"seed={seed}, "
            f"selected_embedding_model="
            f"{matrices.selected_embedding_model}, "
            f"train_oof_shape="
            f"{matrices.train_oof.shape}, "
            f"validation_shape="
            f"{matrices.validation.shape}"
        )

        print(
            f"seed={seed}, "
            f"train_oof_fingerprint="
            f"{array_fingerprint(matrices.train_oof)}, "
            f"validation_fingerprint="
            f"{array_fingerprint(matrices.validation)}"
        )

        print(
            f"seed={seed}, "
            f"matrices_path="
            f"{matrices_path}"
        )

        print(
            f"seed={seed}, "
            f"metadata_path="
            f"{metadata_path}"
        )

    print(
        "All base view caches are valid"
    )


if __name__ == "__main__":
    main()