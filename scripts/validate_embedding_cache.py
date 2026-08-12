import argparse
import json

import numpy as np

from pun_detection.config import (
    EMBEDDING_MODELS,
    EMBEDDINGS,
)
from pun_detection.data import (
    load_train_split,
    load_validation_split,
)
from pun_detection.embedding_cache import (
    get_embedding_cache_dir,
    load_embedding_cache,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(
            EMBEDDING_MODELS.keys()
        ),
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=sorted(
            EMBEDDINGS.allowed_splits
        ),
    )

    return parser.parse_args()


def load_split(
    split_name: str,
):
    if split_name == "train":
        return load_train_split()

    if split_name == "validation":
        return load_validation_split()

    raise ValueError(
        f"Unsupported split: {split_name}"
    )


def main():
    args = parse_args()

    dataframe = load_split(
        args.split
    )

    embeddings = load_embedding_cache(
        model_name=args.model,
        split_name=args.split,
        dataframe=dataframe,
    )

    cache_dir = get_embedding_cache_dir(
        args.model,
        args.split,
    )

    metadata_path = (
        cache_dir
        / "metadata.json"
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    tokenization = metadata[
        "tokenization"
    ]

    print(
        f"model={args.model}, "
        f"split={args.split}, "
        f"shape={embeddings.shape}, "
        f"dtype={embeddings.dtype}"
    )

    print(
        f"norm_min={norms.min():.6f}, "
        f"norm_max={norms.max():.6f}"
    )

    print(
        f"maximum_tokens="
        f"{tokenization['maximum_tokens']}, "
        f"max_seq_length="
        f"{tokenization['max_seq_length']}, "
        f"truncated_instances="
        f"{tokenization['truncated_instances']}"
    )

    print(
        "dataset_fingerprint="
        f"{metadata['dataset_fingerprint']}"
    )

    print(
        "embeddings_fingerprint="
        f"{metadata['embeddings_fingerprint']}"
    )

    print(
        "Embedding cache validation is valid"
    )


if __name__ == "__main__":
    main()